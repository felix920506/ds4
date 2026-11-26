/* Model-free checks for the coordinator's session-release frames.
 *
 * A coordinator session owns KV only once a stage has seen a WORK frame for
 * it, and a worker used to keep that KV until the control link dropped.  The
 * release frame is what lets ordinary session churn reuse a worker's
 * --max-sessions budget.  This test drives the real coordinator broadcast and
 * the real worker frame handler over socketpairs, with placeholder sessions so
 * no model or GPU is needed; it also pins the widened session-id sequence,
 * that the coordinator's wait for a missing stage can be interrupted,
 * which registration a failed route forgets, and when a decode batch has
 * spent its wait for a missing stage.
 */
#include "../ds4_distributed.c"

#include <assert.h>

static ds4_dist_worker_session *placeholder_session(uint64_t id) {
    ds4_dist_worker_session *entry = calloc(1, sizeof(*entry));
    assert(entry);
    entry->session_id = id;
    /* The release path hands this to ds4_session_free(), which accepts NULL;
     * framing and bookkeeping are what this test checks. */
    entry->session = NULL;
    return entry;
}

static void expect_release(int fd, ds4_dist_worker_state *state) {
    uint32_t type = 0, bytes = 0;
    assert(dist_read_frame_header(fd, &type, &bytes, NULL, 0) == 1);
    assert(type == DS4_DIST_MSG_SESSION_RELEASE);
    assert(bytes == sizeof(ds4_dist_session_release_fixed));
    assert(dist_worker_read_session_release(state, fd, bytes) == 1);
}

static void test_session_id_sequence(uint64_t first) {
    /* The low word is a process-wide monotonic counter.  Its predecessor was
     * 16 bits wide, so this also catches the wrap at 65,536. */
    for (uint32_t i = 1; i <= 70000; i++) {
        const uint64_t id = dist_make_session_id(NULL);
        assert((uint32_t)id == (uint32_t)(first + i));
    }
}

static void test_release_broadcast(uint64_t id, uint64_t other_id) {
    int worker_a[2], worker_b[2];
    assert(socketpair(AF_UNIX, SOCK_STREAM, 0, worker_a) == 0);
    assert(socketpair(AF_UNIX, SOCK_STREAM, 0, worker_b) == 0);

    ds4_dist_coordinator *coordinator = calloc(1, sizeof(*coordinator));
    assert(coordinator);
    ds4_dist_coordinator_state *coord = &coordinator->state;
    coord->model_id = 17;
    pthread_mutex_init(&coord->mu, NULL);
    ds4_dist_worker_entry *entry_a = calloc(1, sizeof(*entry_a));
    ds4_dist_worker_entry *entry_b = calloc(1, sizeof(*entry_b));
    assert(entry_a && entry_b);
    entry_a->fd = worker_a[0];
    entry_b->fd = worker_b[0];
    entry_a->next = entry_b;
    coord->workers = entry_a;

    ds4_dist_worker_state state_a;
    ds4_dist_worker_state state_b;
    memset(&state_a, 0, sizeof(state_a));
    memset(&state_b, 0, sizeof(state_b));
    pthread_mutex_init(&state_a.mu, NULL);
    pthread_mutex_init(&state_b.mu, NULL);
    state_a.model_id = 17;
    state_b.model_id = 17;
    state_a.max_sessions = 1;
    state_b.max_sessions = 1;
    state_a.sessions = placeholder_session(id);
    state_a.session_count = 1;
    state_b.sessions = placeholder_session(other_id);
    state_b.session_count = 1;

    /* Freeing the coordinator session is the hook users actually hit; it must
     * broadcast the release to every registered stage. */
    ds4_dist_session *session = calloc(1, sizeof(*session));
    assert(session);
    session->coord = coordinator;
    session->session_id = id;
    ds4_dist_session_free(session);

    expect_release(worker_a[1], &state_a);
    expect_release(worker_b[1], &state_b);

    /* The stage that held the released id dropped it; the unrelated one is
     * untouched. */
    assert(state_a.sessions == NULL && state_a.session_count == 0);
    assert(state_b.sessions != NULL && state_b.session_count == 1);
    /* Both stages remember every released id, so a late frame cannot recreate
     * the session on a stage that had never allocated it. */
    assert(dist_worker_session_released(&state_a, id));
    assert(dist_worker_session_released(&state_b, id));

    char err[128] = {0};
    assert(dist_worker_get_session_locked(&state_a, id, err, sizeof(err)) == NULL);
    assert(err[0] != '\0');

    /* A repeated release is harmless, which matters because a stage that never
     * created the session still receives it. */
    dist_coordinator_release_session(coord, id);
    expect_release(worker_a[1], &state_a);
    expect_release(worker_b[1], &state_b);
    assert(state_a.sessions == NULL && state_a.session_count == 0);
    assert(state_b.sessions != NULL && state_b.session_count == 1);

    free(state_a.sessions);
    free(state_b.sessions);
    dist_worker_entry_free(entry_a);
    dist_worker_entry_free(entry_b);
    free(coordinator);
    pthread_mutex_destroy(&state_a.mu);
    pthread_mutex_destroy(&state_b.mu);
    close(worker_a[0]);
    close(worker_a[1]);
    close(worker_b[0]);
    close(worker_b[1]);
}

/* The worker's control link must accept the frame.  Per-session data
 * connections never see a release, so either control-loop variant is fine. */
static void test_worker_control_loop(int (*loop)(ds4_dist_worker_state *, int),
                                     uint64_t id) {
    int pair[2];
    assert(socketpair(AF_UNIX, SOCK_STREAM, 0, pair) == 0);

    ds4_dist_worker_state state;
    memset(&state, 0, sizeof(state));
    pthread_mutex_init(&state.mu, NULL);
    state.model_id = 17;
    state.max_sessions = 1;
    state.sessions = placeholder_session(id);
    state.session_count = 1;

    ds4_dist_session_release_fixed release = {
        .model_id = 17,
    };
    dist_u64_to_halves(id, &release.session_hi, &release.session_lo);
    dist_session_release_to_wire(&release);
    uint8_t frame[DS4_DIST_RELEASE_FRAME_BYTES];
    dist_encode_session_release(frame, &release);
    assert(dist_write_full(pair[0], frame, sizeof(frame)) == 0);
    shutdown(pair[0], SHUT_WR);

    assert(loop(&state, pair[1]) == 0);
    assert(state.sessions == NULL && state.session_count == 0);
    assert(dist_worker_session_released(&state, id));

    pthread_mutex_destroy(&state.mu);
    close(pair[0]);
    close(pair[1]);
}

/* A stage whose control socket is momentarily full must keep the release and
 * get it later, not lose it and not have its (live) link closed.  Fill the
 * socket buffer first so the non-blocking flush cannot complete, then drain the
 * far end and check the accept-thread retry delivers the frame. */
static void test_release_backpressure(uint64_t id) {
    int pair[2];
    assert(socketpair(AF_UNIX, SOCK_STREAM, 0, pair) == 0);
    /* Small buffers make "full" quick and deterministic. */
    int small = 4096;
    assert(setsockopt(pair[0], SOL_SOCKET, SO_SNDBUF, &small, sizeof(small)) == 0);

    ds4_dist_coordinator *coordinator = calloc(1, sizeof(*coordinator));
    assert(coordinator);
    ds4_dist_coordinator_state *coord = &coordinator->state;
    coord->model_id = 17;
    pthread_mutex_init(&coord->mu, NULL);
    ds4_dist_worker_entry *entry = calloc(1, sizeof(*entry));
    assert(entry);
    entry->fd = pair[0];
    coord->workers = entry;

    /* Fill one byte at a time: stopping on EAGAIN for a single byte proves the
     * sender has no room at all, so the 24-byte frame cannot slip out and the
     * queued path is exercised deterministically. */
    const char one = 'x';
    for (;;) {
        ssize_t n = send(pair[0], &one, 1, DS4_DIST_SEND_NONBLOCK);
        if (n < 0) {
            assert(errno == EAGAIN || errno == EWOULDBLOCK);
            break;
        }
        assert(n == 1);
    }

    dist_coordinator_release_session(coord, id);
    /* Nothing could go out, so the frame is queued for this worker and the
     * link is still open. */
    assert(entry->release_out_len > entry->release_out_off);
    assert(entry->release_out_len == DS4_DIST_RELEASE_FRAME_BYTES);
    assert(dist_coordinator_releases_pending(coord, pair[0]));

    /* Draining the far end frees the window, but only the coordinator's
     * control thread delivers queued bytes; the frame is still pending until
     * it runs. */
    char drain[4096];
    for (;;) {
        ssize_t n = recv(pair[1], drain, sizeof(drain), MSG_DONTWAIT);
        if (n <= 0) break;
    }
    assert(dist_coordinator_releases_pending(coord, pair[0]));
    dist_coordinator_flush_releases(coord, pair[0]);
    assert(!dist_coordinator_releases_pending(coord, pair[0]));

    ds4_dist_worker_state state;
    memset(&state, 0, sizeof(state));
    pthread_mutex_init(&state.mu, NULL);
    state.model_id = 17;
    state.sessions = placeholder_session(id);
    state.session_count = 1;
    expect_release(pair[1], &state);
    assert(state.sessions == NULL && state.session_count == 0);

    free(state.sessions);
    pthread_mutex_destroy(&state.mu);
    dist_worker_entry_free(entry);
    free(coordinator);
    close(pair[0]);
    close(pair[1]);
}

/* A request waiting for a missing stage holds the server's inference lock, so
 * the wait has to end when the request is cancelled, and a KV save (shutdown,
 * eviction) must not wait at all: the missing stage took its shard along. */
static int cancel_polls;
static bool cancel_on_third_poll(void *ud) {
    (void)ud;
    return ++cancel_polls >= 3;
}

static void test_route_wait_interrupt(void) {
    ds4_dist_coordinator_state state;
    memset(&state, 0, sizeof(state));
    pthread_mutex_init(&state.mu, NULL);
    state.n_layers = 4;
    state.local_end = 1;  /* no worker registered for layers 2..3 */

    ds4_dist_route_plan plan;
    char err[128] = {0};
    const double t0 = dist_now_sec();
    assert(dist_coordinator_ensure_route(&state, &plan, NULL, cancel_on_third_poll, NULL,
                                         true, err, sizeof(err)) ==
           DS4_SESSION_SYNC_INTERRUPTED);
    assert(cancel_polls == 3);
    assert(dist_now_sec() - t0 < 5.0);
    assert(strstr(err, "interrupted") != NULL);

    cancel_polls = 0;
    err[0] = '\0';
    assert(dist_coordinator_ensure_route(&state, &plan, NULL, cancel_on_third_poll, NULL,
                                         false, err, sizeof(err)) == 1);
    assert(cancel_polls == 0);
    assert(strstr(err, "missing layer 2") != NULL);
    pthread_mutex_destroy(&state.mu);
}

/* A worker that registers during the same poll sleep in which the request is
 * cancelled must not win: the caller would go on to a full replay for a
 * client that has left. */
static ds4_dist_coordinator_state *late_state;
static ds4_dist_worker_entry late_worker;
static int late_polls;
static bool register_then_cancel(void *ud) {
    (void)ud;
    if (++late_polls == 1) {
        pthread_mutex_lock(&late_state->mu);
        late_state->workers = &late_worker;
        late_state->generation++;
        pthread_mutex_unlock(&late_state->mu);
        return false;
    }
    return true;
}

static void test_route_wait_cancel_beats_registration(void) {
    int link[2];
    assert(socketpair(AF_UNIX, SOCK_STREAM, 0, link) == 0);
    ds4_dist_coordinator_state state;
    memset(&state, 0, sizeof(state));
    pthread_mutex_init(&state.mu, NULL);
    state.n_layers = 4;
    state.local_end = 1;
    /* The first hop is a dup() of the control link: no connect needed. */
    state.use_control_for_work = true;
    memset(&late_worker, 0, sizeof(late_worker));
    late_worker.fd = link[0];
    late_worker.layer_start = 2;
    late_worker.layer_end = 3;
    late_worker.has_output = 1;
    snprintf(late_worker.peer_host, sizeof(late_worker.peer_host), "127.0.0.1");
    late_worker.listen_port = 9;
    late_state = &state;

    ds4_dist_route_plan plan;
    uint64_t generation = 99;
    char err[128] = {0};
    assert(dist_coordinator_ensure_route(&state, &plan, &generation, register_then_cancel,
                                         NULL, true, err, sizeof(err)) ==
           DS4_SESSION_SYNC_INTERRUPTED);
    assert(late_polls == 2);
    assert(plan.count == 0 && plan.entry == NULL);
    assert(generation == 0);
    assert(strstr(err, "interrupted") != NULL);

    /* Without a cancel the same registry gives a route. */
    assert(dist_coordinator_ensure_route(&state, &plan, &generation, NULL, NULL, true,
                                         err, sizeof(err)) == 0);
    assert(plan.count == 1 && plan.entry[0].fd >= 0);
    dist_route_plan_free(&plan);
    pthread_mutex_destroy(&state.mu);
    close(link[0]);
    close(link[1]);
}

/* A failed route forgets the workers it was built from, matched by endpoint.
 * A worker that registered after the plan was built has come back, possibly
 * already holding a session another batch member just replayed, and must be
 * left alone even at the same endpoint.  The workers register and the plans
 * are built through the coordinator's own functions, so the generation a
 * registration records is the one a plan reports. */
static void register_test_worker(ds4_dist_coordinator_state *state) {
    ds4_dist_hello_fixed hello;
    memset(&hello, 0, sizeof(hello));
    hello.layer_start = 2;
    hello.layer_end = 3;
    hello.has_output = 1;
    hello.n_layers = 4;
    hello.listen_port = 9912;
    dist_coordinator_add_worker(state, -1, "10.0.0.2", "40000", &hello, "test");
}

static void test_forget_only_older_registrations(void) {
    ds4_dist_coordinator_state state;
    memset(&state, 0, sizeof(state));
    pthread_mutex_init(&state.mu, NULL);
    state.n_layers = 4;
    state.local_end = 1;  /* the worker serves layers 2..3 and the output */

    register_test_worker(&state);
    assert(state.workers && state.generation == 1);
    assert(state.workers->registered_generation == 1);
    ds4_dist_route_plan plan;
    uint64_t plan_generation = 0;
    char err[128] = {0};
    assert(dist_coordinator_build_route_plan(&state, &plan, &plan_generation, NULL,
                                             err, sizeof(err)));
    assert(plan.count == 1 && plan_generation == 1);

    /* The worker dials back in at the same endpoint: the stale entry is
     * replaced, and the new one is not the one the plan was built from. */
    register_test_worker(&state);
    ds4_dist_worker_entry *back = state.workers;
    assert(back && !back->next && state.generation == 2);
    assert(back->registered_generation == 2);
    dist_coordinator_forget_route_workers(&state, &plan, plan_generation);
    assert(state.workers == back && state.generation == 2);
    dist_route_plan_free(&plan);

    /* A plan built from the new registration does forget it. */
    assert(dist_coordinator_build_route_plan(&state, &plan, &plan_generation, NULL,
                                             err, sizeof(err)));
    assert(plan_generation == 2);
    dist_coordinator_forget_route_workers(&state, &plan, plan_generation);
    assert(state.workers == NULL && state.generation == 3);
    dist_route_plan_free(&plan);
    pthread_mutex_destroy(&state.mu);
}

/* A decode batch waits for a missing stage once: a member's failure spends
 * that wait only if its own route wait ran out, not when it failed at once
 * (a worker-side error, a dead socket, an interrupted wait). */
static void test_batch_route_wait_spent(void) {
    ds4_dist_coordinator coord;
    memset(&coord, 0, sizeof(coord));
    pthread_mutex_init(&coord.state.mu, NULL);
    coord.state.n_layers = 4;
    coord.state.local_end = 1;
    ds4_dist_session d;
    memset(&d, 0, sizeof(d));
    d.coord = &coord;

    const double waited = dist_now_sec() - DS4_DIST_ROUTE_WAIT_SEC;
    assert(!dist_batch_route_wait_spent(&d, dist_now_sec()));
    assert(dist_batch_route_wait_spent(&d, waited));

    register_test_worker(&coord.state);
    assert(!dist_batch_route_wait_spent(&d, waited));

    dist_worker_entry_free(coord.state.workers);
    pthread_mutex_destroy(&coord.state.mu);
}

int main(void) {
    const uint64_t first = dist_make_session_id(NULL);
    test_session_id_sequence(first);
    test_release_broadcast(UINT64_C(0xabc000000001),
                           UINT64_C(0xabc000000002));
    test_worker_control_loop(dist_worker_read_loop, UINT64_C(0xabc000000003));
    test_worker_control_loop(dist_worker_read_loop_prefetch,
                             UINT64_C(0xabc000000004));
    test_release_backpressure(UINT64_C(0xabc000000005));
    test_route_wait_interrupt();
    test_route_wait_cancel_beats_registration();
    test_forget_only_older_registrations();
    test_batch_route_wait_spent();
    fprintf(stderr, "test_dist_session_release PASS\n");
    return 0;
}
