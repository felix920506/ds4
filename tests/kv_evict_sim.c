/* Disk KV cache eviction simulator.
 *
 * Drives the real ds4_kvstore eviction and text-prefix lookup with simulated
 * conversations, so a change to the eviction score can be judged by what it
 * costs: the tokens a server would have to prefill again.  No model is loaded.
 * Checkpoints are sparse files with a real header, and time() is replaced by a
 * simulated clock, so days of traffic against a 200 GiB budget run in seconds
 * and use no disk space.
 *
 * The server model follows ds4_server.c: one conversation per slot, a "cold"
 * checkpoint at the shared system prompt, "continued" frontiers on aligned
 * boundaries during prefill, and an "evict" checkpoint when a slot's live state
 * is replaced.  A live mismatch is a request whose prompt does not extend the
 * slot's state (the client rewrote the tail); the state saved then is a dead
 * branch that no later prompt extends.
 *
 * Usage: kv_evict_sim [SCENARIO...]   (default: all)
 *
 * Environment:
 *   KV_EVICT_SIM_SEED=N       workload seed (default 21)
 *   KV_EVICT_SIM_BUDGET_SCALE=F  multiply every scenario's disk budget
 *   KV_EVICT_SIM_UNLIMITED=1  no disk budget: the prefill no policy can avoid
 *   KV_EVICT_SIM_VERBOSE=1    print every request, store, hit and eviction
 */

#include "../ds4_kvstore.h"

#include <dirent.h>
#include <fcntl.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

/* ---- simulated clock ------------------------------------------------------ */

static uint64_t sim_now = 1000ull * 86400ull;

time_t time(time_t *out) {
    if (out) *out = (time_t)sim_now;
    return (time_t)sim_now;
}

/* ---- deterministic randomness -------------------------------------------- */

static uint64_t rng_state;

static uint64_t rng_next(void) {
    uint64_t z = (rng_state += 0x9e3779b97f4a7c15ull);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
    return z ^ (z >> 31);
}

static double rng_unit(void) {
    return (double)(rng_next() >> 11) / 9007199254740992.0;
}

static int rng_range(int lo, int hi) {
    return lo + (int)(rng_unit() * (double)(hi - lo + 1));
}

/* ---- model ---------------------------------------------------------------- */

#define SIM_BYTES_PER_TOKEN 4
#define SIM_CTX 1048576
#define SIM_QUANT 2
#define SIM_MAX_CONVS 1024
#define SIM_MAX_SLOTS 4
/* Checkpoint size measured on gfx1151 with the Q2 Flash model: 157.34 MiB at
 * 10240 tokens and 1367.35 MiB at 102400. */
#define SIM_FILE_FIXED_BYTES (23ull << 20)
#define SIM_FILE_BYTES_PER_TOKEN 13768ull
#define SIM_PREFILL_TOKENS_PER_SEC 200.0
#define SIM_DEAD_TAIL_TOKENS 64

typedef struct {
    int app;        /* conversations of one app share a system prompt */
    int sys_tokens;
    int parent;     /* conversation this one was forked from, or -1 */
    int fork;       /* tokens shared with the parent */
    int len;        /* tokens the conversation has reached */
    /* How often the client alters the model's last reply before sending the
     * next turn, as one that reformats or trims replies does every time.  A
     * tip saved before that never matches the conversation again. */
    double rewrite;
    char *text;     /* rendered text, grown on demand */
    int text_tokens;
} sim_conv;

typedef struct {
    int conv;       /* -1 when empty */
    int live;       /* tokens held in RAM */
    int continued_last;
    uint64_t used;
} sim_slot;

typedef struct {
    uint64_t requests, prompt_tokens, live_tokens, disk_tokens, prefill_tokens;
    uint64_t stores, store_bytes, evictions;
    uint64_t disk_hits, disk_misses;
    /* Tokens prefilled only because the checkpoint that covered them had been
     * evicted, split by how recently that checkpoint had been used. */
    uint64_t lost_tokens, lost_fresh_tokens;
} sim_stats;

/* Every checkpoint ever written, to tell what a lookup could have found. */
typedef struct {
    char sha[41];
    int conv;       /* -1: text no conversation extends (dead branch, stale) */
    int tokens;
    uint64_t used;  /* last store, refresh or hit */
    bool on_disk;
} sim_record;

typedef struct {
    ds4_kvstore kc;
    char dir[64];
    sim_conv conv[SIM_MAX_CONVS];
    int convs;
    sim_slot slot[SIM_MAX_SLOTS];
    int slots;
    int dead_serial;
    sim_record *rec;
    int recs, rec_cap;
    bool verbose;
    sim_stats st;
} sim;

static uint32_t mix32(uint64_t x) {
    x ^= x >> 33; x *= 0xff51afd7ed558ccdull;
    x ^= x >> 33; x *= 0xc4ceb9fe1a85ec53ull;
    x ^= x >> 33;
    return (uint32_t)x;
}

static void fill_text(char *dst, uint64_t stream, int first_token, int tokens) {
    for (int t = 0; t < tokens; t++) {
        uint32_t v = mix32((stream << 32) ^ (uint64_t)(first_token + t));
        for (int b = 0; b < SIM_BYTES_PER_TOKEN; b++)
            dst[(size_t)t * SIM_BYTES_PER_TOKEN + b] = (char)('a' + ((v >> (b * 5)) & 15));
    }
}

/* Which text stream token `t` of a conversation comes from: the app's system
 * prompt, an ancestor it was forked from, or its own. */
static uint64_t conv_stream(const sim *s, int ci, int t) {
    if (t < s->conv[ci].sys_tokens) return 1000000ull + (uint64_t)s->conv[ci].app;
    while (s->conv[ci].parent >= 0 && t < s->conv[ci].fork) ci = s->conv[ci].parent;
    return (uint64_t)ci;
}

/* Tokens two conversations have in common at the front. */
static int conv_shared(const sim *s, int a, int b) {
    if (a == b) return INT32_MAX;
    int cut[2 * 64 + 3], n = 0;
    cut[n++] = 0;
    cut[n++] = s->conv[a].sys_tokens;
    cut[n++] = s->conv[b].sys_tokens;
    for (int c = a; c >= 0 && n < 64; c = s->conv[c].parent) cut[n++] = s->conv[c].fork;
    for (int c = b; c >= 0 && n < 2 * 64; c = s->conv[c].parent) cut[n++] = s->conv[c].fork;
    for (int i = 1; i < n; i++)
        for (int j = i; j > 0 && cut[j] < cut[j - 1]; j--) {
            int t = cut[j]; cut[j] = cut[j - 1]; cut[j - 1] = t;
        }
    for (int i = 0; i < n; i++)
        if (conv_stream(s, a, cut[i]) != conv_stream(s, b, cut[i])) return cut[i];
    return INT32_MAX;
}

/* Text of the first `tokens` tokens of a conversation, NUL terminated. */
static const char *conv_text(sim *s, int ci, int tokens) {
    sim_conv *c = &s->conv[ci];
    if (tokens > c->text_tokens) {
        c->text = realloc(c->text, (size_t)tokens * SIM_BYTES_PER_TOKEN + 1);
        for (int t = c->text_tokens; t < tokens; t++)
            fill_text(c->text + (size_t)t * SIM_BYTES_PER_TOKEN,
                      conv_stream(s, ci, t), t, 1);
        c->text_tokens = tokens;
    }
    static char *buf;
    static size_t cap;
    size_t n = (size_t)tokens * SIM_BYTES_PER_TOKEN;
    if (n + 1 > cap) {
        cap = n + 1;
        buf = realloc(buf, cap);
    }
    memcpy(buf, c->text, n);
    buf[n] = '\0';
    return buf;
}

static uint64_t file_bytes_for(int tokens) {
    return SIM_FILE_FIXED_BYTES + (uint64_t)tokens * SIM_FILE_BYTES_PER_TOKEN;
}

static void entry_path(const sim *s, const char sha[41], char *out, size_t n) {
    snprintf(out, n, "%s/%.40s.kv", s->dir, sha);
}

/* `text` may be NULL to rewrite the header of an existing file. */
static void write_entry(const char *path, uint8_t reason, uint32_t tokens,
                        uint32_t hits, uint64_t created, uint64_t used,
                        const char *text, uint32_t text_bytes,
                        uint64_t file_bytes) {
    uint8_t h[DS4_KVSTORE_FIXED_HEADER];
    uint64_t payload = file_bytes - DS4_KVSTORE_FIXED_HEADER - 4u - text_bytes;
    ds4_kvstore_fill_header(h, 0, SIM_QUANT, reason, 0, tokens, hits, SIM_CTX,
                            created, used, payload);
    uint8_t tb[4];
    ds4_kvstore_le_put32(tb, text_bytes);
    int fd = open(path, O_WRONLY | O_CREAT, 0644);
    if (fd < 0) { perror(path); exit(1); }
    if (write(fd, h, sizeof(h)) != (ssize_t)sizeof(h) ||
        write(fd, tb, sizeof(tb)) != (ssize_t)sizeof(tb) ||
        (text && write(fd, text, text_bytes) != (ssize_t)text_bytes) ||
        ftruncate(fd, (off_t)file_bytes) != 0) {
        perror(path);
        exit(1);
    }
    close(fd);
}

static void sim_log(void *ud, ds4_kvstore_log_type type, const char *msg) {
    (void)type;
    sim *s = ud;
    if (!strstr(msg, "kv cache evicted")) return;
    s->st.evictions++;
    if (s->verbose) printf("    %s\n", strstr(msg, "kv cache evicted"));
}

static sim_record *record_find(sim *s, const char sha[41]) {
    for (int i = 0; i < s->recs; i++)
        if (!memcmp(s->rec[i].sha, sha, 40)) return &s->rec[i];
    return NULL;
}

static void record_sync(sim *s) {
    for (int i = 0; i < s->recs; i++) {
        char path[160];
        if (!s->rec[i].on_disk) continue;
        entry_path(s, s->rec[i].sha, path, sizeof(path));
        if (access(path, F_OK) != 0) s->rec[i].on_disk = false;
    }
}

static void record_store(sim *s, const char sha[41], int conv, int tokens) {
    sim_record *r = record_find(s, sha);
    if (!r) {
        if (s->recs == s->rec_cap)
            s->rec = realloc(s->rec, (size_t)(s->rec_cap = s->rec_cap ? s->rec_cap * 2 : 256) * sizeof(*s->rec));
        r = &s->rec[s->recs++];
        memcpy(r->sha, sha, 41);
    }
    r->conv = conv;
    r->tokens = tokens;
    r->used = sim_now;
    r->on_disk = true;
}

/* `conv` is the conversation whose text this is a prefix of, or -1. */
static void store(sim *s, const char *text, int tokens, uint8_t reason, int conv) {
    if (tokens < s->kc.opt.min_tokens) return;
    const size_t text_len = (size_t)tokens * SIM_BYTES_PER_TOKEN;
    char sha[41], path[160];
    ds4_kvstore_sha1_bytes_hex(text, text_len, sha);
    entry_path(s, sha, path, sizeof(path));
    const uint64_t bytes = file_bytes_for(tokens);

    ds4_kvstore_entry old = {0};
    if (ds4_kvstore_read_entry_file(path, sha, &old)) {
        /* Same prefix already on disk: the server refreshes it in place. */
        write_entry(path, old.reason, old.tokens, old.hits, old.created_at,
                    sim_now, NULL, (uint32_t)text_len, old.file_size);
        ds4_kvstore_entry_free(&old);
        record_store(s, sha, conv, tokens);
        return;
    }
    if (bytes > s->kc.budget_bytes) return;

    ds4_kvstore_eviction_context incoming = {
        .text = text, .text_len = text_len, .model_id = 0,
        .quant_bits = SIM_QUANT, .ctx_size = SIM_CTX,
        .reject_different_quant = false,
        .reason = reason,
    };
    ds4_kvstore_evict(&s->kc, NULL, bytes, &incoming);
    record_sync(s);

    write_entry(path, reason, (uint32_t)tokens, 0, sim_now, sim_now, text,
                (uint32_t)text_len, bytes);
    record_store(s, sha, conv, tokens);
    s->st.stores++;
    s->st.store_bytes += bytes;
    if (s->verbose)
        printf("    stored tokens=%d reason=%u size=%.0f MiB\n", tokens, reason,
               (double)bytes / 1048576.0);
}

/* The state a slot holds, saved because the slot is being reused. */
static void store_slot(sim *s, sim_slot *sl, bool dead_branch) {
    if (sl->conv < 0 || sl->live < s->kc.opt.min_tokens) return;
    const char *text = conv_text(s, sl->conv, sl->live);
    if (!dead_branch) {
        store(s, text, sl->live, DS4_KVSTORE_REASON_EVICT, sl->conv);
        return;
    }
    /* The tail the client later rewrote: unique bytes nothing will extend. */
    char *copy = malloc((size_t)sl->live * SIM_BYTES_PER_TOKEN + 1);
    memcpy(copy, text, (size_t)sl->live * SIM_BYTES_PER_TOKEN + 1);
    int tail = sl->live < SIM_DEAD_TAIL_TOKENS ? sl->live : SIM_DEAD_TAIL_TOKENS;
    fill_text(copy + (size_t)(sl->live - tail) * SIM_BYTES_PER_TOKEN,
              2000000ull + (uint64_t)s->dead_serial++, 0, tail);
    store(s, copy, sl->live, DS4_KVSTORE_REASON_EVICT, -1);
    free(copy);
}

static int disk_lookup(sim *s, int ci, int len, const char *prompt) {
    int idx = ds4_kvstore_find_text_prefix(&s->kc, prompt, 0, SIM_QUANT, SIM_CTX);
    int found = idx < 0 ? 0 : (int)s->kc.entry[idx].tokens;
    /* The longest checkpoint ever written that this prompt extends.  Another
     * conversation shares its system prompt, or the turns before a fork. */
    const sim_record *best = NULL;
    for (int i = 0; i < s->recs; i++) {
        const sim_record *r = &s->rec[i];
        if (r->conv < 0 || r->tokens > len || r->tokens <= found) continue;
        bool extends = r->tokens <= conv_shared(s, ci, r->conv);
        if (extends && (!best || r->tokens > best->tokens)) best = r;
    }
    if (best) {
        uint64_t lost = (uint64_t)(best->tokens - found);
        s->st.lost_tokens += lost;
        if (sim_now - best->used < 3600) s->st.lost_fresh_tokens += lost;
        if (s->verbose) {
            printf("    LOST %d tokens: evicted checkpoint tokens=%d last used %llus ago\n",
                   best->tokens - found, best->tokens,
                   (unsigned long long)(sim_now - best->used));
            /* What the cache held instead. */
            for (int i = 0; i < s->kc.len; i++) {
                const ds4_kvstore_entry *e = &s->kc.entry[i];
                const sim_record *r = record_find(s, e->sha);
                printf("      kept conv=%d tokens=%u reason=%u hits=%u idle=%llus\n",
                       r ? r->conv : -2, e->tokens, e->reason, e->hits,
                       (unsigned long long)(sim_now - (e->last_used ? e->last_used : e->created_at)));
            }
        }
    }
    if (idx < 0) {
        s->st.disk_misses++;
        return 0;
    }
    ds4_kvstore_entry *e = &s->kc.entry[idx];
    int tokens = (int)e->tokens;
    ds4_kvstore_touch_file(e->path, e->hits + 1);
    sim_record *hit = record_find(s, e->sha);
    if (hit) hit->used = sim_now;
    s->st.disk_hits++;
    if (s->verbose) printf("    hit tokens=%d hits=%u\n", tokens, e->hits + 1);
    return tokens;
}

static int continued_step(const sim *s) {
    int step = s->kc.opt.continued_interval_tokens;
    int align = s->kc.opt.boundary_align_tokens;
    if (step <= 0) return 0;
    if (align > 0) step = ((step + align - 1) / align) * align;
    return step;
}

/* One request: conversation `ci` asks with a prompt of `len` tokens. */
static void request(sim *s, int ci, int len, bool mismatch) {
    sim_conv *c = &s->conv[ci];
    if (len > c->len) c->len = len;
    sim_slot *sl = NULL;
    for (int i = 0; i < s->slots; i++)
        if (s->slot[i].conv == ci) sl = &s->slot[i];
    int cached = 0;
    bool from_live = false;
    if (sl && c->rewrite > 0.0 && rng_unit() < c->rewrite) mismatch = true;
    if (sl && !mismatch && sl->live <= len) {
        cached = sl->live;
        from_live = true;
    } else {
        if (!sl) {
            sl = &s->slot[0];
            for (int i = 1; i < s->slots; i++) {
                if (sl->conv < 0) break;
                if (s->slot[i].conv < 0 || s->slot[i].used < sl->used)
                    sl = &s->slot[i];
            }
            const double rewrite = sl->conv < 0 ? 0.0 : s->conv[sl->conv].rewrite;
            store_slot(s, sl, rewrite > 0.0 && rng_unit() < rewrite);
        } else {
            store_slot(s, sl, true);
        }
        sl->conv = ci;
        sl->continued_last = 0;
        cached = disk_lookup(s, ci, len, conv_text(s, ci, len));
        sl->live = cached;
    }
    if (s->verbose)
        printf("  req conv=%d prompt=%d cached=%d (%s)\n", ci, len, cached,
               from_live ? "live" : "disk");

    s->st.requests++;
    s->st.prompt_tokens += (uint64_t)len;
    if (from_live) s->st.live_tokens += (uint64_t)cached;
    else s->st.disk_tokens += (uint64_t)cached;
    s->st.prefill_tokens += (uint64_t)(len - cached);

    /* Prefill, writing the cold anchor and the aligned frontiers on the way. */
    int cold = 0;
    if (cached == 0 && s->kc.opt.cold_max_tokens > 0 &&
        len <= s->kc.opt.cold_max_tokens) {
        cold = c->sys_tokens >= s->kc.opt.min_tokens && c->sys_tokens < len
            ? c->sys_tokens : ds4_kvstore_store_len(&s->kc, len);
    }
    const int step = continued_step(s);
    int pos = cached;
    while (pos < len) {
        int next = len;
        if (cold > pos && cold < next) next = cold;
        if (step > 0) {
            int frontier = (pos / step + 1) * step;
            if (frontier < next) next = frontier;
        }
        sim_now += (uint64_t)((double)(next - pos) / SIM_PREFILL_TOKENS_PER_SEC) + 1;
        pos = next;
        sl->live = pos;
        if (pos == cold) {
            store(s, conv_text(s, ci, pos), pos, DS4_KVSTORE_REASON_COLD, ci);
            sl->continued_last = pos;
        } else if (step > 0 && pos % step == 0 && pos > sl->continued_last) {
            store(s, conv_text(s, ci, pos), pos, DS4_KVSTORE_REASON_CONTINUED, ci);
            sl->continued_last = pos;
        }
    }
    sl->live = len;
    sl->used = sim_now;
}

/* The server process dies: live state is lost without a shutdown save. */
static void crash(sim *s) {
    for (int i = 0; i < s->slots; i++) {
        s->slot[i].conv = -1;
        s->slot[i].live = 0;
        s->slot[i].continued_last = 0;
    }
}

static int add_conv(sim *s, int app, int sys_tokens) {
    sim_conv *c = &s->conv[s->convs];
    memset(c, 0, sizeof(*c));
    c->app = app;
    c->sys_tokens = sys_tokens;
    c->parent = -1;
    if (s->convs + 1 >= SIM_MAX_CONVS) { fprintf(stderr, "too many conversations\n"); exit(1); }
    return s->convs++;
}

/* A conversation whose first `fork` tokens are another one's: an edited or
 * regenerated turn. */
static int add_fork(sim *s, int parent, int fork) {
    int ci = add_conv(s, s->conv[parent].app, s->conv[parent].sys_tokens);
    s->conv[ci].parent = parent;
    s->conv[ci].fork = fork;
    return ci;
}

/* Old checkpoints from conversations that never return. */
static void add_stale(sim *s, int count, int min_tokens, int max_tokens,
                      double min_days, double max_days) {
    for (int i = 0; i < count; i++) {
        /* Skewed to the short end, as a real directory is. */
        double u = rng_unit();
        int tokens = min_tokens + (int)((double)(max_tokens - min_tokens) * u * u);
        size_t n = (size_t)tokens * SIM_BYTES_PER_TOKEN;
        char *text = malloc(n + 1);
        fill_text(text, 3000000ull + rng_next() % 1000000ull, 0, tokens);
        text[n] = '\0';
        char sha[41], path[160];
        ds4_kvstore_sha1_bytes_hex(text, n, sha);
        entry_path(s, sha, path, sizeof(path));
        uint64_t age = (uint64_t)((min_days + rng_unit() * (max_days - min_days)) * 86400.0);
        static const uint8_t reasons[] = {
            DS4_KVSTORE_REASON_COLD, DS4_KVSTORE_REASON_CONTINUED,
            DS4_KVSTORE_REASON_CONTINUED, DS4_KVSTORE_REASON_EVICT,
            DS4_KVSTORE_REASON_SHUTDOWN,
        };
        write_entry(path, reasons[rng_next() % 5], (uint32_t)tokens,
                    (uint32_t)rng_range(0, 20), sim_now - age - 3600,
                    sim_now - age, text, (uint32_t)n, file_bytes_for(tokens));
        free(text);
    }
}

static void sim_open(sim *s, uint64_t budget_mib, int slots, int interval,
                     int align, int min_tokens, int cold_max) {
    memset(s, 0, sizeof(*s));
    snprintf(s->dir, sizeof(s->dir), "/tmp/ds4-kv-evict-sim.XXXXXX");
    if (!mkdtemp(s->dir)) { perror("mkdtemp"); exit(1); }
    s->kc.enabled = true;
    s->kc.dir = strdup(s->dir);
    s->kc.budget_bytes = budget_mib << 20;
    const char *scale = getenv("KV_EVICT_SIM_BUDGET_SCALE");
    if (scale) s->kc.budget_bytes = (uint64_t)((double)s->kc.budget_bytes * atof(scale));
    if (getenv("KV_EVICT_SIM_UNLIMITED")) s->kc.budget_bytes = UINT64_MAX / 2;
    s->kc.opt.min_tokens = min_tokens;
    s->kc.opt.cold_max_tokens = cold_max;
    s->kc.opt.continued_interval_tokens = interval;
    s->kc.opt.boundary_trim_tokens = 32;
    s->kc.opt.boundary_align_tokens = align;
    s->kc.log = sim_log;
    s->kc.log_ud = s;
    s->kc.log_name = "sim";
    s->slots = slots;
    for (int i = 0; i < slots; i++) s->slot[i].conv = -1;
    s->verbose = getenv("KV_EVICT_SIM_VERBOSE") != NULL;
    const char *seed = getenv("KV_EVICT_SIM_SEED");
    rng_state = seed ? strtoull(seed, NULL, 10) : 21;
    sim_now = 1000ull * 86400ull;
}

static void sim_close(sim *s, const char *name) {
    DIR *d = opendir(s->dir);
    struct dirent *de;
    uint64_t files = 0, bytes = 0;
    while ((de = readdir(d)) != NULL) {
        char sha[41], path[160];
        if (!ds4_kvstore_sha_hex_name(de->d_name, sha)) continue;
        entry_path(s, sha, path, sizeof(path));
        struct stat st;
        if (stat(path, &st) == 0) {
            files++;
            bytes += (uint64_t)st.st_size;
        }
        unlink(path);
    }
    closedir(d);
    rmdir(s->dir);
    const sim_stats *t = &s->st;
    /* prefill: tokens computed again.  lost: the part of it a checkpoint that
     * was evicted would have covered; fresh: that checkpoint had been used in
     * the last hour. */
    if (name)
        printf("%-12s prefill=%9llu tok (%4.1f%%)  lost=%8llu (fresh %7llu)  "
               "stores=%5llu (%4.0f GiB)  evicted=%5llu  hits=%llu/%llu  "
               "left=%llu files %.0f GiB\n",
               name, (unsigned long long)t->prefill_tokens,
               t->prompt_tokens ? 100.0 * (double)t->prefill_tokens / (double)t->prompt_tokens : 0.0,
               (unsigned long long)t->lost_tokens, (unsigned long long)t->lost_fresh_tokens,
               (unsigned long long)t->stores, (double)t->store_bytes / 1073741824.0,
               (unsigned long long)t->evictions,
               (unsigned long long)t->disk_hits,
               (unsigned long long)(t->disk_hits + t->disk_misses),
               (unsigned long long)files, (double)bytes / 1073741824.0);
    for (int i = 0; i < s->convs; i++) free(s->conv[i].text);
    free(s->rec);
    ds4_kvstore_close(&s->kc);
}

/* ---- scenarios ------------------------------------------------------------ */

/* Issue #21: a 200 GiB budget already full of checkpoints idle for weeks.  Two
 * long conversations take turns; the server is restarted now and then, so each
 * has to come back from disk. */
static void scenario_issue21(void) {
    sim s;
    sim_open(&s, 204800, 2, 5120, 512, 256, 1048576);
    add_stale(&s, 330, 5000, 250000, 1.0, 96.0);
    ds4_kvstore_evict(&s.kc, NULL, 0, NULL);
    s.st = (sim_stats){0};
    int a = add_conv(&s, 0, 6000), b = add_conv(&s, 1, 4000);
    int la = 110000, lb = 40000;
    for (int round = 0; round < 24; round++) {
        if (round % 4 == 0) crash(&s);
        request(&s, a, la, false);
        sim_now += 240;
        request(&s, b, lb, false);
        sim_now += 600;
        la += rng_range(2000, 8000);
        lb += rng_range(1000, 5000);
    }
    sim_close(&s, "issue21");
}

/* The end-to-end run on gfx1151: a budget that holds about nine checkpoints,
 * every turn a live mismatch, then the first conversation again. */
static void scenario_tiny(void) {
    sim s;
    sim_open(&s, 1300, 1, 2048, 2048, 512, 30000);
    add_stale(&s, 8, 8192, 12500, 60.0, 60.0);
    ds4_kvstore_evict(&s.kc, NULL, 0, NULL);
    s.st = (sim_stats){0};
    int x = add_conv(&s, 0, 0), y = add_conv(&s, 1, 0);
    for (int pass = 0; pass < 2; pass++) {
        for (int t = 1; t <= 6; t++) { request(&s, x, t * 2190, true); sim_now += 3; }
        for (int t = 1; t <= 3; t++) { request(&s, y, t * 2190, true); sim_now += 3; }
    }
    sim_close(&s, "tiny");
}

/* One slot shared by many chat conversations on two system prompts.  Popular
 * conversations return often, the rest are abandoned; the budget holds only a
 * few of them. */
static void scenario_chat(double mismatch_rate, const char *name) {
    sim s;
    sim_open(&s, 16384, 1, 10000, 2048, 512, 30000);
    int active[8], n = 0;
    for (int i = 0; i < 8; i++) active[n++] = add_conv(&s, i % 2, 6000);
    int len[SIM_MAX_CONVS] = {0};
    for (int turn = 0; turn < 900; turn++) {
        /* Skewed choice: low indexes are the conversations people stay in. */
        double u = rng_unit();
        int k = (int)((double)n * u * u);
        int ci = active[k];
        if (len[ci] == 0) len[ci] = s.conv[ci].sys_tokens + rng_range(300, 2000);
        else len[ci] += rng_range(400, 4000);
        request(&s, ci, len[ci], rng_unit() < mismatch_rate);
        sim_now += (uint64_t)rng_range(20, 400);
        if (turn % 150 == 149) sim_now += 14 * 3600;  /* overnight */
        if (len[ci] > 90000 || rng_unit() < 0.01) {
            /* Finished or abandoned: a new conversation takes its place. */
            active[k] = add_conv(&s, k % 2, 6000);
        }
    }
    sim_close(&s, name);
}

/* Long-context agents: a 200 GiB budget with nothing stale, six conversations
 * of 100k to 300k tokens on two slots, occasional restarts. */
static void scenario_long(void) {
    sim s;
    sim_open(&s, 204800, 2, 5120, 512, 256, 1048576);
    int active[6], len[SIM_MAX_CONVS] = {0};
    for (int i = 0; i < 6; i++) {
        active[i] = add_conv(&s, i % 3, 8000);
        len[active[i]] = rng_range(60000, 200000);
    }
    for (int turn = 0; turn < 300; turn++) {
        double u = rng_unit();
        int k = (int)(6.0 * u * u), ci = active[k];
        len[ci] += rng_range(1500, 9000);
        if (len[ci] > 320000) {
            ci = active[k] = add_conv(&s, k % 3, 8000);
            len[ci] = rng_range(20000, 60000);
        }
        if (turn % 60 == 59) crash(&s);
        request(&s, ci, len[ci], rng_unit() < 0.05);
        sim_now += (uint64_t)rng_range(30, 900);
        if (turn % 100 == 99) sim_now += 12 * 3600;
    }
    sim_close(&s, "long");
}

/* A system-prompt anchor that every conversation of an app starts from, left
 * idle over a weekend while another app fills the cache with one-off prompts.
 * On Monday the first app's conversations start again. */
static void scenario_weekend(void) {
    sim s;
    sim_open(&s, 8192, 1, 10000, 2048, 512, 30000);
    for (int i = 0; i < 20; i++) {
        int ci = add_conv(&s, 0, 12000);
        request(&s, ci, 12000 + rng_range(500, 3000), false);
        sim_now += 900;
    }
    sim_now += 36 * 3600;
    for (int i = 0; i < 12; i++) {
        int ci = add_conv(&s, 100 + i, 0);
        request(&s, ci, rng_range(15000, 28000), false);
        sim_now += 3600;
    }
    s.st = (sim_stats){0};
    for (int i = 0; i < 10; i++) {
        int ci = add_conv(&s, 0, 12000);
        request(&s, ci, 12000 + rng_range(500, 3000), false);
        sim_now += 900;
    }
    sim_close(&s, "weekend");
}

/* A busy single slot: eight conversations of up to 30k tokens, a request every
 * few seconds, a tenth of them with an edited reply, a crash every 60 turns,
 * and a budget that holds about six tips.  Everything in the cache is minutes
 * old, so only an ordering that separates minutes keeps the right ones.  This
 * is the shape of the real-model soak the policy was checked against. */
static void scenario_busy(const char *name, int rewriting_of_4, double rewrite) {
    sim s;
    sim_open(&s, 2500, 1, 4096, 2048, 512, 30000);
    int active[8], len[SIM_MAX_CONVS] = {0};
    for (int i = 0; i < 8; i++) {
        active[i] = add_conv(&s, i % 2, 1900);
        if (i % 4 < rewriting_of_4) s.conv[active[i]].rewrite = rewrite;
    }
    for (int turn = 0; turn < 600; turn++) {
        double u = rng_unit();
        int k = (int)(8.0 * u * u), ci = active[k];
        if (len[ci] > 30000) {
            ci = active[k] = add_conv(&s, k % 2, 1900);
            if (k % 4 < rewriting_of_4) s.conv[ci].rewrite = rewrite;
        }
        if (len[ci] == 0) len[ci] = 1900;
        len[ci] += rng_range(450, 2400);
        if (turn % 60 == 59) crash(&s);
        request(&s, ci, len[ci], rng_unit() < 0.1);
        sim_now += (uint64_t)rng_range(1, 4);
    }
    sim_close(&s, name);
}

/* Chat where a quarter of the turns edit or regenerate an earlier message, so
 * the prompt forks off somewhere in the last third of the conversation.  Only
 * a checkpoint from before the fork helps. */
static void scenario_branch(void) {
    sim s;
    sim_open(&s, 16384, 1, 10000, 2048, 512, 30000);
    int active[6], len[SIM_MAX_CONVS] = {0};
    for (int i = 0; i < 6; i++) active[i] = add_conv(&s, i % 2, 6000);
    for (int turn = 0; turn < 700; turn++) {
        double u = rng_unit();
        int k = (int)(6.0 * u * u), ci = active[k];
        if (len[ci] == 0) {
            len[ci] = 6000 + rng_range(300, 2000);
        } else if (len[ci] > 12000 && rng_unit() < 0.25) {
            int fork = len[ci] - (int)((double)(len[ci] - 6000) * rng_unit() / 3.0);
            int child = add_fork(&s, ci, fork);
            len[child] = fork + rng_range(200, 2000);
            ci = active[k] = child;
        } else {
            len[ci] += rng_range(400, 4000);
        }
        request(&s, ci, len[ci], false);
        sim_now += (uint64_t)rng_range(20, 400);
        if (turn % 150 == 149) sim_now += 14 * 3600;
        if (len[ci] > 90000) active[k] = add_conv(&s, k % 2, 6000);
    }
    sim_close(&s, "branch");
}

/* Coding agents: a conversation grows to its compaction limit, then restarts
 * as a summary on the same system prompt.  Three agents share two slots and
 * the server crashes now and then. */
static void scenario_agent(void) {
    sim s;
    sim_open(&s, 32768, 2, 10000, 2048, 512, 30000);
    int active[3], len[SIM_MAX_CONVS] = {0}, limit[3];
    for (int i = 0; i < 3; i++) {
        active[i] = add_conv(&s, 0, 14000);
        limit[i] = rng_range(120000, 200000);
    }
    for (int turn = 0; turn < 1200; turn++) {
        int k = (int)(3.0 * rng_unit()), ci = active[k];
        if (len[ci] == 0) len[ci] = 14000 + rng_range(500, 9000);
        else len[ci] += rng_range(300, 6000);
        if (len[ci] > limit[k]) {
            ci = active[k] = add_conv(&s, 0, 14000);
            len[ci] = 14000 + rng_range(4000, 12000);
        }
        if (turn % 200 == 199) crash(&s);
        request(&s, ci, len[ci], rng_unit() < 0.02);
        sim_now += (uint64_t)rng_range(5, 120);
        if (turn % 300 == 299) sim_now += 10 * 3600;
    }
    sim_close(&s, "agent");
}

/* Many short conversations on three system prompts, a 2 GiB budget. */
static void scenario_small(void) {
    sim s;
    sim_open(&s, 2048, 1, 10000, 2048, 512, 30000);
    for (int i = 0; i < 400; i++) {
        int ci = add_conv(&s, (int)(3.0 * rng_unit() * rng_unit()), 3000);
        int len = 3000, turns = rng_range(1, 3);
        for (int t = 0; t < turns; t++) {
            len += rng_range(500, 3000);
            request(&s, ci, len, false);
            sim_now += (uint64_t)rng_range(10, 200);
            if (rng_unit() < 0.3) {
                /* Someone else's conversation takes the slot in between. */
                int other = rng_range(0, ci);
                if (s.conv[other].len) request(&s, other, s.conv[other].len + rng_range(200, 1500), false);
            }
        }
        if (i % 80 == 79) sim_now += 12 * 3600;
    }
    sim_close(&s, "small");
}

/* Ten conversations on four slots: most switches are served from RAM, and the
 * disk holds what the slots kicked out. */
static void scenario_slots(void) {
    sim s;
    sim_open(&s, 24576, 4, 10000, 2048, 512, 30000);
    int active[10], len[SIM_MAX_CONVS] = {0};
    for (int i = 0; i < 10; i++) active[i] = add_conv(&s, i % 3, 8000);
    for (int turn = 0; turn < 1500; turn++) {
        double u = rng_unit();
        int k = (int)(10.0 * u * u), ci = active[k];
        if (len[ci] == 0) len[ci] = 8000 + rng_range(500, 4000);
        else len[ci] += rng_range(300, 3000);
        if (len[ci] > 250000) {
            ci = active[k] = add_conv(&s, k % 3, 8000);
            len[ci] = 8000 + rng_range(500, 4000);
        }
        if (turn % 400 == 399) crash(&s);
        request(&s, ci, len[ci], rng_unit() < 0.05);
        sim_now += (uint64_t)rng_range(5, 200);
        if (turn % 250 == 249) sim_now += 9 * 3600;
    }
    sim_close(&s, "slots");
}

/* Issue #17's shape: a cache full of checkpoints from the last day, and one
 * 200k-token conversation that loses its live state every third turn and has
 * to prefill again, with another conversation saving in between. */
static void scenario_reprefill(void) {
    sim s;
    sim_open(&s, 204800, 2, 5120, 512, 256, 1048576);
    add_stale(&s, 330, 5000, 250000, 0.02, 1.0);
    ds4_kvstore_evict(&s.kc, NULL, 0, NULL);
    s.st = (sim_stats){0};
    int a = add_conv(&s, 0, 6000), la = 110000;
    int others[4], lo[4];
    for (int i = 0; i < 4; i++) { others[i] = add_conv(&s, 1 + i, 4000); lo[i] = rng_range(20000, 90000); }
    for (int round = 0; round < 30; round++) {
        if (round % 3 == 0) crash(&s);
        request(&s, a, la, false);
        sim_now += 300;
        int o = rng_range(0, 3);
        lo[o] += rng_range(1000, 6000);
        request(&s, others[o], lo[o], false);
        sim_now += 600;
        la += rng_range(2000, 8000);
    }
    sim_close(&s, "reprefill");
}

/* Cost of one eviction pass on a directory the size of the one in issue #21. */
static void scenario_bench(void) {
    sim s;
    sim_open(&s, 204800, 1, 5120, 512, 256, 1048576);
    add_stale(&s, 330, 5000, 250000, 1.0, 96.0);
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    ds4_kvstore_evict(&s.kc, NULL, 0, NULL);
    clock_gettime(CLOCK_MONOTONIC, &t1);
    double first = (double)(t1.tv_sec - t0.tv_sec) * 1e3 + (double)(t1.tv_nsec - t0.tv_nsec) / 1e6;
    int files = s.kc.len;
    int ci = add_conv(&s, 0, 6000);
    const int passes = 20;
    double total = 0.0;
    for (int i = 0; i < passes; i++) {
        int tokens = 10240 * (i + 1);
        const char *text = conv_text(&s, ci, tokens);
        ds4_kvstore_eviction_context incoming = {
            .text = text, .text_len = (size_t)tokens * SIM_BYTES_PER_TOKEN,
            .quant_bits = SIM_QUANT, .ctx_size = SIM_CTX,
        };
        clock_gettime(CLOCK_MONOTONIC, &t0);
        ds4_kvstore_evict(&s.kc, NULL, file_bytes_for(tokens), &incoming);
        clock_gettime(CLOCK_MONOTONIC, &t1);
        total += (double)(t1.tv_sec - t0.tv_sec) * 1e3 + (double)(t1.tv_nsec - t0.tv_nsec) / 1e6;
        store(&s, text, tokens, DS4_KVSTORE_REASON_CONTINUED, ci);
    }
    printf("bench        %d files: first pass %.1f ms, then %.1f ms per pass before a store\n",
           files, first, total / passes);
    sim_close(&s, NULL);
}

int main(int argc, char **argv) {
    static const char *all[] = {"issue21", "tiny", "chat", "chat-rewrite", "long", "weekend",
                                "branch", "agent", "small", "slots", "reprefill", "busy",
                                "reformat-some", "reformat-all", "edit-often"};
    int n = argc > 1 ? argc - 1 : (int)(sizeof(all) / sizeof(all[0]));
    for (int i = 0; i < n; i++) {
        const char *name = argc > 1 ? argv[i + 1] : all[i];
        if (!strcmp(name, "issue21")) scenario_issue21();
        else if (!strcmp(name, "tiny")) scenario_tiny();
        else if (!strcmp(name, "chat")) scenario_chat(0.0, "chat");
        else if (!strcmp(name, "chat-rewrite")) scenario_chat(0.3, "chat-rewrite");
        else if (!strcmp(name, "long")) scenario_long();
        else if (!strcmp(name, "weekend")) scenario_weekend();
        else if (!strcmp(name, "branch")) scenario_branch();
        else if (!strcmp(name, "agent")) scenario_agent();
        else if (!strcmp(name, "small")) scenario_small();
        else if (!strcmp(name, "slots")) scenario_slots();
        else if (!strcmp(name, "reprefill")) scenario_reprefill();
        else if (!strcmp(name, "busy")) scenario_busy("busy", 0, 0.0);
        /* The busy server again, with clients that never send a reply back
         * the way the model wrote it: one conversation in four, then all. */
        else if (!strcmp(name, "reformat-some")) scenario_busy("reformat-some", 1, 1.0);
        else if (!strcmp(name, "reformat-all")) scenario_busy("reformat-all", 4, 1.0);
        else if (!strcmp(name, "edit-often")) scenario_busy("edit-often", 4, 0.3);
        else if (!strcmp(name, "bench")) scenario_bench();
        else { fprintf(stderr, "unknown scenario %s\n", name); return 2; }
    }
    return 0;
}
