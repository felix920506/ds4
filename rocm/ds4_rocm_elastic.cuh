/*
 * Elastic routed-expert weights (resident DeepSeek V4 on unified memory).
 *
 * Routed MoE experts are ~90% of the resident model, yet a token uses only
 * n_expert_used of the n_expert in each layer, and the bytes are a read-only
 * copy of the GGUF.  So when the KV cache needs memory that is not free, cold
 * experts can simply be dropped and read from the file again on demand.
 * Attention, shared experts and every other weight are used by every token
 * and never take part in this.
 *
 * Each layer owns one device block of K expert slots laid out like the GGUF
 * tensors, [gate x K | up x K | down x K].  A full layer has K = n_expert
 * with slot i holding expert i, which is exactly the resident table: the
 * launcher uses it as is and nothing below costs anything.  Under memory
 * pressure a layer is shrunk to a smaller block that keeps its most recently
 * used experts:
 *
 *   decode (1 token)  the selected experts are looked up in the block, misses
 *                     are read from disk over the least recently used slots,
 *                     and the resident kernels run on the block with expert
 *                     ids replaced by slot ids.  A hit costs no copy, no I/O.
 *   batch (prefill)   the unique experts of the batch are gathered at their
 *                     home positions in one table-shaped scratch block shared
 *                     by all shrunk layers (cached slots by device copy, the
 *                     rest from disk) and the resident kernels run on it with
 *                     the original ids.  Batches do not evict slots: one
 *                     batch can touch every expert of a layer and would flush
 *                     the set decode depends on.
 *
 * Both paths hand the unmodified resident kernels the same weight bytes, so
 * eviction changes where the bytes live and how long a token takes, never the
 * result.
 *
 * Policy.  Routing is only mildly skewed (about 80 of 256 experts cover 90%
 * of a layer's decode selections, 180 cover 99%; the first, hash-routed
 * layers are uniform) and a miss costs milliseconds of SSD time, so what
 * matters is evicting the coldest experts of the whole model first rather
 * than whole layers.  Every layer counts how often each expert is selected:
 * in batches, in decode on shrunk layers, and in every
 * DS4_ROCM_ELASTIC_SAMPLE-th decode launch on full layers, so the counts
 * exist before the first eviction.  Reclaim shrinks, one ladder step at a
 * time, the layer whose coldest slots carry the smallest share of its
 * selections.  Layers grow back, smallest first, once free memory exceeds
 * the streaming reserve again.
 */

enum {
    /* Smallest block a layer is shrunk to.  Must be >= DS4_ROCM_N_EXPERT_USED. */
    DS4_ROCM_ELASTIC_MIN_SLOTS = 32u,
    /* Slots dropped per shrink step while a layer is above twice the minimum. */
    DS4_ROCM_ELASTIC_STEP_SLOTS = 64u,
    DS4_ROCM_ELASTIC_SAMPLE = 8u,
    DS4_ROCM_ELASTIC_READ_CHUNK = 32u << 20
};

struct cuda_elastic_layer {
    ds4_gpu_stream_expert_table t;
    uint64_t expert_bytes;      /* 2 * gate + down; 0 marks an unused entry */
    char *block;                /* slots expert slots, see the layout above */
    uint32_t slots;
    uint32_t launches;          /* decode launches, for sampling full layers */
    int32_t slot_expert[DS4_ROCM_MAX_N_EXPERT];     /* slot -> expert or -1 */
    int16_t expert_slot[DS4_ROCM_MAX_N_EXPERT];     /* expert -> slot or -1 */
    uint64_t slot_used[DS4_ROCM_MAX_N_EXPERT];      /* slot -> last use */
    uint32_t expert_hits[DS4_ROCM_MAX_N_EXPERT];    /* expert -> selections seen */
    uint64_t hits;                                  /* sum of expert_hits */
};

struct cuda_elastic_stats {
    uint64_t shrinks;
    uint64_t grows;
    uint64_t decode_hits;
    uint64_t decode_misses;
    uint64_t batch_cached;
    uint64_t batch_reads;
    uint64_t read_bytes;
    double read_sec;
};

static std::vector<cuda_elastic_layer> g_elastic_layers;    /* by layer index */
static char *g_elastic_scratch;
static uint64_t g_elastic_scratch_bytes;
static int64_t g_elastic_scratch_layer = -1;
static uint8_t g_elastic_scratch_valid[DS4_ROCM_MAX_N_EXPERT];
static int32_t *g_elastic_slot_ids;
static ds4_gpu_tensor g_elastic_slot_tensor;
static uint64_t g_elastic_clock;
static double g_elastic_grow_checked;
static cuda_elastic_stats g_elastic_stats;

static uint64_t cuda_elastic_block_bytes(const cuda_elastic_layer &l, uint32_t slots) {
    return (uint64_t)slots * l.expert_bytes;
}

static int cuda_elastic_full(const cuda_elastic_layer &l) {
    return l.slots == l.t.n_total_expert;
}

/* The three sections of a block with the given slot count. */
struct cuda_elastic_view {
    char *gate;
    char *up;
    char *down;
};

static cuda_elastic_view cuda_elastic_view_of(
        const cuda_elastic_layer &l,
        char *block,
        uint32_t slots) {
    cuda_elastic_view v;
    v.gate = block;
    v.up = v.gate + (uint64_t)slots * l.t.gate_expert_bytes;
    v.down = v.up + (uint64_t)slots * l.t.gate_expert_bytes;
    return v;
}

static void cuda_elastic_stats_print(void) {
    const cuda_elastic_stats &s = g_elastic_stats;
    fprintf(stderr,
            DS4_GPU_LOG_PREFIX "elastic experts: shrinks=%llu grows=%llu "
            "decode hits=%llu misses=%llu batch cached=%llu read=%llu "
            "ssd=%.2f GiB in %.2fs\n",
            (unsigned long long)s.shrinks,
            (unsigned long long)s.grows,
            (unsigned long long)s.decode_hits,
            (unsigned long long)s.decode_misses,
            (unsigned long long)s.batch_cached,
            (unsigned long long)s.batch_reads,
            (double)s.read_bytes / 1073741824.0,
            s.read_sec);
}

/* Diagnostic: DS4_ROCM_ELASTIC_STATS prints the counters at exit and a line
 * per reclaim. */
static int cuda_elastic_stats_on(void) {
    static int on = -1;
    if (on < 0) {
        on = getenv("DS4_ROCM_ELASTIC_STATS") != NULL;
        if (on) atexit(cuda_elastic_stats_print);
    }
    return on;
}

static int cuda_elastic_read_jobs(std::vector<cuda_stream_read_job> &jobs) {
    int ok = 1;
    const double t0 = cuda_wall_sec();
    /* The reader pool takes a bounded job array per call. */
    for (size_t at = 0; ok && at < jobs.size(); at += DS4_ROCM_STREAM_READ_MAX_JOBS) {
        const uint32_t n = (uint32_t)std::min<size_t>(jobs.size() - at,
                                                      DS4_ROCM_STREAM_READ_MAX_JOBS);
        ok = cuda_stream_read_jobs_parallel(jobs.data() + at, n);
        cuda_stream_read_jobs_free(jobs.data() + at, n);
    }
    g_elastic_stats.read_sec += cuda_wall_sec() - t0;
    for (const cuda_stream_read_job &job : jobs) {
        if (ok) g_elastic_stats.read_bytes += job.bytes;
    }
    jobs.clear();
    return ok;
}

/* Queue the reads of one expert's gate, up and down rows into slot of view. */
static void cuda_elastic_expert_jobs(
        const cuda_elastic_layer &l,
        uint32_t expert,
        const cuda_elastic_view &v,
        uint32_t slot,
        std::vector<cuda_stream_read_job> &jobs) {
    const uint64_t gate_eb = l.t.gate_expert_bytes;
    const uint64_t down_eb = l.t.down_expert_bytes;
    jobs.push_back({v.gate + slot * gate_eb, l.t.gate_offset + expert * gate_eb,
                    gate_eb, NULL, NULL, 0, 0});
    jobs.push_back({v.up + slot * gate_eb, l.t.up_offset + expert * gate_eb,
                    gate_eb, NULL, NULL, 0, 0});
    jobs.push_back({v.down + slot * down_eb, l.t.down_offset + expert * down_eb,
                    down_eb, NULL, NULL, 0, 0});
}

/* Queue device copies of one expert from a slot of one view to another. */
static int cuda_elastic_copy_expert(
        const cuda_elastic_layer &l,
        const cuda_elastic_view &dst,
        uint32_t dst_slot,
        const cuda_elastic_view &src,
        uint32_t src_slot) {
    const uint64_t gate_eb = l.t.gate_expert_bytes;
    const uint64_t down_eb = l.t.down_expert_bytes;
    return cuda_ok(cudaMemcpyAsync(dst.gate + dst_slot * gate_eb,
                                   src.gate + src_slot * gate_eb,
                                   (size_t)gate_eb, cudaMemcpyDeviceToDevice, 0),
                   "elastic expert gate copy") &&
           cuda_ok(cudaMemcpyAsync(dst.up + dst_slot * gate_eb,
                                   src.up + src_slot * gate_eb,
                                   (size_t)gate_eb, cudaMemcpyDeviceToDevice, 0),
                   "elastic expert up copy") &&
           cuda_ok(cudaMemcpyAsync(dst.down + dst_slot * down_eb,
                                   src.down + src_slot * down_eb,
                                   (size_t)down_eb, cudaMemcpyDeviceToDevice, 0),
                   "elastic expert down copy");
}

static void cuda_elastic_release_all(void) {
    for (cuda_elastic_layer &l : g_elastic_layers) {
        if (l.block) (void)cudaFree(l.block);
    }
    g_elastic_layers.clear();
    if (g_elastic_scratch) (void)cudaFree(g_elastic_scratch);
    g_elastic_scratch = NULL;
    g_elastic_scratch_bytes = 0;
    g_elastic_scratch_layer = -1;
    if (g_elastic_slot_ids) (void)cudaFree(g_elastic_slot_ids);
    g_elastic_slot_ids = NULL;
}

static uint64_t cuda_elastic_resident_bytes(void) {
    uint64_t bytes = g_elastic_scratch_bytes;
    for (const cuda_elastic_layer &l : g_elastic_layers) {
        if (l.block) bytes += cuda_elastic_block_bytes(l, l.slots);
    }
    return bytes;
}

static cuda_elastic_layer *cuda_elastic_find(
        const void *model_map,
        uint32_t layer,
        uint32_t n_total_expert,
        uint64_t gate_offset,
        uint64_t up_offset,
        uint64_t down_offset,
        uint64_t gate_expert_bytes,
        uint64_t down_expert_bytes) {
    if (layer >= g_elastic_layers.size()) return NULL;
    cuda_elastic_layer &l = g_elastic_layers[layer];
    if (l.expert_bytes == 0 ||
        l.t.model_map != model_map ||
        l.t.n_total_expert != n_total_expert ||
        l.t.gate_offset != gate_offset ||
        l.t.up_offset != up_offset ||
        l.t.down_offset != down_offset ||
        l.t.gate_expert_bytes != gate_expert_bytes ||
        l.t.down_expert_bytes != down_expert_bytes) {
        return NULL;
    }
    return &l;
}

/* 1 when [offset, offset + bytes) lies inside an elastic expert tensor, so the
 * generic model range cache must not load a second copy of it. */
static int cuda_elastic_owns_range(const void *model_map, uint64_t offset, uint64_t bytes) {
    for (const cuda_elastic_layer &l : g_elastic_layers) {
        if (l.expert_bytes == 0 || l.t.model_map != model_map) continue;
        const uint64_t gate_bytes = (uint64_t)l.t.n_total_expert * l.t.gate_expert_bytes;
        const uint64_t down_bytes = (uint64_t)l.t.n_total_expert * l.t.down_expert_bytes;
        const uint64_t part[3][2] = {
            {l.t.gate_offset, gate_bytes},
            {l.t.up_offset, gate_bytes},
            {l.t.down_offset, down_bytes},
        };
        for (const auto &p : part) {
            if (offset >= p[0] && bytes <= p[1] && offset - p[0] <= p[1] - bytes) return 1;
        }
    }
    return 0;
}

/* Count selections of an expert.  Halving everything now and then keeps the
 * counts a picture of recent routing instead of the whole session. */
static void cuda_elastic_note_hits(cuda_elastic_layer &l, uint32_t expert, uint32_t n) {
    l.expert_hits[expert] += n;
    l.hits += n;
    if (l.hits < (1u << 22)) return;
    l.hits = 0;
    for (uint32_t e = 0; e < l.t.n_total_expert; e++) {
        l.expert_hits[e] >>= 1;
        l.hits += l.expert_hits[e];
    }
}

/* Slots of l ordered hottest expert first; empty slots sort last.  This is
 * the order a shrink keeps slots in. */
static void cuda_elastic_slots_by_heat(const cuda_elastic_layer &l, uint32_t *order) {
    for (uint32_t s = 0; s < l.slots; s++) order[s] = s;
    std::stable_sort(order, order + l.slots, [&l](uint32_t a, uint32_t b) {
        const int a_set = l.slot_expert[a] >= 0;
        const int b_set = l.slot_expert[b] >= 0;
        if (a_set != b_set) return a_set > b_set;
        if (!a_set) return false;
        const uint32_t a_hits = l.expert_hits[l.slot_expert[a]];
        const uint32_t b_hits = l.expert_hits[l.slot_expert[b]];
        if (a_hits != b_hits) return a_hits > b_hits;
        return l.slot_used[a] > l.slot_used[b];
    });
}

static void cuda_elastic_clear_slots(cuda_elastic_layer &l, uint32_t slots) {
    l.slots = slots;
    for (uint32_t i = 0; i < DS4_ROCM_MAX_N_EXPERT; i++) {
        l.slot_expert[i] = -1;
        l.expert_slot[i] = -1;
        l.slot_used[i] = 0;
    }
}

/*
 * Replace the block of l by one with new_slots slots that keeps the hottest
 * experts.  The caller has synchronized the device.  The new
 * block is allocated before the old one is freed, so the experts move by
 * device copy; when even that much memory is not free the experts are parked
 * in the scratch block across the free.  Growing to the full count restores
 * the resident layout, reading the experts that are not cached from disk.
 */
static int cuda_elastic_resize(cuda_elastic_layer &l, uint32_t new_slots) {
    const uint32_t n_total = l.t.n_total_expert;
    const int to_full = new_slots == n_total;
    uint32_t order[DS4_ROCM_MAX_N_EXPERT];
    cuda_elastic_slots_by_heat(l, order);
    uint32_t keep = 0;
    while (keep < l.slots && keep < new_slots && l.slot_expert[order[keep]] >= 0) keep++;

    char *old_block = l.block;
    const cuda_elastic_view old_view = cuda_elastic_view_of(l, old_block, l.slots);
    void *block = NULL;
    const uint64_t new_bytes = cuda_elastic_block_bytes(l, new_slots);
    int parked = 0;
    if (cudaMalloc(&block, (size_t)new_bytes) != cudaSuccess) {
        (void)cudaGetLastError();
        if (new_slots > l.slots || !g_elastic_scratch) return 0;
        const cuda_elastic_view park = cuda_elastic_view_of(l, g_elastic_scratch, new_slots);
        g_elastic_scratch_layer = -1;
        for (uint32_t i = 0; i < keep; i++) {
            if (!cuda_elastic_copy_expert(l, park, i, old_view, order[i])) return 0;
        }
        if (!cuda_ok(cudaDeviceSynchronize(), "elastic shrink park sync")) return 0;
        (void)cudaFree(old_block);
        old_block = NULL;
        l.block = NULL;
        parked = 1;
        if (!cuda_ok(cudaMalloc(&block, (size_t)new_bytes), "elastic expert block")) {
            /* The layer is unusable now; launches fail instead of misrouting. */
            cuda_elastic_clear_slots(l, 0);
            return 0;
        }
    }

    const cuda_elastic_view new_view = cuda_elastic_view_of(l, (char *)block, new_slots);
    const cuda_elastic_view src_view = parked ?
        cuda_elastic_view_of(l, g_elastic_scratch, new_slots) : old_view;
    cuda_elastic_layer next = l;
    cuda_elastic_clear_slots(next, new_slots);
    next.block = (char *)block;
    int ok = 1;
    for (uint32_t i = 0; ok && i < keep; i++) {
        const int32_t expert = l.slot_expert[order[i]];
        const uint32_t slot = to_full ? (uint32_t)expert : i;
        ok = cuda_elastic_copy_expert(l, new_view, slot, src_view, parked ? i : order[i]);
        next.slot_expert[slot] = expert;
        next.expert_slot[expert] = (int16_t)slot;
        next.slot_used[slot] = l.slot_used[order[i]];
    }
    std::vector<cuda_stream_read_job> jobs;
    for (uint32_t e = 0; ok && to_full && e < n_total; e++) {
        if (next.slot_expert[e] >= 0) continue;
        cuda_elastic_expert_jobs(l, e, new_view, e, jobs);
        next.slot_expert[e] = (int32_t)e;
        next.expert_slot[e] = (int16_t)e;
    }
    ok = ok && cuda_elastic_read_jobs(jobs) &&
         cuda_ok(cudaDeviceSynchronize(), "elastic resize sync");
    if (!ok && !parked) {
        (void)cudaFree(block);
        return 0;
    }
    if (old_block) (void)cudaFree(old_block);
    if (!ok) cuda_elastic_clear_slots(next, new_slots);
    l = next;
    return ok;
}

/* The slot count one step below (dir < 0) or above (dir > 0) the current one. */
static uint32_t cuda_elastic_next_slots(const cuda_elastic_layer &l, int dir) {
    const uint32_t min_slots = DS4_ROCM_ELASTIC_MIN_SLOTS;
    const uint32_t step = DS4_ROCM_ELASTIC_STEP_SLOTS;
    if (dir < 0) {
        if (l.slots <= min_slots) return l.slots;
        return l.slots > step + min_slots ? l.slots - step : min_slots;
    }
    if (l.slots >= l.t.n_total_expert) return l.slots;
    const uint32_t up = l.slots < step ? step : l.slots + step;
    return up < l.t.n_total_expert ? up : l.t.n_total_expert;
}

/*
 * Expected cost of the next shrink step of l: the share of the layer's
 * selections that went to the experts it would drop, per slot dropped.  That
 * share is the miss rate the step adds, so shrinking the cheapest layer first
 * approximates evicting the coldest experts of the whole model, and leaves
 * uniformly routed layers, which have no cold experts, for last.
 */
static double cuda_elastic_shrink_cost(const cuda_elastic_layer &l) {
    uint32_t order[DS4_ROCM_MAX_N_EXPERT];
    cuda_elastic_slots_by_heat(l, order);
    const uint32_t kept = cuda_elastic_next_slots(l, -1);
    uint64_t dropped_hits = 0;
    for (uint32_t i = kept; i < l.slots; i++) {
        if (l.slot_expert[order[i]] >= 0) dropped_hits += l.expert_hits[l.slot_expert[order[i]]];
    }
    return l.hits ? (double)dropped_hits / (double)l.hits / (double)(l.slots - kept) : 0.0;
}

/*
 * Shrunk layers gather batches in a table-shaped scratch block.  It is taken
 * while the first layer is shrunk: that layer's full block becomes the
 * scratch block and the layer moves into a new, smaller one, so the scratch
 * block never has to be found in memory that reclaim was called to relieve.
 */
static int cuda_elastic_take_scratch(cuda_elastic_layer &l) {
    const cuda_elastic_layer full = l;
    uint32_t slots = cuda_elastic_next_slots(l, -1);
    void *block = NULL;
    while (cudaMalloc(&block, (size_t)cuda_elastic_block_bytes(l, slots)) != cudaSuccess) {
        (void)cudaGetLastError();
        if (slots == DS4_ROCM_ELASTIC_MIN_SLOTS) {
            l = full;
            return 0;
        }
        l.slots = slots;
        slots = cuda_elastic_next_slots(l, -1);
    }
    uint32_t order[DS4_ROCM_MAX_N_EXPERT];
    cuda_elastic_slots_by_heat(full, order);
    const cuda_elastic_view src = cuda_elastic_view_of(full, full.block, full.slots);
    const cuda_elastic_view dst = cuda_elastic_view_of(full, (char *)block, slots);
    cuda_elastic_clear_slots(l, slots);
    l.block = (char *)block;
    int ok = 1;
    for (uint32_t i = 0; ok && i < slots; i++) {
        const uint32_t expert = order[i];   /* full layer: slot == expert */
        ok = cuda_elastic_copy_expert(full, dst, i, src, expert);
        l.slot_expert[i] = (int32_t)expert;
        l.expert_slot[expert] = (int16_t)i;
        l.slot_used[i] = full.slot_used[expert];
    }
    if (!ok || !cuda_ok(cudaDeviceSynchronize(), "elastic scratch sync")) {
        (void)cudaFree(block);
        l = full;
        return 0;
    }
    g_elastic_scratch = full.block;
    g_elastic_scratch_bytes = cuda_elastic_block_bytes(full, full.slots);
    g_elastic_scratch_layer = -1;
    return 1;
}

static int cuda_elastic_ensure_slot_ids(void) {
    if (g_elastic_slot_ids) return 1;
    void *ids = NULL;
    const size_t bytes = DS4_ROCM_N_EXPERT_USED * sizeof(int32_t);
    if (!cuda_ok(cudaMalloc(&ids, bytes), "elastic slot ids")) return 0;
    g_elastic_slot_ids = (int32_t *)ids;
    g_elastic_slot_tensor.ptr = ids;
    g_elastic_slot_tensor.bytes = bytes;
    g_elastic_slot_tensor.owner = 0;
    return 1;
}

/*
 * Give the smallest layer one step back when memory is plentiful again.  The
 * reserve that bounds streaming cache growth is the hysteresis: reclaim runs
 * near zero free memory, growth only above the reserve, so the two cannot
 * chase each other.  Checked at most once a second because the free-memory
 * query is a system call on the decode path.
 */
static void cuda_elastic_maybe_grow(void) {
    /* Diagnostic: keep forced evictions in place on a machine with free memory. */
    static const int disabled = getenv("DS4_ROCM_ELASTIC_NO_GROW") != NULL;
    const double now = cuda_wall_sec();
    if (disabled || now - g_elastic_grow_checked < 1.0) return;
    g_elastic_grow_checked = now;

    cuda_elastic_layer *smallest = NULL;
    for (cuda_elastic_layer &l : g_elastic_layers) {
        if (l.block && !cuda_elastic_full(l) &&
            (!smallest || l.slots < smallest->slots)) {
            smallest = &l;
        }
    }
    if (!smallest) return;
    const uint32_t slots = cuda_elastic_next_slots(*smallest, 1);
    size_t free_b = 0;
    size_t total_b = 0;
    if (cudaMemGetInfo(&free_b, &total_b) != cudaSuccess) {
        (void)cudaGetLastError();
        return;
    }
    const uint64_t reserve = cuda_stream_resident_free_reserve_bytes();
    if ((uint64_t)free_b < reserve ||
        cuda_elastic_block_bytes(*smallest, slots) > (uint64_t)free_b - reserve ||
        !cuda_ok(cudaDeviceSynchronize(), "elastic grow sync") ||
        !cuda_elastic_resize(*smallest, slots)) {
        return;
    }
    g_elastic_stats.grows++;
    for (const cuda_elastic_layer &l : g_elastic_layers) {
        if (l.block && !cuda_elastic_full(l)) return;
    }
    /* Every layer is whole again: nothing gathers into the scratch block. */
    (void)cudaFree(g_elastic_scratch);
    g_elastic_scratch = NULL;
    g_elastic_scratch_bytes = 0;
    g_elastic_scratch_layer = -1;
}

/* Copy the selected ids to the host.  The blocking copy and synchronize also
 * order this thread after every queued kernel, which is what makes it safe
 * to overwrite slots and the scratch block afterwards. */
static int cuda_elastic_read_selected(
        const ds4_gpu_tensor *selected,
        int32_t *ids,
        uint64_t n_ids,
        uint32_t n_total_expert) {
    if (!cuda_ok(cudaMemcpy(ids, selected->ptr, (size_t)n_ids * sizeof(ids[0]),
                            cudaMemcpyDeviceToHost),
                 "elastic selected ids readback") ||
        !cuda_ok(cudaDeviceSynchronize(), "elastic selected ids sync")) {
        return 0;
    }
    for (uint64_t i = 0; i < n_ids; i++) {
        if (ids[i] < 0 || (uint32_t)ids[i] >= n_total_expert) {
            fprintf(stderr,
                    DS4_GPU_LOG_PREFIX "elastic selected expert id %d outside 0..%u\n",
                    ids[i], n_total_expert);
            return 0;
        }
    }
    return 1;
}

/* A batch is a large sample of this context's routing: it is what tells
 * reclaim hot from cold experts before decode has produced any history. */
static void cuda_elastic_note_batch(
        cuda_elastic_layer &l,
        const int32_t *ids,
        uint64_t n_ids) {
    for (uint64_t i = 0; i < n_ids; i++) cuda_elastic_note_hits(l, (uint32_t)ids[i], 1u);
}

static int cuda_elastic_decode(
        cuda_elastic_layer &l,
        const ds4_gpu_tensor *selected,
        uint32_t n_expert,
        const ds4_gpu_tensor **selected_exec,
        const char **gate_w,
        const char **up_w,
        const char **down_w) {
    int32_t ids[DS4_ROCM_N_EXPERT_USED];
    int32_t slots[DS4_ROCM_N_EXPERT_USED];
    uint8_t pinned[DS4_ROCM_MAX_N_EXPERT] = {0};
    if (n_expert > DS4_ROCM_N_EXPERT_USED || n_expert > l.slots ||
        !cuda_elastic_ensure_slot_ids() ||
        !cuda_elastic_read_selected(selected, ids, n_expert, l.t.n_total_expert)) {
        return 0;
    }
    /* Pin the hits first so a miss cannot evict a slot this token reads. */
    for (uint32_t i = 0; i < n_expert; i++) {
        const int16_t slot = l.expert_slot[ids[i]];
        if (slot >= 0) pinned[slot] = 1;
    }
    const cuda_elastic_view view = cuda_elastic_view_of(l, l.block, l.slots);
    std::vector<cuda_stream_read_job> jobs;
    for (uint32_t i = 0; i < n_expert; i++) {
        int32_t slot = l.expert_slot[ids[i]];
        if (slot < 0) {
            uint64_t oldest = UINT64_MAX;
            for (uint32_t s = 0; s < l.slots; s++) {
                const uint64_t used = l.slot_expert[s] >= 0 ? l.slot_used[s] : 0;
                if (!pinned[s] && (slot < 0 || used < oldest)) {
                    oldest = used;
                    slot = (int32_t)s;
                }
            }
            if (slot < 0) return 0;
            if (l.slot_expert[slot] >= 0) l.expert_slot[l.slot_expert[slot]] = -1;
            l.slot_expert[slot] = ids[i];
            l.expert_slot[ids[i]] = (int16_t)slot;
            pinned[slot] = 1;
            cuda_elastic_expert_jobs(l, (uint32_t)ids[i], view, (uint32_t)slot, jobs);
            g_elastic_stats.decode_misses++;
        } else {
            g_elastic_stats.decode_hits++;
        }
        l.slot_used[slot] = g_elastic_clock;
        cuda_elastic_note_hits(l, (uint32_t)ids[i], 1u);
        slots[i] = slot;
    }
    if (!cuda_elastic_read_jobs(jobs) ||
        !cuda_ok(cudaMemcpy(g_elastic_slot_ids, slots, n_expert * sizeof(slots[0]),
                            cudaMemcpyHostToDevice),
                 "elastic slot ids upload")) {
        /* Slots claimed above may hold partial data: forget everything. */
        cuda_elastic_clear_slots(l, l.slots);
        return 0;
    }
    *selected_exec = &g_elastic_slot_tensor;
    *gate_w = view.gate;
    *up_w = view.up;
    *down_w = view.down;
    return 1;
}

static int cuda_elastic_batch(
        cuda_elastic_layer &l,
        const ds4_gpu_tensor *selected,
        uint32_t n_tokens,
        uint32_t n_expert,
        const char **gate_w,
        const char **up_w,
        const char **down_w) {
    const uint64_t n_ids = (uint64_t)n_tokens * n_expert;
    std::vector<int32_t> ids((size_t)n_ids);
    if (!g_elastic_scratch ||
        g_elastic_scratch_bytes < cuda_elastic_block_bytes(l, l.t.n_total_expert) ||
        !cuda_elastic_read_selected(selected, ids.data(), n_ids, l.t.n_total_expert)) {
        return 0;
    }
    cuda_elastic_note_batch(l, ids.data(), n_ids);
    /* The scratch block keeps what it gathered for a layer until another
     * layer needs it, which saves the work while only one layer is shrunk. */
    if (g_elastic_scratch_layer != (int64_t)l.t.layer) {
        memset(g_elastic_scratch_valid, 0, sizeof(g_elastic_scratch_valid));
        g_elastic_scratch_layer = (int64_t)l.t.layer;
    }
    const cuda_elastic_view home =
        cuda_elastic_view_of(l, g_elastic_scratch, l.t.n_total_expert);
    const cuda_elastic_view cached = cuda_elastic_view_of(l, l.block, l.slots);
    std::vector<cuda_stream_read_job> jobs;
    int ok = 1;
    for (uint64_t i = 0; ok && i < n_ids; i++) {
        const uint32_t e = (uint32_t)ids[i];
        if (g_elastic_scratch_valid[e]) continue;
        g_elastic_scratch_valid[e] = 1;
        const int16_t slot = l.expert_slot[e];
        if (slot < 0) {
            cuda_elastic_expert_jobs(l, e, home, e, jobs);
            g_elastic_stats.batch_reads++;
        } else {
            ok = cuda_elastic_copy_expert(l, home, e, cached, (uint32_t)slot);
            g_elastic_stats.batch_cached++;
        }
    }
    if (!ok || !cuda_elastic_read_jobs(jobs)) {
        g_elastic_scratch_layer = -1;
        return 0;
    }
    *gate_w = home.gate;
    *up_w = home.up;
    *down_w = home.down;
    return 1;
}

/* Diagnostic: DS4_ROCM_ELASTIC_TRACE=FILE appends every routed selection as
 * int32 {layer, n_tokens, n_expert, ids...}, to measure routing skew and to
 * replay eviction policies offline.  It adds a readback per launch. */
static void cuda_elastic_trace(
        const cuda_elastic_layer &l,
        const ds4_gpu_tensor *selected,
        uint32_t n_tokens,
        uint32_t n_expert) {
    static FILE *fp;
    static int opened;
    if (!opened) {
        opened = 1;
        const char *path = getenv("DS4_ROCM_ELASTIC_TRACE");
        if (path && path[0]) fp = fopen(path, "ab");
    }
    if (!fp) return;
    std::vector<int32_t> rec(3u + (size_t)n_tokens * n_expert);
    rec[0] = (int32_t)l.t.layer;
    rec[1] = (int32_t)n_tokens;
    rec[2] = (int32_t)n_expert;
    if (cuda_elastic_read_selected(selected, rec.data() + 3, rec.size() - 3u,
                                   l.t.n_total_expert)) {
        fwrite(rec.data(), sizeof(rec[0]), rec.size(), fp);
        fflush(fp);
    }
}

/*
 * Resolve the expert table of one routed MoE launch.  Returns 0 when the
 * layer is not elastic (the caller uses the ordinary model range cache), 1
 * with weights the resident kernels can read, -1 on failure.
 */
static int cuda_elastic_apply(
        const void *model_map,
        uint32_t layer,
        uint32_t n_total_expert,
        uint64_t gate_offset,
        uint64_t up_offset,
        uint64_t down_offset,
        uint64_t gate_expert_bytes,
        uint64_t down_expert_bytes,
        const ds4_gpu_tensor *selected,
        uint32_t n_tokens,
        uint32_t n_expert,
        const ds4_gpu_tensor **selected_exec,
        const char **gate_w,
        const char **up_w,
        const char **down_w) {
    if (g_elastic_layers.empty()) return 0;
    cuda_elastic_layer *l = cuda_elastic_find(model_map, layer, n_total_expert,
                                              gate_offset, up_offset, down_offset,
                                              gate_expert_bytes, down_expert_bytes);
    if (!l) return 0;
    cuda_elastic_trace(*l, selected, n_tokens, n_expert);
    g_elastic_clock++;
    if (g_elastic_scratch) cuda_elastic_maybe_grow();
    if (!l->block) return -1;
    if (cuda_elastic_full(*l)) {
        /* Resident layout.  Record which experts are in use so reclaim knows
         * the cold ones: every batch (one readback per layer and chunk), but
         * only a sample of decode launches, where a readback per layer and
         * token would cost a few percent of decode speed. */
        if (n_tokens > 1u) {
            std::vector<int32_t> ids((size_t)n_tokens * n_expert);
            if (cuda_elastic_read_selected(selected, ids.data(), ids.size(), n_total_expert)) {
                cuda_elastic_note_batch(*l, ids.data(), ids.size());
            }
        } else if (l->launches++ % DS4_ROCM_ELASTIC_SAMPLE == 0u) {
            int32_t ids[DS4_ROCM_N_EXPERT_USED];
            if (n_expert <= DS4_ROCM_N_EXPERT_USED &&
                cuda_elastic_read_selected(selected, ids, n_expert, n_total_expert)) {
                for (uint32_t i = 0; i < n_expert; i++) {
                    cuda_elastic_note_hits(*l, (uint32_t)ids[i], DS4_ROCM_ELASTIC_SAMPLE);
                }
            }
        }
        const cuda_elastic_view view = cuda_elastic_view_of(*l, l->block, l->slots);
        *gate_w = view.gate;
        *up_w = view.up;
        *down_w = view.down;
        return 1;
    }
    const int ok = n_tokens == 1u ?
        cuda_elastic_decode(*l, selected, n_expert, selected_exec, gate_w, up_w, down_w) :
        cuda_elastic_batch(*l, selected, n_tokens, n_expert, gate_w, up_w, down_w);
    return ok ? 1 : -1;
}

extern "C" int ds4_gpu_elastic_expert_table_load(const ds4_gpu_stream_expert_table *table) {
    if (!table || !table->model_map || g_ssd_streaming_mode || g_model_fd < 0 ||
        (g_model_fd_host_base != NULL && table->model_map != g_model_fd_host_base) ||
        table->n_total_expert < 2u * DS4_ROCM_ELASTIC_MIN_SLOTS ||
        table->n_total_expert > DS4_ROCM_MAX_N_EXPERT ||
        table->gate_expert_bytes == 0 || table->down_expert_bytes == 0) {
        return 0;
    }
    cuda_elastic_layer l;
    memset(&l, 0, sizeof(l));
    l.t = *table;
    l.expert_bytes = 2u * table->gate_expert_bytes + table->down_expert_bytes;
    const uint32_t n_total = table->n_total_expert;
    const uint64_t gate_bytes = (uint64_t)n_total * table->gate_expert_bytes;
    const uint64_t down_bytes = (uint64_t)n_total * table->down_expert_bytes;
    if (!cuda_model_range_fits(table->model_size, table->gate_offset, gate_bytes) ||
        !cuda_model_range_fits(table->model_size, table->up_offset, gate_bytes) ||
        !cuda_model_range_fits(table->model_size, table->down_offset, down_bytes)) {
        return 0;
    }
    void *block = NULL;
    if (!cuda_ok(cudaMalloc(&block, (size_t)cuda_elastic_block_bytes(l, n_total)),
                 "elastic expert table")) {
        return 0;
    }
    /* Whole tensors, in chunks the reader pool can spread over its workers. */
    const cuda_elastic_view view = cuda_elastic_view_of(l, (char *)block, n_total);
    const struct { uint64_t offset, bytes; char *dst; } part[3] = {
        {table->gate_offset, gate_bytes, view.gate},
        {table->up_offset, gate_bytes, view.up},
        {table->down_offset, down_bytes, view.down},
    };
    std::vector<cuda_stream_read_job> jobs;
    for (const auto &p : part) {
        for (uint64_t off = 0; off < p.bytes; off += DS4_ROCM_ELASTIC_READ_CHUNK) {
            const uint64_t n = std::min<uint64_t>(p.bytes - off, DS4_ROCM_ELASTIC_READ_CHUNK);
            jobs.push_back({p.dst + off, p.offset + off, n, NULL, NULL, 0, 0});
        }
    }
    if (!cuda_elastic_read_jobs(jobs)) {
        (void)cudaFree(block);
        return 0;
    }
    cuda_elastic_clear_slots(l, n_total);
    l.block = (char *)block;
    for (uint32_t e = 0; e < n_total; e++) {
        l.slot_expert[e] = (int32_t)e;
        l.expert_slot[e] = (int16_t)e;
    }
    if (g_elastic_layers.size() <= table->layer) {
        cuda_elastic_layer unused;
        memset(&unused, 0, sizeof(unused));
        g_elastic_layers.resize((size_t)table->layer + 1u, unused);
    }
    g_elastic_layers[table->layer] = l;
    (void)cuda_elastic_stats_on();
    cuda_model_load_progress_note(g_model_range_bytes + cuda_elastic_resident_bytes());
    return 1;
}

extern "C" int ds4_gpu_model_range_is_elastic(const void *model_map, uint64_t offset, uint64_t bytes) {
    return cuda_elastic_owns_range(model_map, offset, bytes);
}

/* The largest full table: the size of the scratch block shrunk layers need. */
static uint64_t cuda_elastic_max_table_bytes(void) {
    uint64_t bytes = 0;
    for (const cuda_elastic_layer &l : g_elastic_layers) {
        if (l.block) bytes = std::max(bytes, cuda_elastic_block_bytes(l, l.t.n_total_expert));
    }
    return bytes;
}

/* Everything above the minimum slot count of every layer, less the one full
 * table that stays behind as the scratch block. */
extern "C" uint64_t ds4_gpu_reclaimable_weight_bytes(void) {
    uint64_t bytes = 0;
    for (const cuda_elastic_layer &l : g_elastic_layers) {
        if (!l.block || l.slots <= DS4_ROCM_ELASTIC_MIN_SLOTS) continue;
        bytes += cuda_elastic_block_bytes(l, l.slots - DS4_ROCM_ELASTIC_MIN_SLOTS);
    }
    const uint64_t scratch_cost = g_elastic_scratch ? 0 : cuda_elastic_max_table_bytes();
    return bytes > scratch_cost ? bytes - scratch_cost : 0;
}

extern "C" uint64_t ds4_gpu_reclaim_weights(uint64_t need_bytes) {
    if (need_bytes == 0 || ds4_gpu_reclaimable_weight_bytes() == 0) return 0;
    /* No kernel may still read a block that is about to be freed. */
    if (!cuda_ok(cudaDeviceSynchronize(), "elastic reclaim sync")) return 0;

    const uint64_t before = cuda_elastic_resident_bytes();
    const uint64_t max_table = cuda_elastic_max_table_bytes();
    uint64_t now = before;
    /* Taking the scratch block costs memory first, hence the signed form. */
    while (now + need_bytes > before) {
        cuda_elastic_layer *victim = NULL;
        double victim_cost = 0.0;
        for (cuda_elastic_layer &l : g_elastic_layers) {
            if (!l.block || l.slots <= DS4_ROCM_ELASTIC_MIN_SLOTS) continue;
            /* The scratch block must be able to hold any layer's table. */
            if (!g_elastic_scratch &&
                cuda_elastic_block_bytes(l, l.slots) != max_table) {
                continue;
            }
            const double cost = cuda_elastic_shrink_cost(l);
            /* Ties (no routing seen yet) go to the larger, then later layer:
             * the early hash-routed layers have no cold experts at all. */
            if (!victim || cost < victim_cost ||
                (cost == victim_cost && l.slots >= victim->slots)) {
                victim = &l;
                victim_cost = cost;
            }
        }
        if (!victim) break;
        const int ok = g_elastic_scratch ?
            cuda_elastic_resize(*victim, cuda_elastic_next_slots(*victim, -1)) :
            cuda_elastic_take_scratch(*victim);
        if (!ok) break;
        g_elastic_stats.shrinks++;
        now = cuda_elastic_resident_bytes();
    }
    g_elastic_grow_checked = cuda_wall_sec();
    const uint64_t freed = before > now ? before - now : 0;
    if (cuda_elastic_stats_on()) {
        uint32_t min_slots = UINT32_MAX;
        uint32_t shrunk = 0;
        for (const cuda_elastic_layer &l : g_elastic_layers) {
            if (!l.block) continue;
            min_slots = std::min(min_slots, l.slots);
            shrunk += !cuda_elastic_full(l);
        }
        fprintf(stderr,
                DS4_GPU_LOG_PREFIX "elastic reclaim: need=%.2f GiB freed=%.2f GiB "
                "shrunk_layers=%u min_slots=%u\n",
                (double)need_bytes / 1073741824.0,
                (double)freed / 1073741824.0,
                shrunk, min_slots);
    }
    return freed;
}
