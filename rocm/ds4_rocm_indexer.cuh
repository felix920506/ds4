__global__ static void indexer_hadamard_fp4_kernel(float *x, uint32_t n_rows, uint32_t head_dim) {
    uint32_t row = blockIdx.x;
    uint32_t tid = threadIdx.x;
    if (row >= n_rows || head_dim != 128u || tid >= 128u) return;

    __shared__ float vals[128];
    __shared__ float absbuf[128];
    float *xr = x + (uint64_t)row * head_dim;
    vals[tid] = xr[tid];
    __syncthreads();

    for (uint32_t stride = 1u; stride < 128u; stride <<= 1u) {
        if ((tid & stride) == 0u) {
            uint32_t base = (tid & ~(2u * stride - 1u)) + (tid & (stride - 1u));
            float a = vals[base];
            float b = vals[base + stride];
            vals[base] = a + b;
            vals[base + stride] = a - b;
        }
        __syncthreads();
    }

    float v = vals[tid] * 0.08838834764831845f;
    uint32_t fp4_block = tid >> 5u;
    uint32_t lane = tid & 31u;
    uint32_t block_base = fp4_block * 32u;
    absbuf[tid] = fabsf(v);
    __syncthreads();

    for (uint32_t stride = 16u; stride > 0u; stride >>= 1u) {
        if (lane < stride) {
            absbuf[block_base + lane] = fmaxf(absbuf[block_base + lane],
                                              absbuf[block_base + lane + stride]);
        }
        __syncthreads();
    }

    float amax = fmaxf(absbuf[block_base], 7.052966104933725e-38f);
    float scale = exp2f(ceilf(log2f(amax / 6.0f)));
    xr[tid] = dsv4_e2m1fn_dequant_dev(fminf(6.0f, fmaxf(-6.0f, v / scale))) * scale;
}

__global__ static void indexer_scores_kernel(
        float *scores,
        const float *q,
        const float *weights,
        const float *index_comp,
        uint32_t n_comp,
        uint32_t n_tokens,
        uint32_t pos0,
        uint32_t n_head,
        uint32_t head_dim,
        uint32_t ratio,
        float scale,
        int causal) {
    uint32_t c = blockIdx.x;
    uint32_t t = blockIdx.y;
    if (c >= n_comp || t >= n_tokens) return;
    if (causal) {
        uint32_t n_visible = (pos0 + t + 1u) / ratio;
        if (c >= n_visible) {
            if (threadIdx.x == 0) scores[(uint64_t)t * n_comp + c] = -INFINITY;
            return;
        }
    }
    float total = 0.0f;
    for (uint32_t h = 0; h < n_head; h++) {
        const float *qh = q + ((uint64_t)t * n_head + h) * head_dim;
        const float *kh = index_comp + (uint64_t)c * head_dim;
        float dot = 0.0f;
        for (uint32_t d = threadIdx.x; d < head_dim; d += blockDim.x) dot += qh[d] * kh[d];
        __shared__ float partial[256];
        partial[threadIdx.x] = dot;
        __syncthreads();
        for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
            if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
            __syncthreads();
        }
        total += fmaxf(partial[0], 0.0f) * weights[(uint64_t)t * n_head + h];
        __syncthreads();
    }
    if (threadIdx.x == 0) scores[(uint64_t)t * n_comp + c] = total * scale;
}

__global__ static void indexer_score_one_direct_kernel(
        float *scores,
        const float *q,
        const float *weights,
        const float *index_comp,
        uint32_t n_comp,
        uint32_t pos0,
        uint32_t ratio,
        float scale,
        int causal) {
    const uint32_t c = blockIdx.x;
    const uint32_t tid = threadIdx.x;
    const uint32_t lane = tid & 31u;
    const uint32_t warp = tid >> 5u;
    if (c >= n_comp || tid >= 128u) return;
    if (causal) {
        const uint32_t visible = ratio ? (pos0 + 1u) / ratio : n_comp;
        if (c >= visible) {
            if (tid == 0) scores[c] = -INFINITY;
            return;
        }
    }

    __shared__ float krow[128];
    __shared__ float partial[4];
    if (tid < 128u) krow[tid] = index_comp[(uint64_t)c * 128u + tid];
    __syncthreads();

    float total = 0.0f;
    for (uint32_t h0 = 0; h0 < 64u; h0 += 4u) {
        const uint32_t h = h0 + warp;
        const float4 qv = ((const float4 *)(q + (uint64_t)h * 128u))[lane];
        const float4 kv = ((const float4 *)krow)[lane];
        float dot = qv.x * kv.x + qv.y * kv.y + qv.z * kv.z + qv.w * kv.w;
        dot = warp_sum_f32(dot);
        if (lane == 0) partial[warp] = fmaxf(dot, 0.0f) * weights[h] * scale;
        __syncthreads();
        if (tid == 0) total += partial[0] + partial[1] + partial[2] + partial[3];
        __syncthreads();
    }
    if (tid == 0) scores[c] = total;
}

/* indexer_score_one_direct_kernel for long contexts, with the same scores bit
 * for bit.  That kernel spends a block of four waves on every compressed row
 * and crosses LDS for each head, which is fine for a few thousand rows and
 * half of a decode step at a million tokens.  Here a lane owns a row: its 128
 * key values stay in registers, the query is the same for every lane, and a
 * wave scores 32 rows with no LDS and no barrier.
 *
 * The rounding is the original's, so the top-k that follows picks the same
 * rows: a four-term fused dot per original lane, the pairwise tree that
 * warp_sum_f32 leaves in lane 0, then the heads added in order.  The build
 * uses -ffast-math, so three things hold that order in place: the fp pragma,
 * the fused multiply-adds spelled out, and the empty asm statements.  The
 * backend fuses a multiply into an add whatever the pragma says, and a value
 * that passes through an asm operand is opaque to it.  The "memory" barrier is
 * for speed: it keeps a batch of query loads ahead of their sums, so a wave
 * waits for memory once per batch instead of once per load.  So is restrict:
 * knowing the scores do not alias the query, the compiler reads the query
 * with scalar loads, 16 values at a time for the whole wave. */
__device__ __forceinline__ static float indexer_score_dot4(float4 a, float4 b) {
#pragma clang fp reassociate(off) contract(off)
    return __builtin_fmaf(a.w, b.w, __builtin_fmaf(a.z, b.z, __builtin_fmaf(a.y, b.y, a.x * b.x)));
}

__global__ static void indexer_score_one_rows_kernel(
        float *__restrict__ scores,
        const float *__restrict__ q,
        const float *__restrict__ weights,
        const float *__restrict__ index_comp,
        uint32_t n_comp,
        float scale) {
#pragma clang fp reassociate(off) contract(off)
    enum { AHEAD = 8 };
    const uint32_t c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= n_comp) return;

    float4 k[32];
    const float4 *krow = (const float4 *)(index_comp + (uint64_t)c * 128u);
#pragma unroll
    for (int i = 0; i < 32; i++) k[i] = krow[i];

    float total = 0.0f;
    for (uint32_t h = 0; h < 64u; h++) {
        const float4 *qh = (const float4 *)(q + (uint64_t)h * 128u);
        float s[16];
#pragma unroll
        for (int b = 0; b < 16; b += AHEAD) {
            float4 qlo[AHEAD], qhi[AHEAD];
#pragma unroll
            for (int j = 0; j < AHEAD; j++) {
                qlo[j] = qh[b + j];
                qhi[j] = qh[b + j + 16];
            }
            __asm__ volatile("" ::: "memory");
#pragma unroll
            for (int j = 0; j < AHEAD; j++) {
                s[b + j] = indexer_score_dot4(qlo[j], k[b + j]) +
                           indexer_score_dot4(qhi[j], k[b + j + 16]);
                __asm__ volatile("" : "+v"(s[b + j]));
            }
        }
#pragma unroll
        for (int i = 0; i < 8; i++) s[i] = s[i] + s[i + 8];
#pragma unroll
        for (int i = 0; i < 4; i++) s[i] = s[i] + s[i + 4];
        s[0] = s[0] + s[2];
        s[1] = s[1] + s[3];
        float p = (fmaxf(s[0] + s[1], 0.0f) * scale) * weights[h];
        __asm__ volatile("" : "+v"(p));
        total = total + p;
    }
    scores[c] = total;
}

/* Indexer scores for a 16-token x 128-key tile: score(t, c) = sum over heads of
 * w[t][h] * relu(q[t][h] . k[c]).  The key tile is staged once; the query rows
 * are staged four heads at a time, and every wave keeps its 16x16 score tile
 * in the WMMA accumulator layout (row = 2i + lane / 16, col = lane % 16) so the
 * ReLU-weighted head sum runs in registers without a trip through LDS. */
__global__ static void __launch_bounds__(256, 1) indexer_scores_wmma128_kernel(
        float *scores,
        const float *q,
        const float *weights,
        const float *index_comp,
        uint32_t n_comp,
        uint32_t n_tokens,
        uint32_t pos0,
        uint32_t n_head,
        uint32_t head_dim,
        uint32_t ratio,
        float scale,
        int causal) {
    constexpr uint32_t HEAD_GROUP = 4u;
    constexpr uint32_t STRIDE = 128u + 8u;
    const uint32_t tile_c = blockIdx.x * 128u;
    const uint32_t tile_t = blockIdx.y * 16u;
    const uint32_t tid = threadIdx.x;
    const uint32_t wave = tid >> 5u;
    const uint32_t lane = tid & 31u;
    const uint32_t lane16 = lane & 15u;
    const uint32_t lane_hi = lane >> 4u;
    if (head_dim != 128u || (n_head % HEAD_GROUP) != 0u) return;

    if (causal) {
        const uint32_t last_token = min(tile_t + 16u, n_tokens);
        const uint32_t max_visible = last_token > tile_t
            ? min((pos0 + last_token) / ratio, n_comp)
            : 0u;
        if (tile_c >= max_visible) {
            for (uint32_t i = tid; i < 16u * 128u; i += 256u) {
                const uint32_t r = i >> 7u;
                const uint32_t c = i & 127u;
                const uint32_t token = tile_t + r;
                const uint32_t comp = tile_c + c;
                if (token < n_tokens && comp < n_comp) {
                    scores[(uint64_t)token * n_comp + comp] = -INFINITY;
                }
            }
            return;
        }
    }

    __shared__ __align__(16) half a_sh[HEAD_GROUP * 16u * STRIDE];
    __shared__ __align__(16) half b_sh[128u * STRIDE];
    __shared__ float w_sh[HEAD_GROUP * 16u];

#pragma unroll
    for (uint32_t step = 0; step < 16u; step++) {
        const uint32_t i = tid + step * 256u;
        const uint32_t c = i >> 5u;
        const uint32_t d = (i & 31u) * 4u;
        const uint32_t comp = tile_c + c;
        float4 f = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        if (comp < n_comp) f = *reinterpret_cast<const float4 *>(index_comp + (uint64_t)comp * 128u + d);
        half2 *dst = reinterpret_cast<half2 *>(b_sh + c * STRIDE + d);
        dst[0] = __floats2half2_rn(f.x, f.y);
        dst[1] = __floats2half2_rn(f.z, f.w);
    }

    float acc[8];
#pragma unroll
    for (uint32_t j = 0; j < 8u; j++) acc[j] = 0.0f;

    /* The next head group's query rows are fetched into registers while the
     * current group runs its MMAs. */
    float4 q_pre[8];
    float w_pre = 0.0f;
    auto load_group = [&](uint32_t h0) {
#pragma unroll
        for (uint32_t step = 0; step < 8u; step++) {
            const uint32_t i = tid + step * 256u;
            const uint32_t hl = i >> 9u;
            const uint32_t r = (i >> 5u) & 15u;
            const uint32_t d = (i & 31u) * 4u;
            const uint32_t token = tile_t + r;
            float4 f = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            if (token < n_tokens) {
                f = *reinterpret_cast<const float4 *>(q + ((uint64_t)token * n_head + h0 + hl) * 128u + d);
            }
            q_pre[step] = f;
        }
        if (tid < HEAD_GROUP * 16u) {
            const uint32_t token = tile_t + (tid & 15u);
            w_pre = token < n_tokens ? weights[(uint64_t)token * n_head + h0 + (tid >> 4u)] : 0.0f;
        }
    };
    load_group(0u);

    for (uint32_t h0 = 0; h0 < n_head; h0 += HEAD_GROUP) {
        __syncthreads();
#pragma unroll
        for (uint32_t step = 0; step < 8u; step++) {
            const uint32_t i = tid + step * 256u;
            const uint32_t hl = i >> 9u;
            const uint32_t r = (i >> 5u) & 15u;
            const uint32_t d = (i & 31u) * 4u;
            half2 *dst = reinterpret_cast<half2 *>(a_sh + (hl * 16u + r) * STRIDE + d);
            dst[0] = __floats2half2_rn(q_pre[step].x, q_pre[step].y);
            dst[1] = __floats2half2_rn(q_pre[step].z, q_pre[step].w);
        }
        if (tid < HEAD_GROUP * 16u) w_sh[tid] = w_pre;
        if (h0 + HEAD_GROUP < n_head) load_group(h0 + HEAD_GROUP);
        __syncthreads();
#pragma unroll
        for (uint32_t hl = 0; hl < HEAD_GROUP; hl++) {
            ds4_q8_float8_t c;
#pragma unroll
            for (uint32_t j = 0; j < 8u; j++) c[j] = 0.0f;
#pragma unroll
            for (uint32_t k0 = 0; k0 < 128u; k0 += 16u) {
                const ds4_q8_half16_t a = lds_load_half16(a_sh + (hl * 16u + lane16) * STRIDE + k0);
                const ds4_q8_half16_t b = lds_load_half16(b_sh + (wave * 16u + lane16) * STRIDE + k0);
                c = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, c);
            }
#pragma unroll
            for (uint32_t j = 0; j < 8u; j++) {
                acc[j] += fmaxf(c[j], 0.0f) * w_sh[hl * 16u + 2u * j + lane_hi];
            }
        }
    }

#pragma unroll
    for (uint32_t j = 0; j < 8u; j++) {
        const uint32_t token = tile_t + 2u * j + lane_hi;
        const uint32_t comp = tile_c + wave * 16u + lane16;
        if (token < n_tokens && comp < n_comp) {
            float out = acc[j] * scale;
            if (causal) {
                const uint32_t visible = (pos0 + token + 1u) / ratio;
                if (comp >= visible) out = -INFINITY;
            }
            scores[(uint64_t)token * n_comp + comp] = out;
        }
    }
}

__global__ static void argmax_kernel(int32_t *out_idx, const float *logits, uint32_t n_vocab) {
    enum { THREADS = 1024 };
    __shared__ float sm_val[THREADS];
    __shared__ int32_t sm_idx[THREADS];

    const uint32_t tid = threadIdx.x;
    float local_v = -INFINITY;
    int32_t local_i = 0;
    for (uint32_t i = tid; i < n_vocab; i += THREADS) {
        const float v = logits[i];
        if (v > local_v) {
            local_v = v;
            local_i = (int32_t)i;
        }
    }
    sm_val[tid] = local_v;
    sm_idx[tid] = local_i;
    __syncthreads();

    for (uint32_t s = THREADS / 2u; s > 0u; s >>= 1u) {
        if (tid < s) {
            const float vr = sm_val[tid + s];
            const int32_t ir = sm_idx[tid + s];
            const float vl = sm_val[tid];
            const int32_t il = sm_idx[tid];
            if ((vr > vl) || (vr == vl && ir < il)) {
                sm_val[tid] = vr;
                sm_idx[tid] = ir;
            }
        }
        __syncthreads();
    }

    if (tid == 0u) *out_idx = sm_idx[0];
}

__global__ static void indexer_topk_kernel(uint32_t *selected, const float *scores, uint32_t n_comp, uint32_t n_tokens, uint32_t top_k) {
    uint32_t t = blockIdx.x;
    if (t >= n_tokens || threadIdx.x != 0) return;
    const float *row = scores + (uint64_t)t * n_comp;
    uint32_t *sel = selected + (uint64_t)t * top_k;
    for (uint32_t k = 0; k < top_k; k++) sel[k] = 0;
    for (uint32_t c = 0; c < n_comp; c++) {
        float v = row[c];
        for (uint32_t k = 0; k < top_k; k++) {
            if ((k >= c) || v > row[sel[k]]) {
                for (uint32_t j = top_k - 1; j > k; j--) sel[j] = sel[j - 1];
                sel[k] = c;
                break;
            }
        }
    }
}

__device__ __forceinline__ static bool topk_score_better(float av, uint32_t ai, float bv, uint32_t bi) {
    return av > bv || (av == bv && ai < bi);
}

template <uint32_t THREADS>
__global__ static void indexer_top1_rows_kernel(
        uint32_t *selected,
        const float *scores,
        uint32_t n_comp,
        uint32_t n_tokens) {
    const uint32_t token = blockIdx.x;
    const uint32_t tid = threadIdx.x;
    if (token >= n_tokens) return;

    const float *row = scores + (uint64_t)token * n_comp;
    float best_value = -INFINITY;
    uint32_t best_index = 0u;
    for (uint32_t i = tid; i < n_comp; i += THREADS) {
        const float value = row[i];
        if (topk_score_better(value, i, best_value, best_index)) {
            best_value = value;
            best_index = i;
        }
    }

    __shared__ float values[THREADS];
    __shared__ uint32_t indices[THREADS];
    values[tid] = best_value;
    indices[tid] = best_index;
    __syncthreads();

    for (uint32_t stride = THREADS / 2u; stride > 0u; stride >>= 1u) {
        if (tid < stride &&
            topk_score_better(values[tid + stride], indices[tid + stride],
                              values[tid], indices[tid])) {
            values[tid] = values[tid + stride];
            indices[tid] = indices[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0u) selected[token] = indices[0];
}

/* DSpark Markov correction: select argmax(logits + W2 * W1[prev]) without
 * moving the vocabulary row back to the host. W1 and W2 are Q8_0. */
__global__ static void dspark_markov_argmax_kernel(
        unsigned long long *out_key,
        const float *logits,
        const unsigned char *w1_row,
        const unsigned char *w2,
        uint32_t vocab,
        uint32_t rank_blocks) {
    __shared__ float state[256];
    const uint32_t tid = threadIdx.x;
    if (tid < rank_blocks * 32u) {
        const uint32_t block = tid >> 5u;
        const uint32_t lane = tid & 31u;
        const unsigned char *qblock = w1_row + (uint64_t)block * 34u;
        const float scale = __half2float(*(const __half *)qblock);
        state[tid] =
            scale * (float)((const int8_t *)(qblock + 2u))[lane];
    }
    __syncthreads();

    float best_value = -INFINITY;
    uint32_t best_index = 0;
    for (uint32_t i = blockIdx.x * blockDim.x + tid; i < vocab;
         i += gridDim.x * blockDim.x) {
        const unsigned char *row =
            w2 + (uint64_t)i * rank_blocks * 34u;
        float acc = 0.0f;
        for (uint32_t block = 0; block < rank_blocks; block++) {
            const unsigned char *qblock = row + (uint64_t)block * 34u;
            const float scale = __half2float(*(const __half *)qblock);
            const int8_t *quants = (const int8_t *)(qblock + 2u);
            float sum = 0.0f;
#pragma unroll
            for (uint32_t lane = 0; lane < 32u; lane++) {
                sum += (float)quants[lane] * state[block * 32u + lane];
            }
            acc += scale * sum;
        }
        const float value = logits[i] + acc;
        if (topk_score_better(value, i, best_value, best_index)) {
            best_value = value;
            best_index = i;
        }
    }

    __shared__ float values[256];
    __shared__ uint32_t indices[256];
    values[tid] = best_value;
    indices[tid] = best_index;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1u; stride > 0u; stride >>= 1u) {
        if (tid < stride &&
            topk_score_better(values[tid + stride], indices[tid + stride],
                              values[tid], indices[tid])) {
            values[tid] = values[tid + stride];
            indices[tid] = indices[tid + stride];
        }
        __syncthreads();
    }
    if (tid == 0u) {
        const unsigned int bits = __float_as_uint(values[0]);
        const unsigned int value_key =
            (bits & 0x80000000u) ? ~bits : (bits | 0x80000000u);
        const unsigned long long key =
            ((unsigned long long)value_key << 32) |
            (unsigned int)(~indices[0]);
        atomicMax(out_key, key);
    }
}

__device__ __forceinline__ static uint32_t topk_float_ordered_key(float v) {
    uint32_t u = __float_as_uint(v);
    /* Match the comparison sort's equal treatment of signed zeros. */
    if ((u & 0x7fffffffu) == 0u) u = 0u;
    return (u & 0x80000000u) ? ~u : (u ^ 0x80000000u);
}

__device__ __forceinline__ static uint64_t topk_pack_key(float v, uint32_t idx) {
    return ((uint64_t)topk_float_ordered_key(v) << 32u) | (uint64_t)(0xffffffffu - idx);
}

__global__ static void indexer_topk_8192_cub_kernel(
        uint32_t *selected,
        const float *scores,
        uint32_t n_comp,
        uint32_t n_tokens,
        uint32_t top_k) {
    constexpr uint32_t BLOCK_THREADS = 512u;
    constexpr uint32_t ITEMS_PER_THREAD = 16u;
    using BlockSort = cub::BlockRadixSort<uint64_t, BLOCK_THREADS, ITEMS_PER_THREAD>;
    extern __shared__ __align__(16) unsigned char sort_smem[];
    typename BlockSort::TempStorage &sort_storage =
        *reinterpret_cast<typename BlockSort::TempStorage *>(sort_smem);

    const uint32_t t = blockIdx.x;
    const uint32_t tid = threadIdx.x;
    if (t >= n_tokens || tid >= BLOCK_THREADS) return;

    const float *row = scores + (uint64_t)t * n_comp;
    uint64_t keys[ITEMS_PER_THREAD];
#pragma unroll
    for (uint32_t item = 0; item < ITEMS_PER_THREAD; item++) {
        const uint32_t i = tid * ITEMS_PER_THREAD + item;
        if (i < n_comp) {
            keys[item] = topk_pack_key(row[i], i);
        } else {
            keys[item] = topk_pack_key(-INFINITY, UINT32_MAX);
        }
    }

    BlockSort(sort_storage).SortDescending(keys);

#pragma unroll
    for (uint32_t item = 0; item < ITEMS_PER_THREAD; item++) {
        const uint32_t i = tid * ITEMS_PER_THREAD + item;
        if (i < top_k) {
            selected[(uint64_t)t * top_k + i] = 0xffffffffu - (uint32_t)keys[item];
        }
    }
}

__global__ static void indexer_topk_1024_kernel(
        uint32_t *selected,
        const float *scores,
        uint32_t n_comp,
        uint32_t n_tokens,
        uint32_t top_k) {
    uint32_t t = blockIdx.x;
    uint32_t tid = threadIdx.x;
    if (t >= n_tokens || tid >= 1024u) return;
    __shared__ float vals[1024];
    __shared__ uint32_t idxs[1024];

    const float *row = scores + (uint64_t)t * n_comp;
    if (tid < n_comp) {
        vals[tid] = row[tid];
        idxs[tid] = tid;
    } else {
        vals[tid] = -INFINITY;
        idxs[tid] = UINT32_MAX;
    }
    __syncthreads();

    for (uint32_t k = 2u; k <= 1024u; k <<= 1u) {
        for (uint32_t j = k >> 1u; j > 0u; j >>= 1u) {
            uint32_t other = tid ^ j;
            if (other > tid && other < 1024u) {
                const float av = vals[tid];
                const float bv = vals[other];
                const uint32_t ai = idxs[tid];
                const uint32_t bi = idxs[other];
                const bool desc_half = (tid & k) == 0u;
                const bool swap = desc_half
                    ? topk_score_better(bv, bi, av, ai)
                    : topk_score_better(av, ai, bv, bi);
                if (swap) {
                    vals[tid] = bv;
                    idxs[tid] = bi;
                    vals[other] = av;
                    idxs[other] = ai;
                }
            }
            __syncthreads();
        }
    }

    if (tid < top_k) selected[(uint64_t)t * top_k + tid] = idxs[tid];
}

template <uint32_t SORT_N>
__global__ static void indexer_topk_pow2_kernel(
        uint32_t *selected,
        const float *scores,
        uint32_t n_comp,
        uint32_t n_tokens,
        uint32_t top_k) {
    uint32_t t = blockIdx.x;
    uint32_t tid = threadIdx.x;
    if (t >= n_tokens) return;
    __shared__ float vals[SORT_N];
    __shared__ uint32_t idxs[SORT_N];

    const float *row = scores + (uint64_t)t * n_comp;
    for (uint32_t i = tid; i < SORT_N; i += blockDim.x) {
        if (i < n_comp) {
            vals[i] = row[i];
            idxs[i] = i;
        } else {
            vals[i] = -INFINITY;
            idxs[i] = UINT32_MAX;
        }
    }
    __syncthreads();

    for (uint32_t k = 2u; k <= SORT_N; k <<= 1u) {
        for (uint32_t j = k >> 1u; j > 0u; j >>= 1u) {
            for (uint32_t i = tid; i < SORT_N; i += blockDim.x) {
                uint32_t other = i ^ j;
                if (other > i && other < SORT_N) {
                    const float av = vals[i];
                    const float bv = vals[other];
                    const uint32_t ai = idxs[i];
                    const uint32_t bi = idxs[other];
                    const bool desc_half = (i & k) == 0u;
                    const bool swap = desc_half
                        ? topk_score_better(bv, bi, av, ai)
                        : topk_score_better(av, ai, bv, bi);
                    if (swap) {
                        vals[i] = bv;
                        idxs[i] = bi;
                        vals[other] = av;
                        idxs[other] = ai;
                    }
                }
            }
            __syncthreads();
        }
    }

    for (uint32_t i = tid; i < top_k; i += blockDim.x) {
        selected[(uint64_t)t * top_k + i] = idxs[i];
    }
}

template <uint32_t SORT_N>
__global__ static void indexer_topk_pow2_u16_kernel(
        uint32_t *selected,
        const float *scores,
        uint32_t n_comp,
        uint32_t n_tokens,
        uint32_t top_k) {
    uint32_t t = blockIdx.x;
    uint32_t tid = threadIdx.x;
    if (t >= n_tokens) return;
    __shared__ float vals[SORT_N];
    __shared__ uint16_t idxs[SORT_N];

    const float *row = scores + (uint64_t)t * n_comp;
    for (uint32_t i = tid; i < SORT_N; i += blockDim.x) {
        if (i < n_comp) {
            vals[i] = row[i];
            idxs[i] = (uint16_t)i;
        } else {
            vals[i] = -INFINITY;
            idxs[i] = UINT16_MAX;
        }
    }
    __syncthreads();

    for (uint32_t k = 2u; k <= SORT_N; k <<= 1u) {
        for (uint32_t j = k >> 1u; j > 0u; j >>= 1u) {
            for (uint32_t i = tid; i < SORT_N; i += blockDim.x) {
                uint32_t other = i ^ j;
                if (other > i && other < SORT_N) {
                    const float av = vals[i];
                    const float bv = vals[other];
                    const uint32_t ai = idxs[i];
                    const uint32_t bi = idxs[other];
                    const bool desc_half = (i & k) == 0u;
                    const bool swap = desc_half
                        ? topk_score_better(bv, bi, av, ai)
                        : topk_score_better(av, ai, bv, bi);
                    if (swap) {
                        vals[i] = bv;
                        idxs[i] = (uint16_t)bi;
                        vals[other] = av;
                        idxs[other] = (uint16_t)ai;
                    }
                }
            }
            __syncthreads();
        }
    }

    for (uint32_t i = tid; i < top_k; i += blockDim.x) {
        selected[(uint64_t)t * top_k + i] = idxs[i];
    }
}

/* Top-k of a long score row by radix select.
 *
 * The result is the TOP_K best rows under the order every other top-k kernel
 * here uses (score descending, then row ascending), best first.  A row's key
 * puts its score above its reversed row number, so keys are distinct and the
 * selection is exact: ten bits a pass, a histogram in LDS finds the next digit
 * of the TOP_K-th largest key, then the keys at or above it are gathered and
 * sorted.  That reads the scores five or six times.  Sorting them in 4096-row
 * chunks and merging the chunks eight at a time took 0.12 ms at 32768 rows
 * and 1.2 ms for 64 prefill tokens over 40000; this takes 0.03 and 0.1 ms. */
__device__ __forceinline__ static uint64_t indexer_topk_key(float v, uint32_t row, uint32_t row_bits) {
    /* +0 and -0 compare equal, so they share a key; then the bits of a float
     * order like the float once the sign is folded in. */
    uint32_t u = v == 0.0f ? 0u : __float_as_uint(v);
    u = (u & 0x80000000u) ? ~u : (u | 0x80000000u);
    return ((uint64_t)u << row_bits) | (uint64_t)((uint32_t)((1ull << row_bits) - 1u) - row);
}

template <uint32_t TOP_K>
__global__ static void indexer_topk_select_kernel(
        uint32_t *selected,
        const float *scores,
        uint32_t n_comp,
        uint32_t n_tokens,
        uint32_t row_bits) {
    enum { AHEAD = 8 };
    const uint32_t t = blockIdx.x;
    const uint32_t tid = threadIdx.x;
    if (t >= n_tokens) return;
    const float *row = scores + (uint64_t)t * n_comp;
    __shared__ uint32_t hist[2][1024];
    __shared__ uint32_t key_hi[TOP_K];
    __shared__ uint32_t key_lo[TOP_K];
    __shared__ uint64_t s_prefix;
    __shared__ uint32_t s_want;
    __shared__ uint32_t s_exact;
    __shared__ uint32_t s_count;

    const uint32_t passes = (32u + row_bits + 9u) / 10u;
    uint64_t prefix = 0;    /* digits of the TOP_K-th key found so far */
    uint32_t want = TOP_K;  /* its rank among the keys that share them */
    uint64_t threshold = 0;
    for (uint32_t p = 0; p < passes; p++) {
        const uint32_t shift = 10u * (passes - 1u - p);
        hist[0][tid] = 0;
        __syncthreads();
        for (uint32_t i = tid; i < n_comp; i += AHEAD * 1024u) {
            float v[AHEAD];
#pragma unroll
            for (uint32_t j = 0; j < AHEAD; j++) {
                v[j] = i + j * 1024u < n_comp ? row[i + j * 1024u] : 0.0f;
            }
#pragma unroll
            for (uint32_t j = 0; j < AHEAD; j++) {
                const uint64_t digits = indexer_topk_key(v[j], i + j * 1024u, row_bits) >> shift;
                if (i + j * 1024u < n_comp && digits >> 10u == prefix) {
                    atomicAdd(&hist[0][(uint32_t)digits & 1023u], 1u);
                }
            }
        }
        __syncthreads();
        /* hist[cur][d] becomes the number of those keys with digit >= d. */
        uint32_t cur = 0;
        for (uint32_t off = 1u; off < 1024u; off <<= 1u) {
            hist[cur ^ 1u][tid] = hist[cur][tid] + (tid + off < 1024u ? hist[cur][tid + off] : 0u);
            cur ^= 1u;
            __syncthreads();
        }
        const uint32_t at = hist[cur][tid];
        const uint32_t above = tid + 1u < 1024u ? hist[cur][tid + 1u] : 0u;
        if (at >= want && above < want) {
            s_prefix = (prefix << 10u) | tid;
            s_want = want - above;
            s_exact = at == want;
        }
        __syncthreads();
        prefix = s_prefix;
        want = s_want;
        threshold = prefix << shift;
        /* Every key with these digits or better is wanted, so the lower
         * digits of the TOP_K-th key no longer matter. */
        if (s_exact) break;
    }

    if (tid == 0) s_count = 0;
    __syncthreads();
    for (uint32_t i = tid; i < n_comp; i += 1024u) {
        const uint64_t key = indexer_topk_key(row[i], i, row_bits);
        if (key >= threshold) {
            const uint32_t slot = atomicAdd(&s_count, 1u);
            key_hi[slot] = (uint32_t)(key >> 32u);
            key_lo[slot] = (uint32_t)key;
        }
    }
    __syncthreads();

    for (uint32_t k = 2u; k <= TOP_K; k <<= 1u) {
        for (uint32_t j = k >> 1u; j > 0u; j >>= 1u) {
            for (uint32_t i = tid; i < TOP_K; i += 1024u) {
                const uint32_t other = i ^ j;
                if (other > i) {
                    const uint64_t a = ((uint64_t)key_hi[i] << 32u) | key_lo[i];
                    const uint64_t b = ((uint64_t)key_hi[other] << 32u) | key_lo[other];
                    if ((i & k) == 0u ? b > a : a > b) {
                        key_hi[i] = (uint32_t)(b >> 32u);
                        key_lo[i] = (uint32_t)b;
                        key_hi[other] = (uint32_t)(a >> 32u);
                        key_lo[other] = (uint32_t)a;
                    }
                }
            }
            __syncthreads();
        }
    }
    const uint32_t row_mask = (uint32_t)((1ull << row_bits) - 1u);
    for (uint32_t i = tid; i < TOP_K; i += 1024u) {
        selected[(uint64_t)t * TOP_K + i] = row_mask - (key_lo[i] & row_mask);
    }
}

__global__ static void indexed_topk_sort_512_asc_kernel(
        int32_t *dst,
        const int32_t *src,
        uint32_t n_tokens) {
    const uint32_t t = blockIdx.x;
    const uint32_t tid = threadIdx.x;
    if (t >= n_tokens || tid >= 512u) return;
    __shared__ int32_t rows[512];

    const int32_t *src_row = src + (uint64_t)t * 512u;
    int32_t *dst_row = dst + (uint64_t)t * 512u;
    rows[tid] = src_row[tid];
    __syncthreads();

    for (uint32_t k = 2u; k <= 512u; k <<= 1u) {
        for (uint32_t j = k >> 1u; j > 0u; j >>= 1u) {
            const uint32_t other = tid ^ j;
            if (other > tid && other < 512u) {
                const int32_t a = rows[tid];
                const int32_t b = rows[other];
                const bool up = (tid & k) == 0u;
                if ((up && a > b) || (!up && a < b)) {
                    rows[tid] = b;
                    rows[other] = a;
                }
            }
            __syncthreads();
        }
    }

    dst_row[tid] = rows[tid];
}

__global__ static void topk_mask_kernel(float *mask, const uint32_t *topk, uint32_t n_comp, uint32_t n_tokens, uint32_t top_k) {
    uint64_t gid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t n = (uint64_t)n_tokens * n_comp;
    if (gid >= n) return;
    uint32_t t = gid / n_comp;
    uint32_t c = gid - (uint64_t)t * n_comp;
    float v = -INFINITY;
    for (uint32_t k = 0; k < top_k; k++) {
        if (topk[(uint64_t)t * top_k + k] == c) {
            v = 0.0f;
            break;
        }
    }
    mask[gid] = v;
}

static int indexer_scores_launch(
        ds4_gpu_tensor       *scores,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *weights,
        const ds4_gpu_tensor *index_comp,
        uint32_t                n_comp,
        uint32_t                n_tokens,
        uint32_t                pos0,
        uint32_t                n_head,
        uint32_t                head_dim,
        uint32_t                ratio,
        float                   scale,
        uint32_t                causal) {
    if (!scores || !q || !weights || !index_comp ||
        n_comp == 0 || n_tokens == 0 || n_head == 0 || head_dim == 0 ||
        q->bytes < (uint64_t)n_tokens * n_head * head_dim * sizeof(float) ||
        weights->bytes < (uint64_t)n_tokens * n_head * sizeof(float) ||
        index_comp->bytes < (uint64_t)n_comp * head_dim * sizeof(float) ||
        scores->bytes < (uint64_t)n_tokens * n_comp * sizeof(float)) {
        return 0;
    }
    if (causal && ratio == 0) return 0;
    /* Below a couple of thousand rows the block-per-row kernel is quicker: a
     * lone wave of the rows kernel takes over 30 us for its 64 heads. */
    if (n_tokens == 1u && head_dim == 128u && n_head == 64u && !causal && n_comp > 2048u) {
        indexer_score_one_rows_kernel<<<(n_comp + 255u) / 256u, 256>>>((float *)scores->ptr,
                                                                      (const float *)q->ptr,
                                                                      (const float *)weights->ptr,
                                                                      (const float *)index_comp->ptr,
                                                                      n_comp, scale);
        return cuda_ok(cudaGetLastError(), "indexer score one rows launch");
    }
    if (n_tokens == 1u && head_dim == 128u && n_head == 64u) {
        indexer_score_one_direct_kernel<<<n_comp, 128>>>((float *)scores->ptr,
                                                         (const float *)q->ptr,
                                                         (const float *)weights->ptr,
                                                         (const float *)index_comp->ptr,
                                                         n_comp, pos0, ratio,
                                                         scale, causal ? 1 : 0);
        return cuda_ok(cudaGetLastError(), "indexer score one direct launch");
    }
    if (!g_quality_mode && head_dim == 128u && n_head == 64u) {
        dim3 grid((n_comp + 127u) / 128u, (n_tokens + 15u) / 16u, 1);
        indexer_scores_wmma128_kernel<<<grid, 256>>>((float *)scores->ptr,
                                                     (const float *)q->ptr,
                                                     (const float *)weights->ptr,
                                                     (const float *)index_comp->ptr,
                                                     n_comp, n_tokens, pos0, n_head,
                                                     head_dim, ratio, scale, causal ? 1 : 0);
        return cuda_ok(cudaGetLastError(), "indexer scores wmma128 launch");
    }
    dim3 grid(n_comp, n_tokens, 1);
    indexer_scores_kernel<<<grid, 256>>>((float *)scores->ptr,
                                         (const float *)q->ptr,
                                         (const float *)weights->ptr,
                                         (const float *)index_comp->ptr,
                                         n_comp, n_tokens, pos0, n_head,
                                         head_dim, ratio, scale, causal ? 1 : 0);
    return cuda_ok(cudaGetLastError(), "indexer scores launch");
}

extern "C" int ds4_gpu_indexer_score_one_tensor(
        ds4_gpu_tensor       *scores,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *weights,
        const ds4_gpu_tensor *index_comp,
        uint32_t                n_comp,
        uint32_t                n_head,
        uint32_t                head_dim,
        float                   scale) {
    return indexer_scores_launch(scores, q, weights, index_comp, n_comp, 1, 0,
                                 n_head, head_dim, 1, scale, 0);
}

extern "C" int ds4_gpu_indexer_scores_prefill_tensor(
        ds4_gpu_tensor       *scores,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *weights,
        const ds4_gpu_tensor *index_comp,
        uint32_t                n_comp,
        uint32_t                n_tokens,
        uint32_t                n_head,
        uint32_t                head_dim,
        uint32_t                ratio,
        float                   scale) {
    return indexer_scores_launch(scores, q, weights, index_comp, n_comp, n_tokens, 0,
                                 n_head, head_dim, ratio, scale, 1);
}

extern "C" int ds4_gpu_indexer_scores_decode_batch_tensor(
        ds4_gpu_tensor       *scores,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *weights,
        const ds4_gpu_tensor *index_comp,
        uint32_t                n_comp,
        uint32_t                n_tokens,
        uint32_t                pos0,
        uint32_t                n_head,
        uint32_t                head_dim,
        uint32_t                ratio,
        float                   scale) {
    return indexer_scores_launch(scores, q, weights, index_comp, n_comp, n_tokens, pos0,
                                 n_head, head_dim, ratio, scale, 1);
}

extern "C" int ds4_gpu_indexer_topk_tensor(
        ds4_gpu_tensor       *selected,
        const ds4_gpu_tensor *scores,
        uint32_t                n_comp,
        uint32_t                n_tokens,
        uint32_t                top_k) {
    if (!selected || !scores || n_comp == 0 || n_tokens == 0 || top_k == 0 ||
        top_k > n_comp ||
        scores->bytes < (uint64_t)n_tokens * n_comp * sizeof(float) ||
        selected->bytes < (uint64_t)n_tokens * top_k * sizeof(uint32_t)) {
        return 0;
    }
    if (top_k == 1u) {
        indexer_top1_rows_kernel<256u><<<n_tokens, 256u>>>(
                (uint32_t *)selected->ptr,
                (const float *)scores->ptr,
                n_comp, n_tokens);
        return cuda_ok(cudaGetLastError(), "indexer top1 rows launch");
    }
    if (top_k == 512u && n_comp <= 1024u) {
        indexer_topk_1024_kernel<<<n_tokens, 1024>>>((uint32_t *)selected->ptr,
                                                     (const float *)scores->ptr,
                                                     n_comp, n_tokens, top_k);
        return cuda_ok(cudaGetLastError(), "indexer topk 1024 launch");
    }
    if (top_k == 512u && n_comp <= 2048u) {
        indexer_topk_pow2_kernel<2048><<<n_tokens, 1024>>>((uint32_t *)selected->ptr,
                                                           (const float *)scores->ptr,
                                                           n_comp, n_tokens, top_k);
        return cuda_ok(cudaGetLastError(), "indexer topk 2048 launch");
    }
    if (top_k == 512u && n_comp <= 4096u) {
        if (n_comp == 4096u) {
            using TopkCubSort = cub::BlockRadixSort<uint64_t, 512, 16>;
            const int smem = (int)sizeof(typename TopkCubSort::TempStorage);
            int dev = 0;
            int max_optin_smem = 0;
            cudaError_t attr_err = cudaGetDevice(&dev);
            if (attr_err == cudaSuccess) {
                attr_err = cudaDeviceGetAttribute(&max_optin_smem,
                                                  cudaDevAttrMaxSharedMemoryPerBlockOptin,
                                                  dev);
            }
            if (attr_err == cudaSuccess && max_optin_smem >= smem) {
                attr_err = cudaFuncSetAttribute(indexer_topk_8192_cub_kernel,
                                                cudaFuncAttributeMaxDynamicSharedMemorySize,
                                                smem);
                if (attr_err == cudaSuccess) {
                    indexer_topk_8192_cub_kernel<<<n_tokens, 512, (size_t)smem>>>((uint32_t *)selected->ptr,
                                                                                 (const float *)scores->ptr,
                                                                                 n_comp, n_tokens, top_k);
                    return cuda_ok(cudaGetLastError(), "indexer topk 4096 cub launch");
                }
            }
        }
        indexer_topk_pow2_kernel<4096><<<n_tokens, 1024>>>((uint32_t *)selected->ptr,
                                                           (const float *)scores->ptr,
                                                           n_comp, n_tokens, top_k);
        return cuda_ok(cudaGetLastError(), "indexer topk 4096 launch");
    }
    if (top_k == 512u && n_comp <= 8192u) {
        if (n_comp > 4096u) {
            using TopkCubSort = cub::BlockRadixSort<uint64_t, 512, 16>;
            const int smem = (int)sizeof(typename TopkCubSort::TempStorage);
            int dev = 0;
            int max_optin_smem = 0;
            cudaError_t attr_err = cudaGetDevice(&dev);
            if (attr_err == cudaSuccess) {
                attr_err = cudaDeviceGetAttribute(&max_optin_smem,
                                                  cudaDevAttrMaxSharedMemoryPerBlockOptin,
                                                  dev);
            }
            if (attr_err == cudaSuccess && max_optin_smem >= smem) {
                attr_err = cudaFuncSetAttribute(indexer_topk_8192_cub_kernel,
                                                cudaFuncAttributeMaxDynamicSharedMemorySize,
                                                smem);
                if (attr_err == cudaSuccess) {
                    indexer_topk_8192_cub_kernel<<<n_tokens, 512, (size_t)smem>>>((uint32_t *)selected->ptr,
                                                                                 (const float *)scores->ptr,
                                                                                 n_comp, n_tokens, top_k);
                    return cuda_ok(cudaGetLastError(), "indexer topk 8192 cub launch");
                }
            }
        }
        indexer_topk_pow2_u16_kernel<8192><<<n_tokens, 1024>>>((uint32_t *)selected->ptr,
                                                               (const float *)scores->ptr,
                                                               n_comp, n_tokens, top_k);
        return cuda_ok(cudaGetLastError(), "indexer topk 8192 launch");
    }
    if (top_k == 1024u && n_comp <= 1024u) {
        indexer_topk_1024_kernel<<<n_tokens, 1024>>>((uint32_t *)selected->ptr,
                                                     (const float *)scores->ptr,
                                                     n_comp, n_tokens, top_k);
        return cuda_ok(cudaGetLastError(), "indexer topk 1024x1024 launch");
    }
    if (top_k == 1024u && n_comp <= 2048u) {
        indexer_topk_pow2_kernel<2048><<<n_tokens, 1024>>>((uint32_t *)selected->ptr,
                                                           (const float *)scores->ptr,
                                                           n_comp, n_tokens, top_k);
        return cuda_ok(cudaGetLastError(), "indexer topk 2048x1024 launch");
    }
    if (top_k == 1024u && n_comp <= 4096u) {
        indexer_topk_pow2_kernel<4096><<<n_tokens, 1024>>>((uint32_t *)selected->ptr,
                                                           (const float *)scores->ptr,
                                                           n_comp, n_tokens, top_k);
        return cuda_ok(cudaGetLastError(), "indexer topk 4096x1024 launch");
    }
    if (top_k == 1024u && n_comp <= 8192u) {
        if (n_comp > 4096u) {
            using TopkCubSort = cub::BlockRadixSort<uint64_t, 512, 16>;
            const int smem = (int)sizeof(typename TopkCubSort::TempStorage);
            int dev = 0;
            int max_optin_smem = 0;
            cudaError_t attr_err = cudaGetDevice(&dev);
            if (attr_err == cudaSuccess) {
                attr_err = cudaDeviceGetAttribute(&max_optin_smem,
                                                  cudaDevAttrMaxSharedMemoryPerBlockOptin,
                                                  dev);
            }
            if (attr_err == cudaSuccess && max_optin_smem >= smem) {
                attr_err = cudaFuncSetAttribute(indexer_topk_8192_cub_kernel,
                                                cudaFuncAttributeMaxDynamicSharedMemorySize,
                                                smem);
                if (attr_err == cudaSuccess) {
                    indexer_topk_8192_cub_kernel<<<n_tokens, 512, (size_t)smem>>>((uint32_t *)selected->ptr,
                                                                                 (const float *)scores->ptr,
                                                                                 n_comp, n_tokens, top_k);
                    return cuda_ok(cudaGetLastError(), "indexer topk 8192x1024 cub launch");
                }
            }
        }
        indexer_topk_pow2_u16_kernel<8192><<<n_tokens, 1024>>>((uint32_t *)selected->ptr,
                                                               (const float *)scores->ptr,
                                                               n_comp, n_tokens, top_k);
        return cuda_ok(cudaGetLastError(), "indexer topk 8192x1024 launch");
    }
    if (top_k == 2048u && n_comp <= 4096u) {
        indexer_topk_pow2_kernel<4096><<<n_tokens, 1024>>>((uint32_t *)selected->ptr,
                                                           (const float *)scores->ptr,
                                                           n_comp, n_tokens, top_k);
        return cuda_ok(cudaGetLastError(), "indexer topk 4096x2048 launch");
    }
    if (top_k == 2048u && n_comp <= 8192u) {
        if (n_comp > 4096u) {
            using TopkCubSort = cub::BlockRadixSort<uint64_t, 512, 16>;
            const int smem = (int)sizeof(typename TopkCubSort::TempStorage);
            int dev = 0;
            int max_optin_smem = 0;
            cudaError_t attr_err = cudaGetDevice(&dev);
            if (attr_err == cudaSuccess) {
                attr_err = cudaDeviceGetAttribute(&max_optin_smem,
                                                  cudaDevAttrMaxSharedMemoryPerBlockOptin,
                                                  dev);
            }
            if (attr_err == cudaSuccess && max_optin_smem >= smem) {
                attr_err = cudaFuncSetAttribute(indexer_topk_8192_cub_kernel,
                                                cudaFuncAttributeMaxDynamicSharedMemorySize,
                                                smem);
                if (attr_err == cudaSuccess) {
                    indexer_topk_8192_cub_kernel<<<n_tokens, 512, (size_t)smem>>>((uint32_t *)selected->ptr,
                                                                                 (const float *)scores->ptr,
                                                                                 n_comp, n_tokens, top_k);
                    return cuda_ok(cudaGetLastError(), "indexer topk 8192x2048 cub launch");
                }
            }
        }
        indexer_topk_pow2_u16_kernel<8192><<<n_tokens, 1024>>>((uint32_t *)selected->ptr,
                                                               (const float *)scores->ptr,
                                                               n_comp, n_tokens, top_k);
        return cuda_ok(cudaGetLastError(), "indexer topk 8192x2048 launch");
    }
    if (top_k == 512u || top_k == 1024u || top_k == 2048u) {
        uint32_t row_bits = 1;
        while (row_bits < 32u && (1ull << row_bits) < n_comp) row_bits++;
        uint32_t *sel = (uint32_t *)selected->ptr;
        const float *sc = (const float *)scores->ptr;
        if (top_k == 512u) {
            indexer_topk_select_kernel<512><<<n_tokens, 1024>>>(sel, sc, n_comp, n_tokens, row_bits);
        } else if (top_k == 1024u) {
            indexer_topk_select_kernel<1024><<<n_tokens, 1024>>>(sel, sc, n_comp, n_tokens, row_bits);
        } else {
            indexer_topk_select_kernel<2048><<<n_tokens, 1024>>>(sel, sc, n_comp, n_tokens, row_bits);
        }
        return cuda_ok(cudaGetLastError(), "indexer topk select launch");
    }
    indexer_topk_kernel<<<n_tokens, 1>>>((uint32_t *)selected->ptr,
                                         (const float *)scores->ptr,
                                         n_comp, n_tokens, top_k);
    return cuda_ok(cudaGetLastError(), "indexer topk launch");
}

extern "C" int ds4_gpu_argmax_tensor(
        ds4_gpu_tensor       *out_idx,
        const ds4_gpu_tensor *logits,
        uint32_t                n_vocab) {
    uint64_t logits_bytes = 0;
    if (!out_idx || !logits || n_vocab == 0u ||
        out_idx->bytes < sizeof(int32_t) ||
        !cuda_u64_mul3_checked(n_vocab, 1u, sizeof(float), &logits_bytes) ||
        logits->bytes < logits_bytes) {
        return 0;
    }
    argmax_kernel<<<1, 1024>>>((int32_t *)out_idx->ptr,
                               (const float *)logits->ptr,
                               n_vocab);
    return cuda_ok(cudaGetLastError(), "argmax launch");
}

extern "C" int ds4_gpu_dspark_markov_argmax_tensor(
        ds4_gpu_tensor       *out_idx,
        const ds4_gpu_tensor *logits_row,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              w1_offset,
        uint64_t              w2_offset,
        uint32_t              prev_token,
        uint32_t              vocab,
        uint32_t              rank) {
    if (!out_idx || !logits_row || !model_map || vocab == 0 ||
        rank == 0 || (rank & 31u) != 0u || rank > 256u ||
        out_idx->bytes < sizeof(unsigned long long) ||
        logits_row->bytes < (uint64_t)vocab * sizeof(float)) {
        return 0;
    }
    const uint32_t rank_blocks = rank / 32u;
    const uint64_t row_bytes = (uint64_t)rank_blocks * 34u;
    if (prev_token > UINT64_MAX / row_bytes ||
        vocab > UINT64_MAX / row_bytes) {
        return 0;
    }
    const uint64_t w1_row_offset = (uint64_t)prev_token * row_bytes;
    const uint64_t w2_bytes = (uint64_t)vocab * row_bytes;
    if (w1_offset > model_size ||
        w1_row_offset > model_size - w1_offset ||
        row_bytes > model_size - w1_offset - w1_row_offset ||
        w2_offset > model_size || w2_bytes > model_size - w2_offset) {
        return 0;
    }
    const unsigned char *w1_row =
        (const unsigned char *)cuda_model_range_ptr(
            model_map,
            w1_offset + w1_row_offset,
            row_bytes,
            "markov_w1_row");
    const unsigned char *w2 =
        (const unsigned char *)cuda_model_range_ptr(
            model_map, w2_offset, w2_bytes, "markov_w2");
    if (!w1_row || !w2) return 0;

    if (!cuda_ok(cudaMemsetAsync(out_idx->ptr, 0,
                                 sizeof(unsigned long long)),
                 "DSpark markov argmax clear")) {
        return 0;
    }
    dspark_markov_argmax_kernel<<<128, 256>>>(
            (unsigned long long *)out_idx->ptr,
            (const float *)logits_row->ptr,
            w1_row,
            w2,
            vocab,
            rank_blocks);
    return cuda_ok(cudaGetLastError(), "DSpark markov argmax launch");
}

extern "C" int ds4_gpu_dsv4_topk_mask_tensor(
        ds4_gpu_tensor       *mask,
        const ds4_gpu_tensor *topk,
        uint32_t                n_comp,
        uint32_t                n_tokens,
        uint32_t                top_k) {
    if (!mask || !topk || n_comp == 0 || n_tokens == 0 || top_k == 0 ||
        mask->bytes < (uint64_t)n_tokens * n_comp * sizeof(float) ||
        topk->bytes < (uint64_t)n_tokens * top_k * sizeof(uint32_t)) {
        return 0;
    }
    uint64_t n = (uint64_t)n_tokens * n_comp;
    uint64_t nk = (uint64_t)n_tokens * top_k;
    uint64_t blocks = ((n > nk ? n : nk) + 255) / 256;
    topk_mask_kernel<<<blocks, 256>>>((float *)mask->ptr,
                                      (const uint32_t *)topk->ptr,
                                      n_comp, n_tokens, top_k);
    return cuda_ok(cudaGetLastError(), "topk mask launch");
}

extern "C" int ds4_gpu_dsv4_indexer_qat_tensor(ds4_gpu_tensor *x, uint32_t n_rows, uint32_t head_dim) {
    if (!x || n_rows == 0 || head_dim != 128u ||
        x->bytes < (uint64_t)n_rows * head_dim * sizeof(float)) {
        return 0;
    }
    indexer_hadamard_fp4_kernel<<<n_rows, 128>>>((float *)x->ptr, n_rows, head_dim);
    return cuda_ok(cudaGetLastError(), "indexer_hadamard_fp4 launch");
}
