/* DeepSeek V4 Flash Vision-Exp image encoder for gfx1151.
 *
 * The checkpoint is BF16 and the encoder is defined with a BF16 rounding
 * after every operation: each tensor below is the round-to-nearest-even BF16
 * value of an F32 computation on BF16 inputs.  Activations are therefore kept
 * as BF16 words, which is what the WMMA GEMM and attention kernels read, and
 * every rounding point of the reference definition is kept.  Only the order
 * of F32 accumulation inside a GEMM or an attention row differs from a scalar
 * evaluation.
 *
 * That rounding makes the encoder chaotic: a last-bit difference in one F32
 * sum flips a BF16 rounding somewhere, and within a few operations the two
 * evaluations differ by the BF16 quantization noise itself.  Any two correct
 * builds, and a float64 evaluation, end a few percent (relative RMS) apart,
 * so embeddings of two builds cannot be compared bit for bit.  What a kernel
 * controls is its own arithmetic error, which is why the WMMA kernels below
 * never chain a long sum through the instruction's truncating accumulator
 * (see the GEMM kernel); tests/deepseek4_vision_ref.py checks every
 * operation against float64.
 *
 * The image is padded to a multiple of 32 rows for the GEMM and attention
 * tiles.  A pad row is an ordinary blank token that attends to the image, but
 * no real row attends to it: attention masks keys by index and V is zero
 * there.
 *
 * One image already saturates the GPU: stacking the images of a request into
 * one pass measured no faster than encoding them one after the other. */

#define DS4_DSV_EMBD 1024u
#define DS4_DSV_HEADS 16u
#define DS4_DSV_MLP 2816u
#define DS4_DSV_PATCH 588u
#define DS4_DSV_ALIGN_IN 9216u
#define DS4_DSV_OUT 4096u

typedef short __attribute__((ext_vector_type(16))) ds4_dsv_bf16x16_t;
typedef float __attribute__((ext_vector_type(8))) ds4_dsv_f32x8_t;

__device__ __forceinline__ static float deepseek4_vision_round_bf16_dev(
        float value) {
    uint32_t bits = __float_as_uint(value);
    if ((bits & 0x7f800000u) == 0x7f800000u) return value;
    bits += 0x00007fffu + ((bits >> 16u) & 1u);
    return __uint_as_float(bits & 0xffff0000u);
}

/* The BF16 word of an already rounded value; a NaN stays a NaN. */
__device__ __forceinline__ static uint16_t deepseek4_vision_bf16_word(
        float rounded) {
    const uint32_t bits = __float_as_uint(rounded);
    return (uint16_t)((bits >> 16u) |
                      ((bits & 0x7fffffffu) > 0x7f800000u ? 0x0040u : 0u));
}

__device__ __forceinline__ static uint16_t deepseek4_vision_to_bf16(
        float value) {
    return deepseek4_vision_bf16_word(deepseek4_vision_round_bf16_dev(value));
}

/* out = bf16(x + bias [+ residual]); out may alias residual. */
__global__ static void deepseek4_vision_bias_kernel(
        uint16_t       *out,
        const float    *x,
        const uint16_t *bias,
        const uint16_t *residual,
        uint64_t        count,
        uint32_t        width) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    float v = x[i] + glm53_vision_bf16(bias + i % width);
    if (residual) v += glm53_vision_bf16(residual + i);
    out[i] = deepseek4_vision_to_bf16(v);
}

/* x = bf16(bf16(update) + x): the MLP output is rounded before the add. */
__global__ static void deepseek4_vision_residual_kernel(
        uint16_t    *x,
        const float *update,
        uint64_t     count) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    x[i] = deepseek4_vision_to_bf16(
            deepseek4_vision_round_bf16_dev(update[i]) +
            glm53_vision_bf16(x + i));
}

__global__ static void deepseek4_vision_rms_kernel(
        uint16_t       *out,
        const uint16_t *x,
        const uint16_t *weight) {
    __shared__ float partial[256];
    const uint32_t tid = threadIdx.x;
    const uint16_t *xr = x + (uint64_t)blockIdx.x * DS4_DSV_EMBD;
    uint16_t *yr = out + (uint64_t)blockIdx.x * DS4_DSV_EMBD;
    float v[DS4_DSV_EMBD / 256u];
    float sum = 0.0f;
#pragma unroll
    for (uint32_t j = 0; j < DS4_DSV_EMBD / 256u; j++) {
        v[j] = glm53_vision_bf16(xr + tid + j * 256u);
        sum = fmaf(v[j], v[j], sum);
    }
    partial[tid] = sum;
    __syncthreads();
    for (uint32_t stride = 128u; stride != 0u; stride >>= 1u) {
        if (tid < stride) partial[tid] += partial[tid + stride];
        __syncthreads();
    }
    const float inv = rsqrtf(partial[0] / (float)DS4_DSV_EMBD + 1.0e-6f);
#pragma unroll
    for (uint32_t j = 0; j < DS4_DSV_EMBD / 256u; j++) {
        const uint32_t d = tid + j * 256u;
        yr[d] = deepseek4_vision_to_bf16(
                v[j] * inv * glm53_vision_bf16(weight + d));
    }
}

/* rope[pos][freq] = (cos, sin) of pos * 10000^(-freq / 16). */
__global__ static void deepseek4_vision_rope_table_kernel(
        float2  *rope,
        uint32_t count) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    const float inv_freq = powf(10000.0f, -(float)(i & 15u) / 16.0f);
    const float angle = (float)(i >> 4u) * inv_freq;
    rope[i] = make_float2(cosf(angle), sinf(angle));
}

/* Bias and 2D RoPE of q and k into the attention layout [head][row][64].
 * The first 16 pairs of a head rotate by the patch row, the last 16 by its
 * column.  The head is the fast grid axis here and in the V kernel, so
 * consecutive blocks walk one row of the GEMM output. */
__global__ static void deepseek4_vision_qk_rope_kernel(
        uint16_t       *q,
        uint16_t       *k,
        const float    *qkv,
        const uint16_t *bias,
        const float2   *rope,
        uint32_t        grid_w,
        uint32_t        n_rows,
        uint32_t        rows) {
    const uint32_t row = blockIdx.y;
    const uint32_t head = blockIdx.x;
    const uint32_t lane = threadIdx.x;
    const uint64_t qkv_base = (uint64_t)row * 3072u + head * 64u;
    const uint16_t *b = bias + head * 64u;
    const float q0 = deepseek4_vision_round_bf16_dev(
            qkv[qkv_base + lane] + glm53_vision_bf16(b + lane));
    const float q1 = deepseek4_vision_round_bf16_dev(
            qkv[qkv_base + lane + 32u] + glm53_vision_bf16(b + lane + 32u));
    const float k0 = deepseek4_vision_round_bf16_dev(
            qkv[qkv_base + 1024u + lane] +
            glm53_vision_bf16(b + 1024u + lane));
    const float k1 = deepseek4_vision_round_bf16_dev(
            qkv[qkv_base + 1024u + lane + 32u] +
            glm53_vision_bf16(b + 1024u + lane + 32u));
    const uint32_t y = row < n_rows ? row / grid_w : 0u;
    const uint32_t x = row < n_rows ? row - y * grid_w : 0u;
    const float2 rotation = rope[(lane < 16u ? y : x) * 16u + (lane & 15u)];
    const float cs = rotation.x;
    const float sn = rotation.y;
    const uint64_t out_base = ((uint64_t)head * rows + row) * 64u;
    q[out_base + lane] = deepseek4_vision_to_bf16(q0 * cs - q1 * sn);
    q[out_base + lane + 32u] = deepseek4_vision_to_bf16(q1 * cs + q0 * sn);
    k[out_base + lane] = deepseek4_vision_to_bf16(k0 * cs - k1 * sn);
    k[out_base + lane + 32u] = deepseek4_vision_to_bf16(k1 * cs + k0 * sn);
}

/* Bias of v, stored transposed as [head][dim][row] so the attention kernel
 * reads sixteen keys of one dimension with a single load.  A block owns 64
 * rows of one head and each thread 16 rows of one dimension: one contiguous
 * 32-byte store, where a row-major walk would scatter single words over 64
 * streams.  V is zero on pad rows. */
__global__ static void deepseek4_vision_v_kernel(
        uint16_t       *vt,
        const float    *qkv,
        const uint16_t *bias,
        uint32_t        n_rows,
        uint32_t        rows) {
    const uint32_t head = blockIdx.x;
    const uint32_t d = threadIdx.x & 63u;
    const uint32_t row = blockIdx.y * 64u + (threadIdx.x >> 6u) * 16u;
    if (row >= rows) return;
    const float b = glm53_vision_bf16(bias + 2048u + head * 64u + d);
    const float *src = qkv + (uint64_t)row * 3072u + 2048u + head * 64u + d;
    union {
        uint16_t w[16];
        uint4    u[2];
    } v;
#pragma unroll
    for (uint32_t i = 0; i < 16u; i++) {
        v.w[i] = row + i < n_rows
            ? deepseek4_vision_to_bf16(src[(uint64_t)i * 3072u] + b)
            : (uint16_t)0u;
    }
    uint4 *dst = reinterpret_cast<uint4 *>(
            vt + ((uint64_t)head * 64u + d) * rows + row);
    dst[0] = v.u[0];
    dst[1] = v.u[1];
}

__device__ __forceinline__ static ds4_dsv_bf16x16_t deepseek4_vision_load16(
        const uint16_t *p) {
    union {
        uint4 u[2];
        ds4_dsv_bf16x16_t v;
    } r;
    r.u[0] = *reinterpret_cast<const uint4 *>(p);
    r.u[1] = *reinterpret_cast<const uint4 *>(p + 8);
    return r.v;
}

/* Full self-attention over the image: one wave owns 16 query rows of one head
 * and walks the keys 16 at a time with the wave32 BF16 WMMA instruction,
 *
 *   C[row = 2i + lane / 16][col = lane % 16] += A[row = lane % 16][k = i] .
 *                                               B[k = i][col = lane % 16]
 *
 * where i indexes the elements a lane holds and both halves of the wave carry
 * the same A and B.  Scores are computed transposed, S^T = K . Q^T, so a lane
 * owns one query and eight keys: the per-query softmax state is then uniform
 * across a lane and needs one exchange with the other half of the wave.
 *
 * P stays F32 as in the reference definition.  It is split exactly into three
 * BF16 terms (8 + 8 + 8 mantissa bits) for P . V.  The instruction truncates
 * its sum at the precision of the largest operand, so every key tile is
 * accumulated from zero, smallest term first, and tiles are added with IEEE
 * F32 adds.  Chaining all tiles through the WMMA accumulator instead measured
 * ten times less accurate than a scalar F32 loop; this is as accurate as one. */
__global__ static void __launch_bounds__(32) deepseek4_vision_attention_kernel(
        uint16_t       *out,
        const uint16_t *q,
        const uint16_t *k,
        const uint16_t *vt,
        uint32_t        n_rows,
        uint32_t        rows) {
    const uint32_t lane16 = threadIdx.x & 15u;
    const uint32_t half = threadIdx.x >> 4u;
    const uint32_t head = blockIdx.y;
    const uint32_t q0 = blockIdx.x * 16u;
    const uint16_t *kh = k + (uint64_t)head * rows * 64u;
    const uint16_t *vh = vt + (uint64_t)head * 64u * rows;
    const uint16_t *qrow = q + ((uint64_t)head * rows + q0 + lane16) * 64u;

    ds4_dsv_f32x8_t zero;
#pragma unroll
    for (uint32_t i = 0; i < 8u; i++) zero[i] = 0.0f;
    ds4_dsv_bf16x16_t qf[4];
    ds4_dsv_f32x8_t acc[4];
#pragma unroll
    for (uint32_t c = 0; c < 4u; c++) {
        qf[c] = deepseek4_vision_load16(qrow + c * 16u);
        acc[c] = zero;
    }
    float row_max = -INFINITY;
    float row_sum = 0.0f;   /* this lane's eight keys per tile */

    for (uint32_t kb = 0; kb < n_rows; kb += 16u) {
        const uint16_t *kr = kh + (uint64_t)(kb + lane16) * 64u;
        ds4_dsv_f32x8_t s[4];
#pragma unroll
        for (uint32_t c = 0; c < 4u; c++) {
            s[c] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(
                    deepseek4_vision_load16(kr + c * 16u), qf[c], zero);
        }
        s[0] = (s[0] + s[1]) + (s[2] + s[3]);

        float tile_max = -INFINITY;
#pragma unroll
        for (uint32_t i = 0; i < 8u; i++) {
            const float score = kb + 2u * i + half < n_rows
                ? s[0][i] * 0.125f : -INFINITY;
            s[0][i] = score;
            tile_max = fmaxf(tile_max, score);
        }
        tile_max = fmaxf(tile_max, __shfl_xor(tile_max, 16));
        const bool grow = tile_max > row_max;
        if (__builtin_amdgcn_ballot_w32(grow) != 0u) {
            /* Rare after the first tiles.  An output lane holds eight
             * queries, so it gathers their scales from the query lanes. */
            const float scale = grow ? expf(row_max - tile_max) : 1.0f;
            row_sum *= scale;
            row_max = fmaxf(row_max, tile_max);
#pragma unroll
            for (uint32_t i = 0; i < 8u; i++) {
                const float query_scale = __shfl(scale, (int)(2u * i + half));
#pragma unroll
                for (uint32_t d = 0; d < 4u; d++) acc[d][i] *= query_scale;
            }
        }

        /* p = hi + mid + lo exactly; each term packs two keys per word. */
        uint32_t term[3][4];
#pragma unroll
        for (uint32_t i = 0; i < 8u; i++) {
            const float p = expf(s[0][i] - row_max);
            row_sum += p;
            const float hi = deepseek4_vision_round_bf16_dev(p);
            const float mid = deepseek4_vision_round_bf16_dev(p - hi);
            const float lo = deepseek4_vision_round_bf16_dev((p - hi) - mid);
            const uint32_t shift = (i & 1u) * 16u;
            const uint32_t w[3] = {
                __float_as_uint(hi) >> 16u,
                __float_as_uint(mid) >> 16u,
                __float_as_uint(lo) >> 16u,
            };
#pragma unroll
            for (uint32_t t = 0; t < 3u; t++) {
                term[t][i >> 1u] = shift ? term[t][i >> 1u] | (w[t] << 16u)
                                         : w[t];
            }
        }
        /* A operand: this lane's query against all 16 keys of the tile.  The
         * lane has the even or the odd keys, the other half has the rest. */
        ds4_dsv_bf16x16_t pf[3];
#pragma unroll
        for (uint32_t t = 0; t < 3u; t++) {
#pragma unroll
            for (uint32_t j = 0; j < 4u; j++) {
                const uint32_t mine = term[t][j];
                const uint32_t other = __shfl_xor(mine, 16);
                const uint32_t even = half ? other : mine;
                const uint32_t odd = half ? mine : other;
                pf[t][4u * j] = (short)(even & 0xffffu);
                pf[t][4u * j + 1u] = (short)(odd & 0xffffu);
                pf[t][4u * j + 2u] = (short)(even >> 16u);
                pf[t][4u * j + 3u] = (short)(odd >> 16u);
            }
        }
#pragma unroll
        for (uint32_t d = 0; d < 4u; d++) {
            const ds4_dsv_bf16x16_t vf = deepseek4_vision_load16(
                    vh + (uint64_t)(d * 16u + lane16) * rows + kb);
            ds4_dsv_f32x8_t tile = zero;
#pragma unroll
            for (int t = 2; t >= 0; t--) {
                tile = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(
                        pf[t], vf, tile);
            }
            acc[d] += tile;
        }
    }

    row_sum += __shfl_xor(row_sum, 16);
#pragma unroll
    for (uint32_t i = 0; i < 8u; i++) {
        const float denom = __shfl(row_sum, (int)(2u * i + half));
        uint16_t *o = out + (uint64_t)(q0 + 2u * i + half) * DS4_DSV_EMBD +
                      head * 64u + lane16;
#pragma unroll
        for (uint32_t d = 0; d < 4u; d++) {
            o[d * 16u] = deepseek4_vision_to_bf16(acc[d][i] / denom);
        }
    }
}

/* out = bf16(silu(bf16(gate)) * bf16(up)) for gate_up = [gate | up]. */
__global__ static void deepseek4_vision_swiglu_kernel(
        uint16_t    *out,
        const float *gate_up,
        uint64_t     count) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    const uint64_t source = i / DS4_DSV_MLP * (2u * DS4_DSV_MLP) +
                            i % DS4_DSV_MLP;
    const float gate = deepseek4_vision_round_bf16_dev(gate_up[source]);
    const float up = deepseek4_vision_round_bf16_dev(
            gate_up[source + DS4_DSV_MLP]);
    out[i] = deepseek4_vision_to_bf16((gate / (1.0f + expf(-gate))) * up);
}

/* Channel-major 3x3 blocks, zero beyond the grid. */
__global__ static void deepseek4_vision_aligner_reorder_kernel(
        uint16_t       *out,
        const uint16_t *x,
        uint32_t        grid_h,
        uint32_t        grid_w) {
    const uint32_t d = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t row = blockIdx.y;
    if (d >= DS4_DSV_ALIGN_IN) return;
    const uint32_t output_w = (grid_w + 2u) / 3u;
    const uint32_t block_y = row / output_w;
    const uint32_t block_x = row - block_y * output_w;
    const uint32_t channel = d / 9u;
    const uint32_t within = d - channel * 9u;
    const uint32_t source_y = block_y * 3u + within / 3u;
    const uint32_t source_x = block_x * 3u + within % 3u;
    uint16_t value = 0u;
    if (source_y < grid_h && source_x < grid_w) {
        value = x[((uint64_t)source_y * grid_w + source_x) * DS4_DSV_EMBD +
                  channel];
    }
    out[(uint64_t)row * DS4_DSV_ALIGN_IN + d] = value;
}

__global__ static void deepseek4_vision_gelu_bias_kernel(
        uint16_t       *out,
        const float    *x,
        const uint16_t *bias,
        uint64_t        count) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    const float value = deepseek4_vision_round_bf16_dev(
            x[i] + glm53_vision_bf16(bias + i % DS4_DSV_OUT));
    out[i] = deepseek4_vision_to_bf16(
            0.5f * value *
            (1.0f + glm53_vision_erf_approx(value * 0.7071067811865475f)));
}

__global__ static void deepseek4_vision_output_kernel(
        float          *x,
        const uint16_t *bias,
        uint64_t        count) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    x[i] = deepseek4_vision_round_bf16_dev(
            x[i] + glm53_vision_bf16(bias + i % DS4_DSV_OUT));
}

/* out[rows][out_dim] = x[rows][in_dim] . w[out_dim][in_dim]^T for the four
 * projections of a block.  One wave owns 32 rows and 128 columns: A holds 16
 * rows, B sixteen weight rows (the operand layout is in the attention
 * kernel), and both are read straight from memory, 16 values of K per load,
 * so a load of either operand feeds several products.  Rows are the fast
 * grid axis: consecutive waves reuse one 128-row block of weights.
 *
 * The WMMA instruction truncates its sum at the precision of its largest
 * operand, accumulator included.  A GEMM that keeps the running sum in that
 * accumulator, as the library kernels do, loses a truncation per 16 values of
 * K: an order of magnitude more than scalar F32 on rows with a few large
 * activations, which this encoder has, and enough to push more of the output
 * across BF16 rounding boundaries.  Here four instructions (64 values of K)
 * are chained from zero and the partial sums are added with IEEE F32 adds,
 * which measures more accurate than the rocBLAS F32 GEMM. */
__global__ static void __launch_bounds__(32) deepseek4_vision_gemm_kernel(
        float          *out,
        const uint16_t *w,
        const uint16_t *x,
        uint32_t        in_dim,
        uint32_t        out_dim) {
    const uint32_t lane16 = threadIdx.x & 15u;
    const uint32_t half = threadIdx.x >> 4u;
    const uint32_t row0 = blockIdx.x * 32u;
    const uint32_t col0 = blockIdx.y * 128u;
    const uint16_t *xr = x + (uint64_t)(row0 + lane16) * in_dim;
    const uint16_t *wr = w + (uint64_t)(col0 + lane16) * in_dim;
    ds4_dsv_f32x8_t zero;
#pragma unroll
    for (uint32_t i = 0; i < 8u; i++) zero[i] = 0.0f;
    ds4_dsv_f32x8_t acc[2][8];
#pragma unroll
    for (uint32_t t = 0; t < 2u; t++) {
#pragma unroll
        for (uint32_t m = 0; m < 8u; m++) acc[t][m] = zero;
    }
    for (uint32_t k0 = 0; k0 < in_dim; k0 += 64u) {
        ds4_dsv_bf16x16_t a[2][4];
#pragma unroll
        for (uint32_t t = 0; t < 2u; t++) {
#pragma unroll
            for (uint32_t c = 0; c < 4u; c++) {
                a[t][c] = deepseek4_vision_load16(
                        xr + (uint64_t)t * 16u * in_dim + k0 + c * 16u);
            }
        }
#pragma unroll
        for (uint32_t m = 0; m < 8u; m++) {
            ds4_dsv_bf16x16_t b[4];
#pragma unroll
            for (uint32_t c = 0; c < 4u; c++) {
                b[c] = deepseek4_vision_load16(
                        wr + (uint64_t)m * 16u * in_dim + k0 + c * 16u);
            }
#pragma unroll
            for (uint32_t t = 0; t < 2u; t++) {
                ds4_dsv_f32x8_t part = zero;
#pragma unroll
                for (uint32_t c = 0; c < 4u; c++) {
                    part = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(
                            a[t][c], b[c], part);
                }
                acc[t][m] += part;
            }
        }
    }
#pragma unroll
    for (uint32_t t = 0; t < 2u; t++) {
#pragma unroll
        for (uint32_t i = 0; i < 8u; i++) {
            float *o = out + (uint64_t)(row0 + t * 16u + 2u * i + half) *
                             out_dim + col0 + lane16;
#pragma unroll
            for (uint32_t m = 0; m < 8u; m++) o[m * 16u] = acc[t][m][i];
        }
    }
}

/* Block projection: rows is a multiple of 32, in_dim of 64, out_dim of 128. */
static int deepseek4_vision_gemm(
        float          *out,
        const void     *model_map,
        uint64_t        model_size,
        uint64_t        weight_offset,
        uint32_t        in_dim,
        uint32_t        out_dim,
        const uint16_t *x,
        uint32_t        rows,
        const char     *label) {
    const uint16_t *w = glm53_vision_weight(
            model_map, model_size, weight_offset,
            (uint64_t)in_dim * out_dim, label);
    if (!w) return 0;
    deepseek4_vision_gemm_kernel<<<
        dim3(rows / 32u, out_dim / 128u, 1u), 32u>>>(
            out, w, x, in_dim, out_dim);
    return glm53_vision_launch_ok(label);
}

/* The patch embedding and the aligner run once per image: rocBLAS, whose BF16
 * GEMM accumulates in IEEE F32.  The patch width (588) is not a multiple of
 * 16, and on the aligner's few rows and 9216-wide weights the kernel above
 * measured no faster. */
static int deepseek4_vision_gemm_blas(
        float          *out,
        const void     *model_map,
        uint64_t        model_size,
        uint64_t        weight_offset,
        uint32_t        in_dim,
        uint32_t        out_dim,
        const uint16_t *x,
        uint32_t        rows,
        const char     *label) {
    const uint16_t *w = glm53_vision_weight(
            model_map, model_size, weight_offset,
            (uint64_t)in_dim * out_dim, label);
    if (!w || !g_cublas_ready) return 0;
    const float alpha = 1.0f, beta = 0.0f;
    return cublas_ok(cublasGemmEx(
            g_cublas, CUBLAS_OP_T, CUBLAS_OP_N,
            (int)out_dim, (int)rows, (int)in_dim, &alpha,
            w, HIPBLAS_R_16B, (int)in_dim, x, HIPBLAS_R_16B, (int)in_dim,
            &beta, out, CUDA_R_32F, (int)out_dim,
            CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT), label);
}

/* DS4_ROCM_VISION_DUMP_DIR writes every tensor of the encoder, so that
 * tests/deepseek4_vision_ref.py can check each operation against a float64
 * evaluation of the same definition. */
static void deepseek4_vision_dump(
        const char *name,
        uint32_t    index,
        const void *device,
        const void *host,
        uint64_t    bytes) {
    static const char *dir = getenv("DS4_ROCM_VISION_DUMP_DIR");
    if (!dir || !dir[0]) return;
    std::vector<char> copy;
    if (device) {
        copy.resize((size_t)bytes);
        host = copy.data();
    }
    char path[4096];
    snprintf(path, sizeof(path), "%s/%s%02u.bin", dir, name, index);
    FILE *fp = fopen(path, "wb");
    if (!fp ||
        (device && !cuda_ok(cudaMemcpy(copy.data(), device, (size_t)bytes,
                                       cudaMemcpyDeviceToHost), "vision dump")) ||
        fwrite(host, 1, (size_t)bytes, fp) != (size_t)bytes) {
        fprintf(stderr, DS4_GPU_LOG_PREFIX "cannot dump %s\n", path);
    }
    if (fp) fclose(fp);
}

static uint16_t deepseek4_vision_host_bf16(float value) {
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    if ((bits & 0x7f800000u) != 0x7f800000u)
        bits += 0x00007fffu + ((bits >> 16u) & 1u);
    else if (bits & 0x007fffffu)
        bits |= 0x00400000u;
    return (uint16_t)(bits >> 16u);
}

#define DS4_DSV_LAUNCH(kernel_, count_) \
    kernel_<<<(unsigned)(((uint64_t)(count_) + 255u) / 256u), 256u>>>

/* Encode row-major normalized 14x14 RGB patches into the natural row-major
 * 3x3-aligned grid of language-model embeddings. */
extern "C" int ds4_gpu_deepseek4_vision_encode(
        float                              *out,
        const float                        *patches,
        uint32_t                            grid_h,
        uint32_t                            grid_w,
        const void                         *model_map,
        uint64_t                            model_size,
        const ds4_deepseek4_vision_weights *weights) {
    if (!out || !patches || !model_map || !weights ||
        weights->projection_dim != DS4_DSV_OUT || grid_h == 0u ||
        grid_w == 0u || grid_h > 4096u || grid_w > 4096u) return 0;
    const uint32_t n_rows = grid_h * grid_w;
    const uint32_t rows = (n_rows + 31u) & ~31u;
    const uint32_t out_rows = ((grid_h + 2u) / 3u) * ((grid_w + 2u) / 3u);
    const uint32_t positions = std::max(grid_h, grid_w);
    const uint64_t row1024 = (uint64_t)rows * DS4_DSV_EMBD;
    const uint64_t e16 = row1024 * sizeof(uint16_t);
    const uint64_t out4096 = (uint64_t)out_rows * DS4_DSV_OUT;

    /* One arena in the scratch buffer: BF16 activations plus one F32 GEMM
     * output, about 140 MB for the largest images.  The buffer persists, so
     * the images of a request do not each pay to map it: a fresh allocation
     * per image cost 10 ms. */
    uint64_t arena = 0;
    auto reserve = [&](uint64_t bytes) {
        const uint64_t offset = arena;
        arena += (bytes + 255u) & ~UINT64_C(255);
        return offset;
    };
    const uint64_t o_patch = reserve((uint64_t)rows * DS4_DSV_PATCH * 2u);
    const uint64_t o_rope = reserve((uint64_t)positions * 16u * sizeof(float2));
    const uint64_t o_x = reserve(e16), o_n = reserve(e16);
    const uint64_t o_q = reserve(e16), o_k = reserve(e16);
    const uint64_t o_vt = reserve(e16), o_attn = reserve(e16);
    const uint64_t o_mid = reserve((uint64_t)rows * DS4_DSV_MLP * 2u);
    const uint64_t o_f32 = reserve((uint64_t)rows * 2u * DS4_DSV_MLP * 4u);
    const uint64_t o_ain = reserve((uint64_t)out_rows * DS4_DSV_ALIGN_IN * 2u);
    const uint64_t o_amid = reserve(out4096 * 2u);
    const uint64_t o_aout = reserve(out4096 * 4u);
    char *base = (char *)cuda_tmp_alloc(arena, "DeepSeek vision encoder");
    if (!base) return 0;
    uint16_t *patch = (uint16_t *)(base + o_patch);
    float2 *rope = (float2 *)(base + o_rope);
    uint16_t *x = (uint16_t *)(base + o_x), *n = (uint16_t *)(base + o_n);
    uint16_t *q = (uint16_t *)(base + o_q), *k = (uint16_t *)(base + o_k);
    uint16_t *vt = (uint16_t *)(base + o_vt);
    uint16_t *attn = (uint16_t *)(base + o_attn);
    uint16_t *mid = (uint16_t *)(base + o_mid);
    float *f32 = (float *)(base + o_f32);
    uint16_t *ain = (uint16_t *)(base + o_ain);
    uint16_t *amid = (uint16_t *)(base + o_amid);
    float *aout = (float *)(base + o_aout);

    {
        std::vector<uint16_t> host((size_t)rows * DS4_DSV_PATCH, 0u);
        for (size_t i = 0; i < (size_t)n_rows * DS4_DSV_PATCH; i++)
            host[i] = deepseek4_vision_host_bf16(patches[i]);
        if (!cuda_ok(cudaMemcpy(patch, host.data(),
                                host.size() * sizeof(uint16_t),
                                cudaMemcpyHostToDevice),
                     "DeepSeek vision patch upload")) return 0;
    }
    const uint32_t grid[2] = {grid_h, grid_w};
    deepseek4_vision_dump("grid", 0, NULL, grid, sizeof(grid));
    deepseek4_vision_dump("patch", 0, patch, NULL,
                          (uint64_t)rows * DS4_DSV_PATCH * 2u);

    const uint16_t *bias = glm53_vision_weight(
            model_map, model_size, weights->patch_bias, DS4_DSV_EMBD,
            "DeepSeek vision patch bias");
    int ok = bias && deepseek4_vision_gemm_blas(
            f32, model_map, model_size, weights->patch_weight,
            DS4_DSV_PATCH, DS4_DSV_EMBD, patch, rows, "DeepSeek vision patch");
    if (ok) {
        deepseek4_vision_dump("patch_w", 0, f32, NULL, row1024 * sizeof(float));
        DS4_DSV_LAUNCH(deepseek4_vision_bias_kernel, row1024)(
                x, f32, bias, NULL, row1024, DS4_DSV_EMBD);
        DS4_DSV_LAUNCH(deepseek4_vision_rope_table_kernel, positions * 16u)(
                rope, positions * 16u);
        ok = glm53_vision_launch_ok("DeepSeek vision patch bias");
        deepseek4_vision_dump("rope", 0, rope, NULL,
                              positions * 16u * sizeof(float2));
        deepseek4_vision_dump("x", 0, x, NULL, e16);
    }

    for (uint32_t il = 0; ok && il < DS4_DEEPSEEK4_VISION_LAYERS; il++) {
        const ds4_deepseek4_vision_layer_weights *w = &weights->layer[il];
        const uint16_t *norm1 = glm53_vision_weight(
                model_map, model_size, w->norm1, DS4_DSV_EMBD,
                "DeepSeek vision norm1");
        const uint16_t *qkv_bias = glm53_vision_weight(
                model_map, model_size, w->qkv_bias, 3u * DS4_DSV_EMBD,
                "DeepSeek vision QKV bias");
        const uint16_t *proj_bias = glm53_vision_weight(
                model_map, model_size, w->attn_proj_bias, DS4_DSV_EMBD,
                "DeepSeek vision attention bias");
        const uint16_t *norm2 = glm53_vision_weight(
                model_map, model_size, w->norm2, DS4_DSV_EMBD,
                "DeepSeek vision norm2");
        if (!norm1 || !qkv_bias || !proj_bias || !norm2) { ok = 0; break; }

        deepseek4_vision_rms_kernel<<<rows, 256u>>>(n, x, norm1);
        deepseek4_vision_dump("norm1_", il, n, NULL, e16);
        ok = glm53_vision_launch_ok("DeepSeek vision norm1") &&
             deepseek4_vision_gemm(
                f32, model_map, model_size, w->qkv_weight,
                DS4_DSV_EMBD, 3u * DS4_DSV_EMBD, n, rows, "DeepSeek vision QKV");
        if (!ok) break;
        deepseek4_vision_dump("qkv_", il, f32, NULL, row1024 * 3u * sizeof(float));
        deepseek4_vision_qk_rope_kernel<<<
            dim3(DS4_DSV_HEADS, rows, 1u), 32u>>>(
                q, k, f32, qkv_bias, rope, grid_w, n_rows, rows);
        deepseek4_vision_v_kernel<<<
            dim3(DS4_DSV_HEADS, (rows + 63u) / 64u, 1u), 256u>>>(
                vt, f32, qkv_bias, n_rows, rows);
        deepseek4_vision_attention_kernel<<<
            dim3(rows / 16u, DS4_DSV_HEADS, 1u), 32u>>>(
                attn, q, k, vt, n_rows, rows);
        ok = glm53_vision_launch_ok("DeepSeek vision attention");
        deepseek4_vision_dump("q_", il, q, NULL, e16);
        deepseek4_vision_dump("k_", il, k, NULL, e16);
        deepseek4_vision_dump("vt_", il, vt, NULL, e16);
        deepseek4_vision_dump("attn_", il, attn, NULL, e16);
        ok = ok && deepseek4_vision_gemm(
                f32, model_map, model_size, w->attn_proj_weight,
                DS4_DSV_EMBD, DS4_DSV_EMBD, attn, rows,
                "DeepSeek vision attention output");
        if (!ok) break;
        deepseek4_vision_dump("proj_", il, f32, NULL, row1024 * sizeof(float));
        DS4_DSV_LAUNCH(deepseek4_vision_bias_kernel, row1024)(
                x, f32, proj_bias, x, row1024, DS4_DSV_EMBD);
        deepseek4_vision_dump("xattn_", il, x, NULL, e16);
        deepseek4_vision_rms_kernel<<<rows, 256u>>>(n, x, norm2);
        deepseek4_vision_dump("norm2_", il, n, NULL, e16);
        ok = glm53_vision_launch_ok("DeepSeek vision norm2") &&
             deepseek4_vision_gemm(
                f32, model_map, model_size, w->mlp_w1,
                DS4_DSV_EMBD, 2u * DS4_DSV_MLP, n, rows,
                "DeepSeek vision MLP input");
        if (!ok) break;
        deepseek4_vision_dump("w1_", il, f32, NULL,
                              (uint64_t)rows * 2u * DS4_DSV_MLP * sizeof(float));
        DS4_DSV_LAUNCH(deepseek4_vision_swiglu_kernel,
                       (uint64_t)rows * DS4_DSV_MLP)(
                mid, f32, (uint64_t)rows * DS4_DSV_MLP);
        deepseek4_vision_dump("mid_", il, mid, NULL,
                              (uint64_t)rows * DS4_DSV_MLP * 2u);
        ok = glm53_vision_launch_ok("DeepSeek vision SwiGLU") &&
             deepseek4_vision_gemm(
                f32, model_map, model_size, w->mlp_w2,
                DS4_DSV_MLP, DS4_DSV_EMBD, mid, rows,
                "DeepSeek vision MLP output");
        if (!ok) break;
        deepseek4_vision_dump("w2_", il, f32, NULL, row1024 * sizeof(float));
        DS4_DSV_LAUNCH(deepseek4_vision_residual_kernel, row1024)(
                x, f32, row1024);
        ok = glm53_vision_launch_ok("DeepSeek vision MLP residual");
        deepseek4_vision_dump("x", il + 1u, x, NULL, e16);
    }

    const uint16_t *post_norm = NULL, *bias1 = NULL, *bias2 = NULL;
    if (ok) {
        post_norm = glm53_vision_weight(
                model_map, model_size, weights->post_norm, DS4_DSV_EMBD,
                "DeepSeek vision final norm");
        bias1 = glm53_vision_weight(
                model_map, model_size, weights->aligner_w1_bias, DS4_DSV_OUT,
                "DeepSeek vision aligner hidden bias");
        bias2 = glm53_vision_weight(
                model_map, model_size, weights->aligner_w2_bias, DS4_DSV_OUT,
                "DeepSeek vision aligner output bias");
        ok = post_norm && bias1 && bias2;
    }
    if (ok) {
        deepseek4_vision_rms_kernel<<<rows, 256u>>>(n, x, post_norm);
        deepseek4_vision_dump("norm", 0, n, NULL, e16);
        deepseek4_vision_aligner_reorder_kernel<<<
            dim3((DS4_DSV_ALIGN_IN + 255u) / 256u, out_rows, 1u), 256u>>>(
                ain, n, grid_h, grid_w);
        ok = glm53_vision_launch_ok("DeepSeek vision aligner reorder") &&
             deepseek4_vision_gemm_blas(
                aout, model_map, model_size, weights->aligner_w1,
                DS4_DSV_ALIGN_IN, DS4_DSV_OUT, ain, out_rows,
                "DeepSeek vision aligner hidden");
    }
    if (ok) {
        deepseek4_vision_dump("align_in", 0, ain, NULL,
                              (uint64_t)out_rows * DS4_DSV_ALIGN_IN * 2u);
        deepseek4_vision_dump("align_w1_", 0, aout, NULL, out4096 * sizeof(float));
        DS4_DSV_LAUNCH(deepseek4_vision_gelu_bias_kernel, out4096)(
                amid, aout, bias1, out4096);
        deepseek4_vision_dump("align_mid", 0, amid, NULL, out4096 * 2u);
        ok = glm53_vision_launch_ok("DeepSeek vision aligner GELU") &&
             deepseek4_vision_gemm_blas(
                aout, model_map, model_size, weights->aligner_w2,
                DS4_DSV_OUT, DS4_DSV_OUT, amid, out_rows,
                "DeepSeek vision aligner output");
    }
    if (ok) {
        deepseek4_vision_dump("align_w2_", 0, aout, NULL, out4096 * sizeof(float));
        DS4_DSV_LAUNCH(deepseek4_vision_output_kernel, out4096)(
                aout, bias2, out4096);
        ok = glm53_vision_launch_ok("DeepSeek vision aligner output bias");
    }
    /* The synchronous copy also waits for the kernels above. */
    if (ok) {
        ok = cuda_ok(cudaMemcpy(out, aout, out4096 * sizeof(float),
                                cudaMemcpyDeviceToHost),
                     "DeepSeek vision output read");
        deepseek4_vision_dump("out", 0, NULL, out, out4096 * sizeof(float));
    } else {
        (void)cudaDeviceSynchronize();
    }
    return ok;
}

#undef DS4_DSV_LAUNCH
