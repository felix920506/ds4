// DS4 ROCm common embedding/dense-matmul kernels and device helpers.
//
// Included from ds4_cuda.cu before more specialized modules; these helpers are
// intentionally kept static in the single translation unit.

static int ds4_rocm_is_gfx1151(void) {
    static int cached = -1;
    if (cached >= 0) return cached;
    int dev = 0;
    cudaDeviceProp prop;
    cached = cudaGetDevice(&dev) == cudaSuccess &&
             cudaGetDeviceProperties(&prop, dev) == cudaSuccess &&
             strncmp(prop.gcnArchName, "gfx1151", 7) == 0 &&
             (prop.gcnArchName[7] == '\0' || prop.gcnArchName[7] == ':');
    return cached;
}

/* The tuned paths are the production default on gfx1151. Setting an
 * individual selector to 0 keeps a simple fallback/debug escape hatch. */
static int ds4_rocm_gfx1151_flag(const char *name) {
    const char *env = getenv(name);
    return env ? env[0] != '0' : ds4_rocm_is_gfx1151();
}

__global__ static void fill_f32_kernel(float *x, uint64_t n, float v) {
    uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] = v;
}

__global__ static void embed_token_hc_kernel(float *out, const unsigned short *w, uint32_t token, uint32_t n_embd, uint32_t n_hc) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t n = n_embd * n_hc;
    if (i >= n) return;
    uint32_t e = i % n_embd;
    out[i] = __half2float(reinterpret_cast<const __half *>(w)[(uint64_t)token * n_embd + e]);
}

__device__ static float embed_q8_0_scale(const unsigned char *blk) {
    const uint16_t bits = (uint16_t)blk[0] | ((uint16_t)blk[1] << 8);
    return __half2float(__ushort_as_half((unsigned short)bits));
}

__global__ static void embed_token_hc_q8_0_kernel(
        float *out,
        const unsigned char *w,
        uint32_t token,
        uint32_t n_embd,
        uint32_t n_hc) {
    uint64_t gid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t n = (uint64_t)n_embd * n_hc;
    if (gid >= n) return;
    const uint32_t d = gid % n_embd;
    const uint32_t blocks = (n_embd + 31u) / 32u;
    const uint32_t b = d >> 5u;
    const uint32_t j = d & 31u;
    const unsigned char *blk = w + ((uint64_t)token * blocks + b) * 34u;
    out[gid] = embed_q8_0_scale(blk) * (float)((const int8_t *)(blk + 2u))[j];
}

__global__ static void embed_tokens_hc_q8_0_kernel(
        float *out,
        const int32_t *tokens,
        const unsigned char *w,
        uint32_t n_vocab,
        uint32_t n_tokens,
        uint32_t n_embd,
        uint32_t n_hc) {
    uint64_t gid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t n = (uint64_t)n_tokens * n_hc * n_embd;
    if (gid >= n) return;
    const uint32_t d = gid % n_embd;
    uint64_t tmp = gid / n_embd;
    const uint32_t t = tmp / n_hc;
    int32_t tok_i = tokens[t];
    uint32_t tok = tok_i < 0 ? 0u : (uint32_t)tok_i;
    if (tok >= n_vocab) tok = 0;
    const uint32_t blocks = (n_embd + 31u) / 32u;
    const uint32_t b = d >> 5u;
    const uint32_t j = d & 31u;
    const unsigned char *blk = w + ((uint64_t)tok * blocks + b) * 34u;
    out[gid] = embed_q8_0_scale(blk) * (float)((const int8_t *)(blk + 2u))[j];
}

__global__ static void embed_tokens_hc_kernel(
        float *out,
        const int32_t *tokens,
        const __half *w,
        uint32_t n_vocab,
        uint32_t n_tokens,
        uint32_t n_embd,
        uint32_t n_hc) {
    uint64_t gid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t n = (uint64_t)n_tokens * n_hc * n_embd;
    if (gid >= n) return;
    uint32_t d = gid % n_embd;
    uint64_t tmp = gid / n_embd;
    uint32_t t = tmp / n_hc;
    int32_t tok_i = tokens[t];
    uint32_t tok = tok_i < 0 ? 0u : (uint32_t)tok_i;
    if (tok >= n_vocab) tok = 0;
    out[gid] = __half2float(w[(uint64_t)tok * n_embd + d]);
}

__device__ static float warp_sum_f32(float v);

__global__ static void matmul_f16_kernel(
        float *out,
        const __half *w,
        const float *x,
        uint64_t in_dim,
        uint64_t out_dim,
        uint64_t n_tok) {
    uint64_t row = (uint64_t)blockIdx.x;
    uint64_t tok = (uint64_t)blockIdx.y;
    if (row >= out_dim || tok >= n_tok) return;

    float sum = 0.0f;
    const __half *wr = w + row * in_dim;
    const float *xr = x + tok * in_dim;
    for (uint64_t i = threadIdx.x; i < in_dim; i += blockDim.x) {
        sum += __half2float(wr[i]) * xr[i];
    }

    __shared__ float partial[256];
    partial[threadIdx.x] = sum;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) out[tok * out_dim + row] = partial[0];
}

/* Each of 32 threads sums one contiguous chunk of the row in order, and the
 * chunk sums are added in order.  A wave waits for memory on every load it
 * consumes at once, so a chunk is fetched eight weights at a time ahead of
 * the sum, which stays one fused multiply-add per element. */
__global__ static void matmul_f16_ordered_chunks_kernel(
        float *out,
        const __half *w,
        const float *x,
        uint64_t in_dim,
        uint64_t out_dim,
        uint64_t n_tok) {
    uint64_t row = (uint64_t)blockIdx.x;
    uint64_t tok = (uint64_t)blockIdx.y;
    if (row >= out_dim || tok >= n_tok) return;

    __shared__ float partial[32];
    const uint32_t tid = threadIdx.x;
    float sum = 0.0f;
    const uint64_t chunk = (in_dim + 31u) / 32u;
    const uint64_t k0 = (uint64_t)tid * chunk;
    uint64_t k1 = k0 + chunk;
    if (k1 > in_dim) k1 = in_dim;
    const __half *wr = w + row * in_dim + k0;
    const float *xr = x + tok * in_dim + k0;
    const uint32_t count = k1 > k0 ? (uint32_t)(k1 - k0) : 0u;
    uint32_t i = 0;
#pragma unroll 2
    for (; i + 8u <= count; i += 8u) {
        __half wv[8];
        float xv[8];
        __builtin_memcpy(wv, wr + i, sizeof(wv));
        __builtin_memcpy(xv, xr + i, sizeof(xv));
#pragma unroll
        for (uint32_t j = 0; j < 8u; j++) sum = __builtin_fmaf(__half2float(wv[j]), xv[j], sum);
    }
    for (; i < count; i++) sum = __builtin_fmaf(__half2float(wr[i]), xr[i], sum);
    partial[tid] = sum;
    __syncthreads();
    if (tid == 0) {
        float total = 0.0f;
        for (uint32_t i = 0; i < 32u; i++) total += partial[i];
        out[tok * out_dim + row] = total;
    }
}

/* How many waves stream weights at once in the single-token row kernels.
 *
 * A decode projection reads each weight once, so it is bound by memory, and
 * on gfx1151 memory throughput peaks when each compute unit keeps only a few
 * kilobytes of weight lines in flight.  One wave per output row puts thirty
 * waves on a unit; their lines overflow its 32 KiB L1 and are fetched again,
 * which costs the F16 projections a third of their throughput.  The row
 * kernels are therefore launched as this many single-wave blocks that walk
 * the rows in turn, each wave asking for about 2 KiB of weights per step. */
#define DS4_ROCM_DECODE_STREAMS 120u

/* Single-token F16 projections: one (NW = 1) or two (NW = 2) weight matrices
 * against the same activations.  Launch with 2 * DS4_ROCM_DECODE_STREAMS / NW
 * blocks of one wave (a step reads 1 KiB of each matrix); block b computes
 * rows b, b + gridDim.x, ...
 *
 * Lane l sums elements l, l + 32, ... in order and the lanes reduce pairwise.
 * The lane offset is folded into the row pointers so that the loop index is
 * uniform across the wave and the loads need no per-lane address arithmetic. */
template <int NW>
__global__ static void matmul_f16_rows_w32_kernel(
        float *out0,
        float *out1,
        const __half *w0,
        const __half *w1,
        const float *x,
        uint32_t in_dim,
        uint32_t out_dim) {
    const uint32_t lane = threadIdx.x;
    const uint32_t full = in_dim & ~31u;
    const float *xr = x + lane;
    for (uint32_t row = blockIdx.x; row < out_dim; row += gridDim.x) {
        const __half *wr[2] = { w0 + (uint64_t)row * in_dim + lane,
                                w1 + (uint64_t)row * in_dim + lane };
        float acc[NW] = {};
#pragma unroll 16
        for (uint32_t i = 0; i < full; i += 32u) {
            const float xv = xr[i];
#pragma unroll
            for (int n = 0; n < NW; n++) acc[n] = __builtin_fmaf(__half2float(wr[n][i]), xv, acc[n]);
        }
        if (lane < in_dim - full) {
            const float xv = xr[full];
#pragma unroll
            for (int n = 0; n < NW; n++) acc[n] = __builtin_fmaf(__half2float(wr[n][full]), xv, acc[n]);
        }
#pragma unroll
        for (int n = 0; n < NW; n++) acc[n] = warp_sum_f32(acc[n]);
        if (lane == 0u) {
            out0[row] = acc[0];
            if (NW > 1) out1[row] = acc[NW - 1];
        }
    }
}

__global__ static void matmul_f16_pair_ordered_chunks_kernel(
        float *out0,
        float *out1,
        const __half *w0,
        const __half *w1,
        const float *x,
        uint64_t in_dim,
        uint64_t out0_dim,
        uint64_t out1_dim) {
    uint64_t row = (uint64_t)blockIdx.x;
    if (row >= out0_dim && row >= out1_dim) return;

    __shared__ float partial0[32];
    __shared__ float partial1[32];
    const uint32_t tid = threadIdx.x;
    float sum0 = 0.0f;
    float sum1 = 0.0f;
    const uint64_t chunk = (in_dim + 31u) / 32u;
    const uint64_t k0 = (uint64_t)tid * chunk;
    uint64_t k1 = k0 + chunk;
    if (k1 > in_dim) k1 = in_dim;
    const __half *wr0 = row < out0_dim ? w0 + row * in_dim : w0;
    const __half *wr1 = row < out1_dim ? w1 + row * in_dim : w1;
    for (uint64_t i = k0; i < k1; i++) {
        const float xv = x[i];
        if (row < out0_dim) sum0 += __half2float(wr0[i]) * xv;
        if (row < out1_dim) sum1 += __half2float(wr1[i]) * xv;
    }
    partial0[tid] = sum0;
    partial1[tid] = sum1;
    __syncthreads();
    if (tid == 0) {
        float total0 = 0.0f;
        float total1 = 0.0f;
        for (uint32_t i = 0; i < 32u; i++) {
            total0 += partial0[i];
            total1 += partial1[i];
        }
        if (row < out0_dim) out0[row] = total0;
        if (row < out1_dim) out1[row] = total1;
    }
}

__global__ static void matmul_f32_kernel(
        float *out,
        const float *w,
        const float *x,
        uint64_t in_dim,
        uint64_t out_dim,
        uint64_t n_tok) {
    uint64_t row = (uint64_t)blockIdx.x;
    uint64_t tok = (uint64_t)blockIdx.y;
    if (row >= out_dim || tok >= n_tok) return;

    float sum = 0.0f;
    const float *wr = w + row * in_dim;
    const float *xr = x + tok * in_dim;
    for (uint64_t i = threadIdx.x; i < in_dim; i += blockDim.x) {
        sum += wr[i] * xr[i];
    }

    __shared__ float partial[256];
    partial[threadIdx.x] = sum;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) out[tok * out_dim + row] = partial[0];
}

__global__ static void repeat_hc_kernel(float *out, const float *row, uint32_t n_embd, uint32_t n_hc) {
    uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t n = (uint64_t)n_embd * n_hc;
    if (i >= n) return;
    out[i] = row[i % n_embd];
}

__global__ static void repeat_hc_rows_kernel(float *out, const float *rows, uint32_t n_tokens, uint32_t n_embd, uint32_t n_hc) {
    uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t n = (uint64_t)n_tokens * n_hc * n_embd;
    if (i >= n) return;

    uint64_t hc_row = (uint64_t)n_hc * n_embd;
    uint64_t tok = i / hc_row;
    uint64_t embd = i % n_embd;
    out[i] = rows[tok * n_embd + embd];
}

__global__ static void pack_slot_rows_f32_kernel(float *out, const float *slots, uint32_t n_rows, uint32_t width, uint32_t n_slots, uint32_t slot_cap) {
    uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t n = (uint64_t)n_rows * n_slots * width;
    if (i >= n) return;

    uint64_t col = i % width;
    uint64_t slot = (i / width) % n_slots;
    uint64_t row = i / ((uint64_t)n_slots * width);
    out[i] = slots[((slot * slot_cap) + row) * width + col];
}

__global__ static void f32_to_f16_kernel(__half *out, const float *x, uint64_t n) {
    uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = __float2half(x[i]);
}

__device__ static float warp_sum_f32(float v) {
    for (int offset = 16; offset > 0; offset >>= 1) {
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
        v += __shfl_down(v, offset, 32);
#else
        v += __shfl_down_sync(FULL_WARP_MASK, v, offset, 32);
#endif
    }
    return v;
}

__device__ static float warp_max_f32(float v) {
    for (int offset = 16; offset > 0; offset >>= 1) {
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
        v = fmaxf(v, __shfl_down(v, offset, 32));
#else
        v = fmaxf(v, __shfl_down_sync(FULL_WARP_MASK, v, offset, 32));
#endif
    }
    return v;
}

__device__ static uint16_t f32_to_f16_bits_hip_round(float f) {
    union { float f; uint32_t u; } v;
    v.f = f;
    uint32_t sign = (v.u >> 16) & 0x8000u;
    int32_t exp = (int32_t)((v.u >> 23) & 0xffu) - 127 + 15;
    uint32_t mant = v.u & 0x7fffffu;
    if (exp <= 0) {
        if (exp < -10) return (uint16_t)sign;
        mant |= 0x800000u;
        uint32_t shift = (uint32_t)(14 - exp);
        uint32_t half_mant = mant >> shift;
        if ((mant >> (shift - 1)) & 1u) half_mant++;
        return (uint16_t)(sign | half_mant);
    }
    if (exp >= 31) return (uint16_t)(sign | 0x7c00u);
    uint32_t half = sign | ((uint32_t)exp << 10) | (mant >> 13);
    if (mant & 0x1000u) half++;
    return (uint16_t)half;
}

__device__ static float f16_bits_to_f32(uint16_t bits) {
    return __half2float(__ushort_as_half((unsigned short)bits));
}

__device__ static float dot4_f32(float4 a, float4 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
}
