// DS4 ROCm routed-MoE quantization/device helpers and kernels.
//
// Included from the ROCm backend translation unit before
// ds4_rocm_moe_launch.cuh so host policy/glue can keep using these static kernels directly.

#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
#include <rocwmma/rocwmma.hpp>
#endif

__global__ static void moe_swiglu_weighted_f32_kernel(
        float * __restrict__ mid,
        const float * __restrict__ gate,
        const float * __restrict__ up,
        const float * __restrict__ weights,
        uint64_t n,
        uint32_t expert_mid_dim,
        float clamp) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const uint64_t pair = i / expert_mid_dim;
    float g = isfinite(gate[i]) ? gate[i] : 0.0f;
    float u = isfinite(up[i]) ? up[i] : 0.0f;
    if (clamp > 1.0e-6f) {
        g = fminf(g, clamp);
        u = fminf(fmaxf(u, -clamp), clamp);
    }
    mid[i] = (g / (1.0f + expf(-g))) * u * weights[pair];
}

__device__ static float dev_f16_to_f32(uint16_t v) {
    return __half2float(*reinterpret_cast<const __half *>(&v));
}

__device__ __forceinline__ static uint32_t dev_pack_half2_bits(float x, float y) {
    const __half2 h = __floats2half2_rn(x, y);
    return *reinterpret_cast<const uint32_t *>(&h);
}

__device__ __forceinline__ static uint32_t dev_unpack_iq2_signs(uint32_t v) {
    const uint32_t p = __popc(v) & 1u;
    const uint32_t s = v ^ (p << 7u);
    return s * 0x01010101u;
}

__device__ __forceinline__ static int32_t dev_iq2_dp4a_8(uint64_t grid, uint32_t sign, const int8_t *q8, int32_t acc) {
    const uint32_t signs = dev_unpack_iq2_signs(sign);
    const int32_t sm0 = __vcmpne4(signs & 0x08040201u, 0);
    const int32_t sm1 = __vcmpne4(signs & 0x80402010u, 0);
    const int32_t g0 = __vsub4((int32_t)(uint32_t)grid ^ sm0, sm0);
    const int32_t g1 = __vsub4((int32_t)(uint32_t)(grid >> 32) ^ sm1, sm1);
    acc = __dp4a(g0, *(const int32_t *)(q8 + 0), acc);
    acc = __dp4a(g1, *(const int32_t *)(q8 + 4), acc);
    return acc;
}

__device__ static int32_t dev_dot_q2_16(const uint8_t *q2, const int8_t *q8, int shift) {
    int32_t sum = 0;
    #pragma unroll
    for (uint32_t i = 0; i < 16; i += 4) {
        const int32_t v = (*(const int32_t *)(q2 + i) >> shift) & 0x03030303;
        sum = __dp4a(v, *(const int32_t *)(q8 + i), sum);
    }
    return sum;
}

__device__ static int32_t dev_dot_iq2_pair_16(uint8_t grid0, uint32_t sign0, uint8_t grid1, uint32_t sign1, const int8_t *q8) {
    int32_t sum = 0;
    sum = dev_iq2_dp4a_8(cuda_iq2xxs_grid[grid0], cuda_ksigns_iq2xs[sign0], q8, sum);
    sum = dev_iq2_dp4a_8(cuda_iq2xxs_grid[grid1], cuda_ksigns_iq2xs[sign1], q8 + 8, sum);
    return sum;
}

__device__ __forceinline__ static void dev_iq2_i8x8_lut(
        const uint64_t *grid,
        const uint8_t *signs,
        uint8_t grid_idx,
        uint32_t sign_idx,
        int32_t *w0,
        int32_t *w1) {
    const uint32_t s = dev_unpack_iq2_signs(signs[sign_idx]);
    const int32_t sm0 = __vcmpne4(s & 0x08040201u, 0);
    const int32_t sm1 = __vcmpne4(s & 0x80402010u, 0);
    const uint64_t g = grid[grid_idx];
    *w0 = __vsub4((int32_t)(uint32_t)g ^ sm0, sm0);
    *w1 = __vsub4((int32_t)(uint32_t)(g >> 32) ^ sm1, sm1);
}

__device__ static float dev_dot_iq2_xxs_q8_K_block(const cuda_block_iq2_xxs *x, const cuda_block_q8_K *y) {
    const float d = dev_f16_to_f32(x->d) * y->d;
    const uint16_t *q2 = x->qs;
    const int8_t *q8 = y->qs;
    int32_t bsum = 0;
    for (int ib32 = 0; ib32 < CUDA_QK_K / 32; ib32++) {
        const uint32_t aux0 = (uint32_t)q2[0] | ((uint32_t)q2[1] << 16);
        const uint32_t aux1 = (uint32_t)q2[2] | ((uint32_t)q2[3] << 16);
        q2 += 4;
        const uint32_t ls = 2u * (aux1 >> 28) + 1u;
        const uint8_t a0 = (uint8_t)(aux0 & 0xffu);
        const uint8_t a1 = (uint8_t)((aux0 >> 8) & 0xffu);
        const uint8_t a2 = (uint8_t)((aux0 >> 16) & 0xffu);
        const uint8_t a3 = (uint8_t)((aux0 >> 24) & 0xffu);
        int32_t sumi = 0;
        sumi += dev_dot_iq2_pair_16(a0, (aux1 >> 0) & 127u, a1, (aux1 >> 7) & 127u, q8);
        q8 += 16;
        sumi += dev_dot_iq2_pair_16(a2, (aux1 >> 14) & 127u, a3, (aux1 >> 21) & 127u, q8);
        q8 += 16;
        bsum += sumi * (int32_t)ls;
    }
    return 0.125f * d * (float)bsum;
}

__device__ static void dev_dot_iq2_xxs_q8_K_block8_deq_lut(
        const cuda_block_iq2_xxs *x,
        const cuda_block_q8_K *y0,
        const cuda_block_q8_K *y1,
        const cuda_block_q8_K *y2,
        const cuda_block_q8_K *y3,
        const cuda_block_q8_K *y4,
        const cuda_block_q8_K *y5,
        const cuda_block_q8_K *y6,
        const cuda_block_q8_K *y7,
        uint32_t n,
        float acc[8],
        const uint64_t *grid,
        const uint8_t *signs) {
    const float xd = dev_f16_to_f32(x->d);
    const uint16_t *q2 = x->qs;
    int32_t bsum[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    const int8_t *q8[8] = {
        y0 ? y0->qs : NULL, y1 ? y1->qs : NULL, y2 ? y2->qs : NULL, y3 ? y3->qs : NULL,
        y4 ? y4->qs : NULL, y5 ? y5->qs : NULL, y6 ? y6->qs : NULL, y7 ? y7->qs : NULL,
    };
    for (int ib32 = 0; ib32 < CUDA_QK_K / 32; ib32++) {
        const uint32_t aux0 = (uint32_t)q2[0] | ((uint32_t)q2[1] << 16);
        const uint32_t aux1 = (uint32_t)q2[2] | ((uint32_t)q2[3] << 16);
        q2 += 4;
        const int32_t ls = (int32_t)(2u * (aux1 >> 28) + 1u);
        int32_t w[8];
        dev_iq2_i8x8_lut(grid, signs, (uint8_t)(aux0 & 0xffu),           (aux1 >> 0)  & 127u, &w[0], &w[1]);
        dev_iq2_i8x8_lut(grid, signs, (uint8_t)((aux0 >> 8)  & 0xffu),   (aux1 >> 7)  & 127u, &w[2], &w[3]);
        dev_iq2_i8x8_lut(grid, signs, (uint8_t)((aux0 >> 16) & 0xffu),   (aux1 >> 14) & 127u, &w[4], &w[5]);
        dev_iq2_i8x8_lut(grid, signs, (uint8_t)((aux0 >> 24) & 0xffu),   (aux1 >> 21) & 127u, &w[6], &w[7]);
        for (uint32_t p = 0; p < n; p++) {
            const int8_t *q = q8[p] + ib32 * 32;
            int32_t sumi = 0;
            sumi = __dp4a(w[0], *(const int32_t *)(q + 0),  sumi);
            sumi = __dp4a(w[1], *(const int32_t *)(q + 4),  sumi);
            sumi = __dp4a(w[2], *(const int32_t *)(q + 8),  sumi);
            sumi = __dp4a(w[3], *(const int32_t *)(q + 12), sumi);
            sumi = __dp4a(w[4], *(const int32_t *)(q + 16), sumi);
            sumi = __dp4a(w[5], *(const int32_t *)(q + 20), sumi);
            sumi = __dp4a(w[6], *(const int32_t *)(q + 24), sumi);
            sumi = __dp4a(w[7], *(const int32_t *)(q + 28), sumi);
            bsum[p] += sumi * ls;
        }
    }
    const cuda_block_q8_K *ys[8] = { y0, y1, y2, y3, y4, y5, y6, y7 };
    for (uint32_t p = 0; p < n; p++) acc[p] += 0.125f * xd * ys[p]->d * (float)bsum[p];
}

__device__ static void dev_dot_iq2_xxs_q8_K_block4(
        const cuda_block_iq2_xxs *x,
        const cuda_block_q8_K *y0,
        const cuda_block_q8_K *y1,
        const cuda_block_q8_K *y2,
        const cuda_block_q8_K *y3,
        uint32_t n,
        float acc[4]) {
    const float xd = dev_f16_to_f32(x->d);
    const uint16_t *q2 = x->qs;
    int32_t bsum[4] = {0, 0, 0, 0};
    const int8_t *q8[4] = {
        y0 ? y0->qs : NULL,
        y1 ? y1->qs : NULL,
        y2 ? y2->qs : NULL,
        y3 ? y3->qs : NULL,
    };
    for (int ib32 = 0; ib32 < CUDA_QK_K / 32; ib32++) {
        const uint32_t aux0 = (uint32_t)q2[0] | ((uint32_t)q2[1] << 16);
        const uint32_t aux1 = (uint32_t)q2[2] | ((uint32_t)q2[3] << 16);
        q2 += 4;
        const uint32_t ls = 2u * (aux1 >> 28) + 1u;
        const uint8_t a0 = (uint8_t)(aux0 & 0xffu);
        const uint8_t a1 = (uint8_t)((aux0 >> 8) & 0xffu);
        const uint8_t a2 = (uint8_t)((aux0 >> 16) & 0xffu);
        const uint8_t a3 = (uint8_t)((aux0 >> 24) & 0xffu);
        for (uint32_t p = 0; p < n; p++) {
            int32_t sumi = 0;
            sumi += dev_dot_iq2_pair_16(a0, (aux1 >> 0) & 127u, a1, (aux1 >> 7) & 127u, q8[p] + ib32 * 32);
            sumi += dev_dot_iq2_pair_16(a2, (aux1 >> 14) & 127u, a3, (aux1 >> 21) & 127u, q8[p] + ib32 * 32 + 16);
            bsum[p] += sumi * (int32_t)ls;
        }
    }
    const cuda_block_q8_K *ys[4] = { y0, y1, y2, y3 };
    for (uint32_t p = 0; p < n; p++) acc[p] += 0.125f * xd * ys[p]->d * (float)bsum[p];
}

__device__ static DS4_ROCM_UNUSED void dev_dot_iq2_xxs_q8_K_block8(
        const cuda_block_iq2_xxs *x,
        const cuda_block_q8_K *y0,
        const cuda_block_q8_K *y1,
        const cuda_block_q8_K *y2,
        const cuda_block_q8_K *y3,
        const cuda_block_q8_K *y4,
        const cuda_block_q8_K *y5,
        const cuda_block_q8_K *y6,
        const cuda_block_q8_K *y7,
        uint32_t n,
        float acc[8]) {
    const float xd = dev_f16_to_f32(x->d);
    const uint16_t *q2 = x->qs;
    int32_t bsum[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    const int8_t *q8[8] = {
        y0 ? y0->qs : NULL, y1 ? y1->qs : NULL, y2 ? y2->qs : NULL, y3 ? y3->qs : NULL,
        y4 ? y4->qs : NULL, y5 ? y5->qs : NULL, y6 ? y6->qs : NULL, y7 ? y7->qs : NULL,
    };
    for (int ib32 = 0; ib32 < CUDA_QK_K / 32; ib32++) {
        const uint32_t aux0 = (uint32_t)q2[0] | ((uint32_t)q2[1] << 16);
        const uint32_t aux1 = (uint32_t)q2[2] | ((uint32_t)q2[3] << 16);
        q2 += 4;
        const uint32_t ls = 2u * (aux1 >> 28) + 1u;
        const uint8_t a0 = (uint8_t)(aux0 & 0xffu);
        const uint8_t a1 = (uint8_t)((aux0 >> 8) & 0xffu);
        const uint8_t a2 = (uint8_t)((aux0 >> 16) & 0xffu);
        const uint8_t a3 = (uint8_t)((aux0 >> 24) & 0xffu);
        for (uint32_t p = 0; p < n; p++) {
            int32_t sumi = 0;
            sumi += dev_dot_iq2_pair_16(a0, (aux1 >> 0) & 127u, a1, (aux1 >> 7) & 127u, q8[p] + ib32 * 32);
            sumi += dev_dot_iq2_pair_16(a2, (aux1 >> 14) & 127u, a3, (aux1 >> 21) & 127u, q8[p] + ib32 * 32 + 16);
            bsum[p] += sumi * (int32_t)ls;
        }
    }
    const cuda_block_q8_K *ys[8] = { y0, y1, y2, y3, y4, y5, y6, y7 };
    for (uint32_t p = 0; p < n; p++) acc[p] += 0.125f * xd * ys[p]->d * (float)bsum[p];
}

__device__ static void dev_q4_K_get_scale_min(
        uint32_t j,
        const uint8_t *scales,
        uint8_t *d_out,
        uint8_t *m_out) {
    if (j < 4u) {
        *d_out = scales[j] & 63u;
        *m_out = scales[j + 4u] & 63u;
    } else {
        *d_out = (scales[j + 4u] & 0x0fu) | ((scales[j - 4u] >> 6u) << 4u);
        *m_out = (scales[j + 4u] >> 4u) | ((scales[j] >> 6u) << 4u);
    }
}

__device__ __forceinline__ static int32_t dev_dot_q4_32(const uint8_t *qs, const int8_t *q8, int shift) {
    int32_t sum = 0;
    #pragma unroll
    for (uint32_t i = 0; i < 32u; i += 4u) {
        const int32_t v = (*(const int32_t *)(qs + i) >> shift) & 0x0f0f0f0f;
        sum = __dp4a(v, *(const int32_t *)(q8 + i), sum);
    }
    return sum;
}

__device__ static float dev_dot_q4_K_q8_K_block(const cuda_block_q4_K *x, const cuda_block_q8_K *y) {
    const float xd = dev_f16_to_f32(x->d);
    const float xmin = dev_f16_to_f32(x->dmin);
    int isum = 0;
    int summs = 0;
    #pragma unroll
    for (uint32_t j = 0; j < 8u; j++) {
        uint8_t sc, m;
        dev_q4_K_get_scale_min(j, x->scales, &sc, &m);
        summs += (int)m * (int)(y->bsums[2u * j] + y->bsums[2u * j + 1u]);
        const uint32_t byte_off = (j >> 1u) * 32u;
        const int shift = (j & 1u) ? 4 : 0;
        isum += (int)sc * dev_dot_q4_32(x->qs + byte_off, y->qs + j * 32u, shift);
    }
    return y->d * xd * (float)isum - y->d * xmin * (float)summs;
}

__device__ __forceinline__ static float dev_e8m0_to_f32(uint8_t e) {
    const uint32_t bits = e == 0u ? 0x00400000u : (uint32_t)e << 23u;
    return __uint_as_float(bits);
}

/* Expand four MXFP4 codes to signed, doubled int8 values in one register.
 * On AMDGPU, V_PERM_B32 treats selector bytes 0..3 as bytes from src1,
 * 4..7 as bytes from src0, 12 as zero, and 13 as 0xff.  Keeping both
 * halves of the tiny value table in SGPR-immediate operands avoids four
 * divergent global LUT reads for every packed word. */
__device__ __forceinline__ static uint32_t dev_mxfp4_unpack4(uint32_t codes) {
#if defined(__AMDGCN__)
    const uint32_t selectors = codes & 0x07070707u;
    const uint32_t positive = __builtin_amdgcn_perm(
        0x0c080604u, 0x03020100u, selectors);
    const uint32_t negative = __builtin_amdgcn_perm(
        0xf4f8fafcu, 0xfdfeff00u, selectors);
    const uint32_t sign_selectors =
        0x0c0c0c0cu + ((codes & 0x08080808u) >> 3u);
    const uint32_t sign_mask = __builtin_amdgcn_perm(
        0u, 0u, sign_selectors);
    return (positive & ~sign_mask) | (negative & sign_mask);
#else
    uint32_t result = 0u;
    #pragma unroll
    for (uint32_t i = 0; i < 4u; i++) {
        const uint32_t code = (codes >> (8u * i)) & 0x0fu;
        const uint32_t base = code & 7u;
        int32_t value = (int32_t)(base + (base > 4u ? base - 4u : 0u) +
                                  (base == 7u ? 2u : 0u));
        if ((code & 8u) != 0u) value = -value;
        result |= (uint32_t)(uint8_t)value << (8u * i);
    }
    return result;
#endif
}

__device__ __forceinline__ static void dev_mxfp4_unpack2x4(
        const uint8_t *q,
        int32_t *low,
        int32_t *high) {
    const uint32_t packed = *(const uint32_t *)q;
    *low = (int32_t)dev_mxfp4_unpack4(packed);
    *high = (int32_t)dev_mxfp4_unpack4(packed >> 4u);
}

/* Split one 32-value MXFP4 block across a pair of lanes.  A wave therefore
 * reads 16 consecutive 17-byte blocks instead of having each quarter-wave
 * lane jump by a full 136-byte Q8_K chunk.  This mirrors the coalesced Metal
 * decode layout while retaining the faster Q8_K activation path on gfx1151. */
__device__ __forceinline__ static float dev_dot_mxfp4_q8_K_half_block(
        const cuda_block_mxfp4 *x,
        const cuda_block_q8_K *y,
        uint32_t subblock,
        uint32_t half) {
    const uint32_t weight_offset = half * 8u;
    const uint32_t activation_offset = subblock * 32u + weight_offset;
    const int8_t *q8_lo = y->qs + activation_offset;
    const int8_t *q8_hi = y->qs + activation_offset + 16u;
    const uint8_t *q = x->qs + weight_offset;
    int32_t bsum = 0;
    #pragma unroll
    for (uint32_t j = 0; j < 8u; j += 4u) {
        int32_t wlo, whi;
        dev_mxfp4_unpack2x4(q + j, &wlo, &whi);
        bsum = __dp4a(wlo, *(const int32_t *)(q8_lo + j), bsum);
        bsum = __dp4a(whi, *(const int32_t *)(q8_hi + j), bsum);
    }
    return 0.5f * y->d * dev_e8m0_to_f32(x->e) * (float)bsum;
}

/* Aligned-activation variant of the 8-pair chunk dot.  The staged copies
 * hold each pair's 256 quant bytes 16-byte aligned with the Q8_K scale in
 * a separate array, so one b128 load replaces four dword loads per half
 * sub-block: this kernel family is issue-bound, not bandwidth-bound, and
 * the load count per dp4a is what gates it.  The low/high nibble sums use
 * separate integer accumulators; integer reassociation is exact, so the
 * result is bit-identical to the struct-layout helper. */
__device__ static void dev_dot_mxfp4_q8a_block8(
        const cuda_block_mxfp4 *x8,
        const int8_t *q0, const int8_t *q1, const int8_t *q2, const int8_t *q3,
        const int8_t *q4, const int8_t *q5, const int8_t *q6, const int8_t *q7,
        float d0, float d1, float d2, float d3,
        float d4, float d5, float d6, float d7,
        uint32_t n,
        float acc[8]) {
    const int8_t *qs[8] = { q0, q1, q2, q3, q4, q5, q6, q7 };
    const float ds[8] = { d0, d1, d2, d3, d4, d5, d6, d7 };
    float chunk[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    #pragma unroll
    for (uint32_t sb = 0; sb < 8u; sb++) {
        const cuda_block_mxfp4 *x = x8 + sb;
        /* One misaligned b128 covers the block's 16 packed bytes; gfx11
         * global loads support arbitrary alignment. */
        const uint4 w4 = *(const uint4 *)(x->qs);
        int32_t wlo[4], whi[4];
        wlo[0] = (int32_t)dev_mxfp4_unpack4(w4.x);
        whi[0] = (int32_t)dev_mxfp4_unpack4(w4.x >> 4u);
        wlo[1] = (int32_t)dev_mxfp4_unpack4(w4.y);
        whi[1] = (int32_t)dev_mxfp4_unpack4(w4.y >> 4u);
        wlo[2] = (int32_t)dev_mxfp4_unpack4(w4.z);
        whi[2] = (int32_t)dev_mxfp4_unpack4(w4.z >> 4u);
        wlo[3] = (int32_t)dev_mxfp4_unpack4(w4.w);
        whi[3] = (int32_t)dev_mxfp4_unpack4(w4.w >> 4u);
        const float d = dev_e8m0_to_f32(x->e);
        #pragma unroll
        for (uint32_t p = 0; p < 8u; p++) {
            if (p < n) {
                const int4 lo = *(const int4 *)(qs[p] + sb * 32u);
                const int4 hi = *(const int4 *)(qs[p] + sb * 32u + 16u);
                int32_t s = 0;
                int32_t t = 0;
                s = __dp4a(wlo[0], lo.x, s);
                t = __dp4a(whi[0], hi.x, t);
                s = __dp4a(wlo[1], lo.y, s);
                t = __dp4a(whi[1], hi.y, t);
                s = __dp4a(wlo[2], lo.z, s);
                t = __dp4a(whi[2], hi.z, t);
                s = __dp4a(wlo[3], lo.w, s);
                t = __dp4a(whi[3], hi.w, t);
                chunk[p] += d * (float)(s + t);
            }
        }
    }
    #pragma unroll
    for (uint32_t p = 0; p < 8u; p++) {
        if (p < n) acc[p] += 0.5f * ds[p] * chunk[p];
    }
}

/* Fused gate+up flavor: one activation load pair feeds both weight
 * matrices, halving the dominant b128 traffic of the gate/up tile
 * kernel.  Per-matrix accumulation is unchanged. */
__device__ static void dev_dot_mxfp4_q8a_pair_block8(
        const cuda_block_mxfp4 *g8,
        const cuda_block_mxfp4 *u8,
        const int8_t *q0, const int8_t *q1, const int8_t *q2, const int8_t *q3,
        const int8_t *q4, const int8_t *q5, const int8_t *q6, const int8_t *q7,
        float d0, float d1, float d2, float d3,
        float d4, float d5, float d6, float d7,
        uint32_t n,
        float gate[8],
        float up[8]) {
    const int8_t *qs[8] = { q0, q1, q2, q3, q4, q5, q6, q7 };
    const float ds[8] = { d0, d1, d2, d3, d4, d5, d6, d7 };
    float gchunk[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    float uchunk[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    #pragma unroll
    for (uint32_t sb = 0; sb < 8u; sb++) {
        const uint4 gw = *(const uint4 *)(g8[sb].qs);
        const uint4 uw = *(const uint4 *)(u8[sb].qs);
        int32_t glo[4], ghi[4], ulo[4], uhi[4];
        glo[0] = (int32_t)dev_mxfp4_unpack4(gw.x);
        ghi[0] = (int32_t)dev_mxfp4_unpack4(gw.x >> 4u);
        glo[1] = (int32_t)dev_mxfp4_unpack4(gw.y);
        ghi[1] = (int32_t)dev_mxfp4_unpack4(gw.y >> 4u);
        glo[2] = (int32_t)dev_mxfp4_unpack4(gw.z);
        ghi[2] = (int32_t)dev_mxfp4_unpack4(gw.z >> 4u);
        glo[3] = (int32_t)dev_mxfp4_unpack4(gw.w);
        ghi[3] = (int32_t)dev_mxfp4_unpack4(gw.w >> 4u);
        ulo[0] = (int32_t)dev_mxfp4_unpack4(uw.x);
        uhi[0] = (int32_t)dev_mxfp4_unpack4(uw.x >> 4u);
        ulo[1] = (int32_t)dev_mxfp4_unpack4(uw.y);
        uhi[1] = (int32_t)dev_mxfp4_unpack4(uw.y >> 4u);
        ulo[2] = (int32_t)dev_mxfp4_unpack4(uw.z);
        uhi[2] = (int32_t)dev_mxfp4_unpack4(uw.z >> 4u);
        ulo[3] = (int32_t)dev_mxfp4_unpack4(uw.w);
        uhi[3] = (int32_t)dev_mxfp4_unpack4(uw.w >> 4u);
        const float gd = dev_e8m0_to_f32(g8[sb].e);
        const float ud = dev_e8m0_to_f32(u8[sb].e);
        #pragma unroll
        for (uint32_t p = 0; p < 8u; p++) {
            if (p < n) {
                const int4 lo = *(const int4 *)(qs[p] + sb * 32u);
                const int4 hi = *(const int4 *)(qs[p] + sb * 32u + 16u);
                int32_t gs = 0;
                int32_t gt = 0;
                int32_t us = 0;
                int32_t ut = 0;
                gs = __dp4a(glo[0], lo.x, gs);
                gt = __dp4a(ghi[0], hi.x, gt);
                us = __dp4a(ulo[0], lo.x, us);
                ut = __dp4a(uhi[0], hi.x, ut);
                gs = __dp4a(glo[1], lo.y, gs);
                gt = __dp4a(ghi[1], hi.y, gt);
                us = __dp4a(ulo[1], lo.y, us);
                ut = __dp4a(uhi[1], hi.y, ut);
                gs = __dp4a(glo[2], lo.z, gs);
                gt = __dp4a(ghi[2], hi.z, gt);
                us = __dp4a(ulo[2], lo.z, us);
                ut = __dp4a(uhi[2], hi.z, ut);
                gs = __dp4a(glo[3], lo.w, gs);
                gt = __dp4a(ghi[3], hi.w, gt);
                us = __dp4a(ulo[3], lo.w, us);
                ut = __dp4a(uhi[3], hi.w, ut);
                gchunk[p] += gd * (float)(gs + gt);
                uchunk[p] += ud * (float)(us + ut);
            }
        }
    }
    #pragma unroll
    for (uint32_t p = 0; p < 8u; p++) {
        if (p < n) {
            gate[p] += 0.5f * ds[p] * gchunk[p];
            up[p] += 0.5f * ds[p] * uchunk[p];
        }
    }
}

__device__ static void dev_dot_mxfp4_q8_K_block8(
        const cuda_block_mxfp4 *x8,
        const cuda_block_q8_K *y0,
        const cuda_block_q8_K *y1,
        const cuda_block_q8_K *y2,
        const cuda_block_q8_K *y3,
        const cuda_block_q8_K *y4,
        const cuda_block_q8_K *y5,
        const cuda_block_q8_K *y6,
        const cuda_block_q8_K *y7,
        uint32_t n,
        float acc[8]) {
    const cuda_block_q8_K *ys[8] = { y0, y1, y2, y3, y4, y5, y6, y7 };
    float chunk[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    #pragma unroll
    for (uint32_t sb = 0; sb < 8u; sb++) {
        const cuda_block_mxfp4 *x = x8 + sb;
        int32_t bsum[8] = {0, 0, 0, 0, 0, 0, 0, 0};
        #pragma unroll
        for (uint32_t j = 0; j < 16u; j += 4u) {
            int32_t wlo, whi;
            dev_mxfp4_unpack2x4(x->qs + j, &wlo, &whi);
            for (uint32_t p = 0; p < n; p++) {
                const int8_t *q8 = ys[p]->qs + sb * 32u;
                bsum[p] = __dp4a(wlo, *(const int32_t *)(q8 + j), bsum[p]);
                bsum[p] = __dp4a(whi, *(const int32_t *)(q8 + 16u + j), bsum[p]);
            }
        }
        const float d = dev_e8m0_to_f32(x->e);
        for (uint32_t p = 0; p < n; p++) chunk[p] += d * (float)bsum[p];
    }
    for (uint32_t p = 0; p < n; p++) acc[p] += 0.5f * ys[p]->d * chunk[p];
}

__device__ static void dev_dot_q4_K_q8_K_block4(
        const cuda_block_q4_K *x,
        const cuda_block_q8_K *y0,
        const cuda_block_q8_K *y1,
        const cuda_block_q8_K *y2,
        const cuda_block_q8_K *y3,
        uint32_t n,
        float acc[4]) {
    const cuda_block_q8_K *ys[4] = { y0, y1, y2, y3 };
    const float xd = dev_f16_to_f32(x->d);
    const float xmin = dev_f16_to_f32(x->dmin);
    int isum[4] = {0, 0, 0, 0};
    int summs[4] = {0, 0, 0, 0};
    #pragma unroll
    for (uint32_t j = 0; j < 8u; j++) {
        uint8_t sc, m;
        dev_q4_K_get_scale_min(j, x->scales, &sc, &m);
        const uint32_t byte_off = (j >> 1u) * 32u;
        const int shift = (j & 1u) ? 4 : 0;
        for (uint32_t p = 0; p < n; p++) {
            if (!ys[p]) continue;
            summs[p] += (int)m * (int)(ys[p]->bsums[2u * j] + ys[p]->bsums[2u * j + 1u]);
            isum[p] += (int)sc * dev_dot_q4_32(x->qs + byte_off, ys[p]->qs + j * 32u, shift);
        }
    }
    for (uint32_t p = 0; p < n; p++) {
        if (ys[p]) acc[p] += ys[p]->d * xd * (float)isum[p] - ys[p]->d * xmin * (float)summs[p];
    }
}

__device__ static void dev_dot_q4_K_q8_K_block8(
        const cuda_block_q4_K *x,
        const cuda_block_q8_K *y0,
        const cuda_block_q8_K *y1,
        const cuda_block_q8_K *y2,
        const cuda_block_q8_K *y3,
        const cuda_block_q8_K *y4,
        const cuda_block_q8_K *y5,
        const cuda_block_q8_K *y6,
        const cuda_block_q8_K *y7,
        uint32_t n,
        float acc[8]) {
    const cuda_block_q8_K *ys[8] = { y0, y1, y2, y3, y4, y5, y6, y7 };
    const float xd = dev_f16_to_f32(x->d);
    const float xmin = dev_f16_to_f32(x->dmin);
    int isum[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    int summs[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    #pragma unroll
    for (uint32_t j = 0; j < 8u; j++) {
        uint8_t sc, m;
        dev_q4_K_get_scale_min(j, x->scales, &sc, &m);
        const uint32_t byte_off = (j >> 1u) * 32u;
        const int shift = (j & 1u) ? 4 : 0;
        for (uint32_t p = 0; p < n; p++) {
            if (!ys[p]) continue;
            summs[p] += (int)m * (int)(ys[p]->bsums[2u * j] + ys[p]->bsums[2u * j + 1u]);
            isum[p] += (int)sc * dev_dot_q4_32(x->qs + byte_off, ys[p]->qs + j * 32u, shift);
        }
    }
    for (uint32_t p = 0; p < n; p++) {
        if (ys[p]) acc[p] += ys[p]->d * xd * (float)isum[p] - ys[p]->d * xmin * (float)summs[p];
    }
}

__device__ static float dev_dot_q2_K_q8_K_block(const cuda_block_q2_K *x, const cuda_block_q8_K *y) {
    const uint8_t *q2 = x->qs;
    const int8_t *q8 = y->qs;
    const uint8_t *sc = x->scales;
    int summs = 0;
    for (int j = 0; j < 16; j++) summs += y->bsums[j] * (sc[j] >> 4);
    const float dall = y->d * dev_f16_to_f32(x->d);
    const float dmin = y->d * dev_f16_to_f32(x->dmin);
    int isum = 0;
    int is = 0;
    for (int k = 0; k < CUDA_QK_K / 128; k++) {
        int shift = 0;
        for (int j = 0; j < 4; j++) {
            int d = sc[is++] & 0x0f;
            isum += d * dev_dot_q2_16(q2, q8, shift);
            d = sc[is++] & 0x0f;
            isum += d * dev_dot_q2_16(q2 + 16, q8 + 16, shift);
            shift += 2;
            q8 += 32;
        }
        q2 += 32;
    }
    return dall * (float)isum - dmin * (float)summs;
}


__device__ static void dev_dot_q2_K_q8_K_block4(
        const cuda_block_q2_K *x,
        const cuda_block_q8_K *y0,
        const cuda_block_q8_K *y1,
        const cuda_block_q8_K *y2,
        const cuda_block_q8_K *y3,
        uint32_t n,
        float acc[4]) {
    const uint8_t *sc = x->scales;
    const float xd = dev_f16_to_f32(x->d);
    const float xmin = dev_f16_to_f32(x->dmin);
    const cuda_block_q8_K *ys[4] = { y0, y1, y2, y3 };
    int isum[4] = {0, 0, 0, 0};
    int summs[4] = {0, 0, 0, 0};
    for (uint32_t p = 0; p < n; p++) {
        for (int j = 0; j < 16; j++) summs[p] += ys[p]->bsums[j] * (sc[j] >> 4);
    }
    for (uint32_t p = 0; p < n; p++) {
        const uint8_t *q2 = x->qs;
        const int8_t *q8 = ys[p]->qs;
        int is = 0;
        for (int k = 0; k < CUDA_QK_K / 128; k++) {
            int shift = 0;
            for (int j = 0; j < 4; j++) {
                int d = sc[is++] & 0x0f;
                isum[p] += d * dev_dot_q2_16(q2, q8, shift);
                d = sc[is++] & 0x0f;
                isum[p] += d * dev_dot_q2_16(q2 + 16, q8 + 16, shift);
                shift += 2;
                q8 += 32;
            }
            q2 += 32;
        }
    }
    for (uint32_t p = 0; p < n; p++) {
        const float yd = ys[p]->d;
        acc[p] += yd * xd * (float)isum[p] - yd * xmin * (float)summs[p];
    }
}

__device__ static void dev_dot_q2_K_q8_K_block8(
        const cuda_block_q2_K *x,
        const cuda_block_q8_K *y0,
        const cuda_block_q8_K *y1,
        const cuda_block_q8_K *y2,
        const cuda_block_q8_K *y3,
        const cuda_block_q8_K *y4,
        const cuda_block_q8_K *y5,
        const cuda_block_q8_K *y6,
        const cuda_block_q8_K *y7,
        uint32_t n,
        float acc[8]) {
    const uint8_t *sc = x->scales;
    const float xd = dev_f16_to_f32(x->d);
    const float xmin = dev_f16_to_f32(x->dmin);
    const cuda_block_q8_K *ys[8] = { y0, y1, y2, y3, y4, y5, y6, y7 };
    int isum[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    int summs[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    for (uint32_t p = 0; p < n; p++) {
        for (int j = 0; j < 16; j++) summs[p] += ys[p]->bsums[j] * (sc[j] >> 4);
    }
    for (uint32_t p = 0; p < n; p++) {
        const uint8_t *q2 = x->qs;
        const int8_t *q8 = ys[p]->qs;
        int is = 0;
        for (int k = 0; k < CUDA_QK_K / 128; k++) {
            int shift = 0;
            for (int j = 0; j < 4; j++) {
                int d = sc[is++] & 0x0f;
                isum[p] += d * dev_dot_q2_16(q2, q8, shift);
                d = sc[is++] & 0x0f;
                isum[p] += d * dev_dot_q2_16(q2 + 16, q8 + 16, shift);
                shift += 2;
                q8 += 32;
            }
            q2 += 32;
        }
    }
    for (uint32_t p = 0; p < n; p++) {
        const float yd = ys[p]->d;
        acc[p] += yd * xd * (float)isum[p] - yd * xmin * (float)summs[p];
    }
}

__device__ static float half_warp_sum_f32(float v, uint32_t lane16) {
    uint32_t mask = 0xffffu << (threadIdx.x & 16u);
    for (int offset = 8; offset > 0; offset >>= 1) {
        v += __shfl_down_sync(static_cast<MASK_T>(mask), v, offset, 16);
    }
    (void)lane16;
    return v;
}

__device__ static float quarter_warp_sum_f32(float v, uint32_t lane8) {
    uint32_t mask = 0xffu << (threadIdx.x & 24u);
    for (int offset = 4; offset > 0; offset >>= 1) {
        v += __shfl_down_sync(static_cast<MASK_T>(mask), v, offset, 8);
    }
    (void)lane8;
    return v;
}

__global__ static void q8_K_quantize_kernel(cuda_block_q8_K *out, const float *x, uint32_t in_dim, uint32_t n_rows) {
    uint32_t b = blockIdx.x;
    uint32_t row = blockIdx.y;
    if (row >= n_rows || b >= in_dim / CUDA_QK_K) return;
    const float *xr = x + (uint64_t)row * in_dim + (uint64_t)b * CUDA_QK_K;
    cuda_block_q8_K *yb = out + (uint64_t)row * (in_dim / CUDA_QK_K) + b;
    __shared__ float abs_part[256];
    __shared__ float val_part[256];
    __shared__ float maxv_s;
    __shared__ float iscale_s;
    uint32_t tid = threadIdx.x;
    float v = tid < CUDA_QK_K ? xr[tid] : 0.0f;
    abs_part[tid] = tid < CUDA_QK_K ? fabsf(v) : 0.0f;
    val_part[tid] = v;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride && abs_part[tid + stride] > abs_part[tid]) {
            abs_part[tid] = abs_part[tid + stride];
            val_part[tid] = val_part[tid + stride];
        }
        __syncthreads();
    }
    float amax = abs_part[0];
    if (amax == 0.0f) {
        if (tid == 0) yb->d = 0.0f;
        if (tid < CUDA_QK_K) yb->qs[tid] = 0;
        if (tid < CUDA_QK_K / 16) yb->bsums[tid] = 0;
        return;
    }
    if (tid == 0) {
        maxv_s = val_part[0];
        iscale_s = -127.0f / maxv_s;
    }
    __syncthreads();
    if (tid < CUDA_QK_K) {
        int qv = (int)lrintf(iscale_s * xr[tid]);
        if (qv > 127) qv = 127;
        if (qv < -128) qv = -128;
        yb->qs[tid] = (int8_t)qv;
    }
    __syncthreads();
    if (tid < CUDA_QK_K / 16) {
        int sum = 0;
        for (int i = 0; i < 16; i++) sum += yb->qs[tid * 16 + i];
        yb->bsums[tid] = (int16_t)sum;
    }
    if (tid == 0) yb->d = 1.0f / iscale_s;
}

__global__ static DS4_ROCM_UNUSED void moe_gate_up_mid_kernel(
        float *gate_out,
        float *up_out,
        float *mid_out,
        const char *gate_base,
        const char *up_base,
        const cuda_block_q8_K *xq,
        const int32_t *selected,
        const float *weights,
        uint64_t gate_expert_bytes,
        uint64_t gate_row_bytes,
        uint32_t xq_blocks,
        uint32_t expert_mid_dim,
        uint32_t n_expert,
        float clamp) {
    uint32_t row = blockIdx.x;
    uint32_t pair = blockIdx.y;
    if (row >= expert_mid_dim) return;
    uint32_t tok = pair / n_expert;
    uint32_t slot = pair - tok * n_expert;
    int32_t expert_i = selected[(uint64_t)tok * n_expert + slot];
    if (expert_i < 0) expert_i = 0;
    uint32_t expert = (uint32_t)expert_i;
    const cuda_block_iq2_xxs *gr = (const cuda_block_iq2_xxs *)(gate_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes);
    const cuda_block_iq2_xxs *ur = (const cuda_block_iq2_xxs *)(up_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes);
    const cuda_block_q8_K *xqb = xq + (uint64_t)tok * xq_blocks;
    float gate = 0.0f;
    float up = 0.0f;
    for (uint32_t b = threadIdx.x; b < xq_blocks; b += blockDim.x) {
        gate += dev_dot_iq2_xxs_q8_K_block(gr + b, xqb + b);
        up += dev_dot_iq2_xxs_q8_K_block(ur + b, xqb + b);
    }
    __shared__ float partial_gate[256];
    __shared__ float partial_up[256];
    partial_gate[threadIdx.x] = gate;
    partial_up[threadIdx.x] = up;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            partial_gate[threadIdx.x] += partial_gate[threadIdx.x + stride];
            partial_up[threadIdx.x] += partial_up[threadIdx.x + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        gate = partial_gate[0];
        up = partial_up[0];
        if (clamp > 1.0e-6f) {
            if (gate > clamp) gate = clamp;
            if (up > clamp) up = clamp;
            if (up < -clamp) up = -clamp;
        }
        const uint64_t off = (uint64_t)pair * expert_mid_dim + row;
        gate_out[off] = gate;
        up_out[off] = up;
        mid_out[off] = (gate / (1.0f + expf(-gate))) * up * weights[(uint64_t)tok * n_expert + slot];
    }
}

__global__ static DS4_ROCM_UNUSED void moe_gate_up_mid_warp8_kernel(
        float *gate_out,
        float *up_out,
        float *mid_out,
        const char *gate_base,
        const char *up_base,
        const cuda_block_q8_K *xq,
        const int32_t *selected,
        const float *weights,
        uint64_t gate_expert_bytes,
        uint64_t gate_row_bytes,
        uint32_t xq_blocks,
        uint32_t expert_mid_dim,
        uint32_t n_expert,
        float clamp) {
    uint32_t lane = threadIdx.x & 31u;
    uint32_t warp = threadIdx.x >> 5u;
    uint32_t row = blockIdx.x * 8u + warp;
    uint32_t pair = blockIdx.y;
    if (row >= expert_mid_dim) return;
    uint32_t tok = pair / n_expert;
    uint32_t slot = pair - tok * n_expert;
    int32_t expert_i = selected[(uint64_t)tok * n_expert + slot];
    if (expert_i < 0) expert_i = 0;
    uint32_t expert = (uint32_t)expert_i;
    const cuda_block_iq2_xxs *gr = (const cuda_block_iq2_xxs *)(gate_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes);
    const cuda_block_iq2_xxs *ur = (const cuda_block_iq2_xxs *)(up_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes);
    const cuda_block_q8_K *xqb = xq + (uint64_t)tok * xq_blocks;
    float gate = 0.0f;
    float up = 0.0f;
    for (uint32_t b = lane; b < xq_blocks; b += 32u) {
        gate += dev_dot_iq2_xxs_q8_K_block(gr + b, xqb + b);
        up += dev_dot_iq2_xxs_q8_K_block(ur + b, xqb + b);
    }
    gate = warp_sum_f32(gate);
    up = warp_sum_f32(up);
    if (lane == 0) {
        if (clamp > 1.0e-6f) {
            if (gate > clamp) gate = clamp;
            if (up > clamp) up = clamp;
            if (up < -clamp) up = -clamp;
        }
        const uint64_t off = (uint64_t)pair * expert_mid_dim + row;
        gate_out[off] = gate;
        up_out[off] = up;
        mid_out[off] = (gate / (1.0f + expf(-gate))) * up * weights[(uint64_t)tok * n_expert + slot];
    }
}

__global__ static DS4_ROCM_UNUSED void moe_gate_up_mid_hwarp16_kernel(
        float *gate_out,
        float *up_out,
        float *mid_out,
        const char *gate_base,
        const char *up_base,
        const cuda_block_q8_K *xq,
        const int32_t *selected,
        const float *weights,
        uint64_t gate_expert_bytes,
        uint64_t gate_row_bytes,
        uint32_t xq_blocks,
        uint32_t expert_mid_dim,
        uint32_t n_expert,
        float clamp) {
    uint32_t lane = threadIdx.x & 15u;
    uint32_t row = blockIdx.x * 16u + (threadIdx.x >> 4u);
    uint32_t pair = blockIdx.y;
    if (row >= expert_mid_dim) return;
    uint32_t tok = pair / n_expert;
    uint32_t slot = pair - tok * n_expert;
    int32_t expert_i = selected[(uint64_t)tok * n_expert + slot];
    if (expert_i < 0) expert_i = 0;
    uint32_t expert = (uint32_t)expert_i;
    const cuda_block_iq2_xxs *gr = (const cuda_block_iq2_xxs *)(gate_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes);
    const cuda_block_iq2_xxs *ur = (const cuda_block_iq2_xxs *)(up_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes);
    const cuda_block_q8_K *xqb = xq + (uint64_t)tok * xq_blocks;
    float gate = 0.0f;
    float up = 0.0f;
    for (uint32_t b = lane; b < xq_blocks; b += 16u) {
        gate += dev_dot_iq2_xxs_q8_K_block(gr + b, xqb + b);
        up += dev_dot_iq2_xxs_q8_K_block(ur + b, xqb + b);
    }
    gate = half_warp_sum_f32(gate, lane);
    up = half_warp_sum_f32(up, lane);
    if (lane == 0) {
        if (clamp > 1.0e-6f) {
            if (gate > clamp) gate = clamp;
            if (up > clamp) up = clamp;
            if (up < -clamp) up = -clamp;
        }
        const uint64_t off = (uint64_t)pair * expert_mid_dim + row;
        gate_out[off] = gate;
        up_out[off] = up;
        mid_out[off] = (gate / (1.0f + expf(-gate))) * up * weights[(uint64_t)tok * n_expert + slot];
    }
}

__global__ static void moe_gate_up_mid_qwarp32_kernel(
        float *gate_out,
        float *up_out,
        float *mid_out,
        const char *gate_base,
        const char *up_base,
        const cuda_block_q8_K *xq,
        const int32_t *selected,
        const float *weights,
        uint64_t gate_expert_bytes,
        uint64_t gate_row_bytes,
        uint32_t xq_blocks,
        uint32_t expert_mid_dim,
        uint32_t n_expert,
        uint32_t active_mask,
        float clamp) {
    uint32_t lane = threadIdx.x & 7u;
    uint32_t row_lane = threadIdx.x >> 3u;
    uint32_t pair = blockIdx.y;
    uint32_t tok = pair / n_expert;
    uint32_t slot = pair - tok * n_expert;
    if ((active_mask & (1u << slot)) == 0) return;
    int32_t expert_i = selected[(uint64_t)tok * n_expert + slot];
    if (expert_i < 0) expert_i = 0;
    uint32_t expert = (uint32_t)expert_i;
    const cuda_block_q8_K *xqb = xq + (uint64_t)tok * xq_blocks;
    for (uint32_t rr = 0; rr < 4u; rr++) {
        uint32_t row = blockIdx.x * 128u + row_lane + rr * 32u;
        if (row >= expert_mid_dim) continue;
        const cuda_block_iq2_xxs *gr = (const cuda_block_iq2_xxs *)(gate_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes);
        const cuda_block_iq2_xxs *ur = (const cuda_block_iq2_xxs *)(up_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes);
        float gate = 0.0f;
        float up = 0.0f;
        for (uint32_t b = lane; b < xq_blocks; b += 8u) {
            gate += dev_dot_iq2_xxs_q8_K_block(gr + b, xqb + b);
            up += dev_dot_iq2_xxs_q8_K_block(ur + b, xqb + b);
        }
        gate = quarter_warp_sum_f32(gate, lane);
        up = quarter_warp_sum_f32(up, lane);
        if (lane == 0) {
            if (clamp > 1.0e-6f) {
                if (gate > clamp) gate = clamp;
                if (up > clamp) up = clamp;
                if (up < -clamp) up = -clamp;
            }
            const uint64_t off = (uint64_t)pair * expert_mid_dim + row;
            gate_out[off] = gate;
            up_out[off] = up;
            mid_out[off] = (gate / (1.0f + expf(-gate))) * up * weights[(uint64_t)tok * n_expert + slot];
        }
    }
}

__global__ static void moe_gate_up_mid_qwarp32_ptrs_kernel(
        float *gate_out,
        float *up_out,
        float *mid_out,
        const char * const *gate_slots,
        const char * const *up_slots,
        const cuda_block_q8_K *xq,
        const int32_t *selected,
        const float *weights,
        uint64_t gate_row_bytes,
        uint32_t xq_blocks,
        uint32_t expert_mid_dim,
        uint32_t n_expert,
        uint32_t active_mask,
        float clamp) {
    uint32_t lane = threadIdx.x & 7u;
    uint32_t row_lane = threadIdx.x >> 3u;
    uint32_t pair = blockIdx.y;
    uint32_t tok = pair / n_expert;
    uint32_t slot = pair - tok * n_expert;
    if ((active_mask & (1u << slot)) == 0) return;
    int32_t slot_i = selected[(uint64_t)tok * n_expert + slot];
    if (slot_i < 0) slot_i = 0;
    const uint32_t compact_slot = (uint32_t)slot_i;
    const char *gate_base = gate_slots[compact_slot];
    const char *up_base = up_slots[compact_slot];
    if (!gate_base || !up_base) return;
    const cuda_block_q8_K *xqb = xq + (uint64_t)tok * xq_blocks;
    for (uint32_t rr = 0; rr < 4u; rr++) {
        uint32_t row = blockIdx.x * 128u + row_lane + rr * 32u;
        if (row >= expert_mid_dim) continue;
        const cuda_block_iq2_xxs *gr = (const cuda_block_iq2_xxs *)(gate_base + (uint64_t)row * gate_row_bytes);
        const cuda_block_iq2_xxs *ur = (const cuda_block_iq2_xxs *)(up_base + (uint64_t)row * gate_row_bytes);
        float gate = 0.0f;
        float up = 0.0f;
        for (uint32_t b = lane; b < xq_blocks; b += 8u) {
            gate += dev_dot_iq2_xxs_q8_K_block(gr + b, xqb + b);
            up += dev_dot_iq2_xxs_q8_K_block(ur + b, xqb + b);
        }
        gate = quarter_warp_sum_f32(gate, lane);
        up = quarter_warp_sum_f32(up, lane);
        if (lane == 0) {
            if (clamp > 1.0e-6f) {
                if (gate > clamp) gate = clamp;
                if (up > clamp) up = clamp;
                if (up < -clamp) up = -clamp;
            }
            const uint64_t off = (uint64_t)pair * expert_mid_dim + row;
            gate_out[off] = gate;
            up_out[off] = up;
            mid_out[off] = (gate / (1.0f + expf(-gate))) * up * weights[(uint64_t)tok * n_expert + slot];
        }
    }
}

__global__ static void moe_gate_up_mid_qwarp32_ptrs_split_kernel(
        float *gate_out,
        float *up_out,
        float *mid_out,
        const char * const *gate_slots,
        const char * const *up_slots,
        const uint8_t *pair_missing,
        uint32_t want_missing,
        const cuda_block_q8_K *xq,
        const int32_t *selected,
        const float *weights,
        uint64_t gate_row_bytes,
        uint32_t xq_blocks,
        uint32_t expert_mid_dim,
        uint32_t n_expert,
        uint32_t active_mask,
        float clamp) {
    uint32_t lane = threadIdx.x & 7u;
    uint32_t row_lane = threadIdx.x >> 3u;
    uint32_t pair = blockIdx.y;
    uint32_t tok = pair / n_expert;
    uint32_t slot = pair - tok * n_expert;
    if ((active_mask & (1u << slot)) == 0) return;
    if (!pair_missing || ((uint32_t)(pair_missing[pair] != 0) != want_missing)) return;
    int32_t slot_i = selected[(uint64_t)tok * n_expert + slot];
    if (slot_i < 0) slot_i = 0;
    const uint32_t compact_slot = (uint32_t)slot_i;
    const char *gate_base = gate_slots[compact_slot];
    const char *up_base = up_slots[compact_slot];
    if (!gate_base || !up_base) return;
    const cuda_block_q8_K *xqb = xq + (uint64_t)tok * xq_blocks;
    for (uint32_t rr = 0; rr < 4u; rr++) {
        uint32_t row = blockIdx.x * 128u + row_lane + rr * 32u;
        if (row >= expert_mid_dim) continue;
        const cuda_block_iq2_xxs *gr = (const cuda_block_iq2_xxs *)(gate_base + (uint64_t)row * gate_row_bytes);
        const cuda_block_iq2_xxs *ur = (const cuda_block_iq2_xxs *)(up_base + (uint64_t)row * gate_row_bytes);
        float gate = 0.0f;
        float up = 0.0f;
        for (uint32_t b = lane; b < xq_blocks; b += 8u) {
            gate += dev_dot_iq2_xxs_q8_K_block(gr + b, xqb + b);
            up += dev_dot_iq2_xxs_q8_K_block(ur + b, xqb + b);
        }
        gate = quarter_warp_sum_f32(gate, lane);
        up = quarter_warp_sum_f32(up, lane);
        if (lane == 0) {
            if (clamp > 1.0e-6f) {
                if (gate > clamp) gate = clamp;
                if (up > clamp) up = clamp;
                if (up < -clamp) up = -clamp;
            }
            const uint64_t off = (uint64_t)pair * expert_mid_dim + row;
            gate_out[off] = gate;
            up_out[off] = up;
            mid_out[off] = (gate / (1.0f + expf(-gate))) * up * weights[(uint64_t)tok * n_expert + slot];
        }
    }
}

/* Eight signed IQ2 weights dotted with eight Q8_K activations.  The grid holds
 * nonzero magnitudes, so negating a byte is (w ^ 0xff) + 1 with no carry into
 * its neighbour: one xor and one add sign a whole dword. */
__device__ __forceinline__ static int32_t dev_iq2_signed_dp4a_8(
        uint64_t grid,
        uint64_t sign_mask,
        int32_t x0,
        int32_t x1,
        int32_t acc) {
    const uint64_t w = grid ^ sign_mask;
    acc = __dp4a((int32_t)((uint32_t)w + ((uint32_t)sign_mask & 0x01010101u)), x0, acc);
    acc = __dp4a((int32_t)((uint32_t)(w >> 32u) + ((uint32_t)(sign_mask >> 32u) & 0x01010101u)), x1, acc);
    return acc;
}

/* What a block of the decode gate/up kernels keeps in LDS: the token's Q8_K
 * activations, the IQ2 grid, and a byte mask per sign code (0xff where the
 * weight is negative). */
__device__ __forceinline__ static void moe_decode_iq2_stage(
        cuda_block_q8_K *sxq,
        uint64_t *grid,
        uint64_t *sign_mask,
        const cuda_block_q8_K *xq,
        uint32_t xq_blocks) {
    for (uint32_t i = threadIdx.x; i < xq_blocks; i += blockDim.x) sxq[i] = xq[i];
    for (uint32_t i = threadIdx.x; i < 256u; i += blockDim.x) grid[i] = cuda_iq2xxs_grid[i];
    for (uint32_t i = threadIdx.x; i < 128u; i += blockDim.x) {
        const uint32_t s = dev_unpack_iq2_signs(cuda_ksigns_iq2xs[i]);
        sign_mask[i] = (uint64_t)(uint32_t)__vcmpne4(s & 0x08040201u, 0) |
                       ((uint64_t)(uint32_t)__vcmpne4(s & 0x80402010u, 0) << 32u);
    }
    __syncthreads();
}

/* silu(gate) * up * weight in the operation order the fast-math build gives
 * the row kernels, pinned so this kernel stays bit-identical to them. */
__device__ __forceinline__ static float moe_decode_swiglu_weighted(float gate, float up, float weight) {
#pragma clang fp reassociate(off)
    return ((up * gate) * weight) / (1.0f + expf(-gate));
}

/* Single-token routed gate/up over IQ2_XXS experts and Q8_K activations.
 *
 * Decode of these experts is bound by arithmetic, not by reading the weights,
 * so the layout is chosen to spend few instructions per weight.  A thread owns
 * one 256-weight block of one gate row and of the matching up row: it fetches
 * both blocks up front (eight 8-byte loads each), reads every activation dword
 * once for the two rows, and signs the grid bytes from a table.  A wave
 * holds two rows of sixteen blocks, so each wave streams 4 KiB of weights.
 *
 * The block sums are exact integers.  The float tail keeps the summation order
 * of the row kernels this replaces (blocks b and b+8, then a tree over eight
 * partials), so the result is bit-identical to them. */
__device__ __forceinline__ static void moe_gate_up_mid_decode_iq2_rows(
        float *gate_out,
        float *up_out,
        float *mid_out,
        const unsigned char *gate_expert,
        const unsigned char *up_expert,
        const cuda_block_q8_K *xq,
        const uint64_t *grid,
        const uint64_t *sign_mask,
        float weight,
        uint64_t gate_row_bytes,
        uint32_t xq_blocks,
        uint32_t expert_mid_dim,
        uint64_t out_base,
        uint32_t write_aux,
        float clamp) {
    const uint32_t b = threadIdx.x & 15u;
    const uint32_t row = blockIdx.x * (blockDim.x >> 4u) + (threadIdx.x >> 4u);
    if (row >= expert_mid_dim) return;
    /* Block scale and integer block sum of this thread's gate and up block. */
    float gc = 0.0f, gs = 0.0f;
    float uc = 0.0f, us = 0.0f;
    if (b < xq_blocks) {
        const unsigned char *gb = gate_expert + (uint64_t)row * gate_row_bytes + b * sizeof(cuda_block_iq2_xxs);
        const unsigned char *ub = up_expert + (uint64_t)row * gate_row_bytes + b * sizeof(cuda_block_iq2_xxs);
        uint64_t gq[8], uq[8];
        uint16_t gd, ud;
        __builtin_memcpy(&gd, gb, sizeof(gd));
        __builtin_memcpy(&ud, ub, sizeof(ud));
#pragma unroll
        for (uint32_t k = 0; k < 8u; k++) {
            __builtin_memcpy(&gq[k], gb + 2u + 8u * k, sizeof(gq[k]));
            __builtin_memcpy(&uq[k], ub + 2u + 8u * k, sizeof(uq[k]));
        }
        const int32_t *x = reinterpret_cast<const int32_t *>(xq[b].qs);
        int32_t gsum = 0;
        int32_t usum = 0;
#pragma unroll
        for (uint32_t k = 0; k < 8u; k++) {
            const uint32_t g0 = (uint32_t)gq[k], g1 = (uint32_t)(gq[k] >> 32u);
            const uint32_t u0 = (uint32_t)uq[k], u1 = (uint32_t)(uq[k] >> 32u);
            int32_t gi = 0;
            int32_t ui = 0;
#pragma unroll
            for (uint32_t j = 0; j < 4u; j++) {
                const int32_t x0 = x[8u * k + 2u * j];
                const int32_t x1 = x[8u * k + 2u * j + 1u];
                gi = dev_iq2_signed_dp4a_8(grid[(g0 >> (8u * j)) & 0xffu],
                                           sign_mask[(g1 >> (7u * j)) & 127u], x0, x1, gi);
                ui = dev_iq2_signed_dp4a_8(grid[(u0 >> (8u * j)) & 0xffu],
                                           sign_mask[(u1 >> (7u * j)) & 127u], x0, x1, ui);
            }
            gsum += gi * (int32_t)(2u * (g1 >> 28u) + 1u);
            usum += ui * (int32_t)(2u * (u1 >> 28u) + 1u);
        }
        const float xd = 0.125f * xq[b].d;
        gc = dev_f16_to_f32(gd) * xd;
        uc = dev_f16_to_f32(ud) * xd;
        gs = (float)gsum;
        us = (float)usum;
    }
    /* Lane b adds block b+8 with a fused multiply-add, as the row kernels'
     * per-lane loop does, then the eight partials reduce pairwise. */
    float gate = __builtin_fmaf(__shfl_down_sync(FULL_WARP_MASK, gc, 8, 16),
                                __shfl_down_sync(FULL_WARP_MASK, gs, 8, 16), gc * gs);
    float up = __builtin_fmaf(__shfl_down_sync(FULL_WARP_MASK, uc, 8, 16),
                              __shfl_down_sync(FULL_WARP_MASK, us, 8, 16), uc * us);
    for (uint32_t offset = 4u; offset > 0u; offset >>= 1u) {
        gate += __shfl_down_sync(FULL_WARP_MASK, gate, offset, 16);
        up += __shfl_down_sync(FULL_WARP_MASK, up, offset, 16);
    }
    if (b == 0u) {
        if (clamp > 1.0e-6f) {
            if (gate > clamp) gate = clamp;
            if (up > clamp) up = clamp;
            if (up < -clamp) up = -clamp;
        }
        const uint64_t off = out_base + row;
        if (write_aux) {
            gate_out[off] = gate;
            up_out[off] = up;
        }
        mid_out[off] = moe_decode_swiglu_weighted(gate, up, weight);
    }
}

/* Rows covered by one block of the decode gate/up kernels below. */
#define DS4_ROCM_MOE_DECODE_IQ2_BLOCK_ROWS 16u
#define DS4_ROCM_MOE_DECODE_IQ2_BLOCK_THREADS (16u * DS4_ROCM_MOE_DECODE_IQ2_BLOCK_ROWS)

__global__ static void moe_gate_up_mid_decode_iq2_kernel(
        float *gate_out,
        float *up_out,
        float *mid_out,
        const char *gate_base,
        const char *up_base,
        const cuda_block_q8_K *xq,
        const int32_t *selected,
        const float *weights,
        uint64_t gate_expert_bytes,
        uint64_t gate_row_bytes,
        uint32_t xq_blocks,
        uint32_t expert_mid_dim,
        uint32_t write_aux,
        float clamp) {
    const uint32_t slot = blockIdx.y;
    int32_t expert_i = selected[slot];
    if (expert_i < 0) expert_i = 0;
    __shared__ cuda_block_q8_K sxq[16];
    __shared__ uint64_t s_iq2_grid[256];
    __shared__ uint64_t s_iq2_sign_mask[128];
    moe_decode_iq2_stage(sxq, s_iq2_grid, s_iq2_sign_mask, xq, xq_blocks);
    moe_gate_up_mid_decode_iq2_rows(
            gate_out, up_out, mid_out,
            (const unsigned char *)gate_base + (uint64_t)(uint32_t)expert_i * gate_expert_bytes,
            (const unsigned char *)up_base + (uint64_t)(uint32_t)expert_i * gate_expert_bytes,
            sxq, s_iq2_grid, s_iq2_sign_mask, weights[slot],
            gate_row_bytes, xq_blocks, expert_mid_dim,
            (uint64_t)slot * expert_mid_dim, write_aux, clamp);
}

__global__ static void moe_gate_up_mid_decode_iq2_ptrs_kernel(
        float *gate_out,
        float *up_out,
        float *mid_out,
        const char * const *gate_slots,
        const char * const *up_slots,
        const cuda_block_q8_K *xq,
        const int32_t *selected,
        const float *weights,
        uint64_t gate_row_bytes,
        uint32_t xq_blocks,
        uint32_t expert_mid_dim,
        uint32_t write_aux,
        uint32_t active_mask,
        float clamp) {
    const uint32_t slot = blockIdx.y;
    if ((active_mask & (1u << slot)) == 0) return;
    int32_t slot_i = selected[slot];
    if (slot_i < 0) slot_i = 0;
    const char *gate_base = gate_slots[(uint32_t)slot_i];
    const char *up_base = up_slots[(uint32_t)slot_i];
    if (!gate_base || !up_base) return;
    __shared__ cuda_block_q8_K sxq[16];
    __shared__ uint64_t s_iq2_grid[256];
    __shared__ uint64_t s_iq2_sign_mask[128];
    moe_decode_iq2_stage(sxq, s_iq2_grid, s_iq2_sign_mask, xq, xq_blocks);
    moe_gate_up_mid_decode_iq2_rows(
            gate_out, up_out, mid_out,
            (const unsigned char *)gate_base, (const unsigned char *)up_base,
            sxq, s_iq2_grid, s_iq2_sign_mask, weights[slot],
            gate_row_bytes, xq_blocks, expert_mid_dim,
            (uint64_t)slot * expert_mid_dim, write_aux, clamp);
}

__global__ static void moe_count_sorted_pairs_kernel(
        uint32_t *counts,
        const int32_t *selected,
        uint32_t pair_count,
        uint32_t n_total_expert) {
    uint32_t pair = (uint32_t)((uint64_t)blockIdx.x * blockDim.x + threadIdx.x);
    if (pair >= pair_count) return;
    int32_t expert_i = selected[pair];
    if (expert_i < 0) expert_i = 0;
    if ((uint32_t)expert_i >= n_total_expert) return;
    atomicAdd(counts + (uint32_t)expert_i, 1u);
}

__global__ static void moe_prefix_sorted_pairs_kernel(
        uint32_t *offsets,
        uint32_t *cursors,
        const uint32_t *counts,
        uint32_t n_total_expert) {
    if (threadIdx.x == 0) {
        uint32_t sum = 0;
        for (uint32_t e = 0; e < n_total_expert; e++) {
            offsets[e] = sum;
            cursors[e] = sum;
            sum += counts[e];
        }
        offsets[n_total_expert] = sum;
    }
}

__global__ static void moe_scatter_sorted_pairs_kernel(
        uint32_t *sorted_pairs,
        uint32_t *cursors,
        const int32_t *selected,
        uint32_t pair_count,
        uint32_t n_total_expert) {
    uint32_t pair = (uint32_t)((uint64_t)blockIdx.x * blockDim.x + threadIdx.x);
    if (pair >= pair_count) return;
    int32_t expert_i = selected[pair];
    if (expert_i < 0) expert_i = 0;
    if ((uint32_t)expert_i >= n_total_expert) return;
    uint32_t pos = atomicAdd(cursors + (uint32_t)expert_i, 1u);
    sorted_pairs[pos] = pair;
}

/* Keep pair order stable inside each expert bucket.  The MoE WMMA kernels are
 * row-position sensitive enough that atomic append order changes logits. */
__global__ static void moe_scatter_sorted_pairs_deterministic_kernel(
        uint32_t *sorted_pairs,
        const uint32_t *offsets,
        const int32_t *selected,
        uint32_t pair_count,
        uint32_t n_total_expert) {
    const uint32_t expert = (uint32_t)blockIdx.x;
    if (expert >= n_total_expert) return;
    __shared__ uint32_t wave_counts[8];
    const uint32_t tid = threadIdx.x;
    const uint32_t wave = tid >> 5u;
    const uint32_t lane = tid & 31u;
    uint32_t pos = offsets[expert];
    for (uint32_t base = 0; base < pair_count; base += blockDim.x) {
        const uint32_t pair = base + tid;
        bool match = false;
        if (pair < pair_count) {
            int32_t expert_i = selected[pair];
            if (expert_i < 0) expert_i = 0;
            match = (uint32_t)expert_i == expert;
        }
        const unsigned long long ballot = __ballot(match);
        if (lane == 0u) wave_counts[wave] = (uint32_t)__popcll(ballot);
        __syncthreads();
        uint32_t wave_base = 0u, total = 0u;
        for (uint32_t w = 0; w < (blockDim.x >> 5u); w++) {
            if (w < wave) wave_base += wave_counts[w];
            total += wave_counts[w];
        }
        if (match) {
            const uint32_t before = (uint32_t)__popcll(ballot & ((1ull << lane) - 1ull));
            sorted_pairs[pos + wave_base + before] = pair;
        }
        pos += total;
        __syncthreads();
    }
}

__global__ static void moe_build_expert_tile_offsets_kernel(
        uint32_t *tile_offsets,
        uint32_t *tile_total,
        const uint32_t *counts,
        uint32_t block_m,
        uint32_t n_total_expert) {
    if (threadIdx.x == 0) {
        uint32_t sum = 0;
        for (uint32_t e = 0; e < n_total_expert; e++) {
            tile_offsets[e] = sum;
            sum += (counts[e] + block_m - 1u) / block_m;
        }
        tile_offsets[n_total_expert] = sum;
        *tile_total = sum;
    }
}

__global__ static void moe_build_expert_tiles_kernel(
        uint32_t *tile_experts,
        uint32_t *tile_starts,
        const uint32_t *tile_offsets,
        const uint32_t *counts,
        uint32_t block_m,
        uint32_t n_total_expert) {
    uint32_t e = (uint32_t)((uint64_t)blockIdx.x * blockDim.x + threadIdx.x);
    if (e >= n_total_expert) return;
    uint32_t ntiles = (counts[e] + block_m - 1u) / block_m;
    uint32_t off = tile_offsets[e];
    for (uint32_t t = 0; t < ntiles; t++) {
        tile_experts[off + t] = e;
        tile_starts[off + t] = t * block_m;
    }
}

__global__ static void moe_build_active_experts_kernel(
        uint32_t *active_count,
        uint32_t *active_experts,
        const uint32_t *counts,
        uint32_t n_total_expert) {
    if (blockIdx.x != 0u || threadIdx.x != 0u) return;
    uint32_t total = 0u;
    for (uint32_t e = 0; e < n_total_expert; e++) {
        if (counts[e] != 0u) active_experts[total++] = e;
    }
    *active_count = total;
}

__global__ static void moe_gate_up_mid_sorted_qwarp32_kernel(
        float *gate_out,
        float *up_out,
        float *mid_out,
        const char *gate_base,
        const char *up_base,
        const cuda_block_q8_K *xq,
        const uint32_t *sorted_pairs,
        const int32_t *selected,
        const float *weights,
        uint64_t gate_expert_bytes,
        uint64_t gate_row_bytes,
        uint32_t xq_blocks,
        uint32_t expert_mid_dim,
        uint32_t n_expert,
        float clamp) {
    uint32_t lane = threadIdx.x & 7u;
    uint32_t row = blockIdx.x * 32u + (threadIdx.x >> 3u);
    uint32_t pair = sorted_pairs[blockIdx.y];
    if (row >= expert_mid_dim) return;
    uint32_t tok = pair / n_expert;
    uint32_t slot = pair - tok * n_expert;
    int32_t expert_i = selected[(uint64_t)tok * n_expert + slot];
    if (expert_i < 0) expert_i = 0;
    uint32_t expert = (uint32_t)expert_i;
    const cuda_block_iq2_xxs *gr = (const cuda_block_iq2_xxs *)(gate_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes);
    const cuda_block_iq2_xxs *ur = (const cuda_block_iq2_xxs *)(up_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes);
    const cuda_block_q8_K *xqb = xq + (uint64_t)tok * xq_blocks;
    float gate = 0.0f;
    float up = 0.0f;
    for (uint32_t b = lane; b < xq_blocks; b += 8u) {
        gate += dev_dot_iq2_xxs_q8_K_block(gr + b, xqb + b);
        up += dev_dot_iq2_xxs_q8_K_block(ur + b, xqb + b);
    }
    gate = quarter_warp_sum_f32(gate, lane);
    up = quarter_warp_sum_f32(up, lane);
    if (lane == 0) {
        if (clamp > 1.0e-6f) {
            if (gate > clamp) gate = clamp;
            if (up > clamp) up = clamp;
            if (up < -clamp) up = -clamp;
        }
        const uint64_t off = (uint64_t)pair * expert_mid_dim + row;
        gate_out[off] = gate;
        up_out[off] = up;
        mid_out[off] = (gate / (1.0f + expf(-gate))) * up * weights[(uint64_t)tok * n_expert + slot];
    }
}

__global__ static DS4_ROCM_UNUSED void moe_gate_up_mid_expert_tile8_kernel(
        float *gate_out,
        float *up_out,
        float *mid_out,
        const char *gate_base,
        const char *up_base,
        const cuda_block_q8_K *xq,
        const uint32_t *sorted_pairs,
        const uint32_t *offsets,
        const uint32_t *counts,
        const uint32_t *tile_total,
        const uint32_t *tile_experts,
        const uint32_t *tile_starts,
        const float *weights,
        uint64_t gate_expert_bytes,
        uint64_t gate_row_bytes,
        uint32_t xq_blocks,
        uint32_t expert_mid_dim,
        uint32_t n_expert,
        float clamp) {
    uint32_t tile = blockIdx.y;
    if (tile >= *tile_total) return;
    uint32_t group = threadIdx.x >> 3u;
    uint32_t lane = threadIdx.x & 7u;
    uint32_t pair_slot = group & 7u;
    uint32_t row_lane = group >> 3u;
    uint32_t expert = tile_experts[tile];
    uint32_t local_pair = tile_starts[tile] + pair_slot;
    if (local_pair >= counts[expert]) return;
    uint32_t sorted_idx = offsets[expert] + local_pair;
    uint32_t pair = sorted_pairs[sorted_idx];
    uint32_t tok = pair / n_expert;
    uint32_t slot = pair - tok * n_expert;
    const cuda_block_q8_K *xqb = xq + (uint64_t)tok * xq_blocks;

    for (uint32_t rr = 0; rr < 2u; rr++) {
        uint32_t row = blockIdx.x * 8u + row_lane + rr * 4u;
        if (row >= expert_mid_dim) continue;
        const cuda_block_iq2_xxs *gr = (const cuda_block_iq2_xxs *)(gate_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes);
        const cuda_block_iq2_xxs *ur = (const cuda_block_iq2_xxs *)(up_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes);
        float gate = 0.0f;
        float up = 0.0f;
        for (uint32_t b = lane; b < xq_blocks; b += 8u) {
            gate += dev_dot_iq2_xxs_q8_K_block(gr + b, xqb + b);
            up += dev_dot_iq2_xxs_q8_K_block(ur + b, xqb + b);
        }
        gate = quarter_warp_sum_f32(gate, lane);
        up = quarter_warp_sum_f32(up, lane);
        if (lane == 0) {
            if (clamp > 1.0e-6f) {
                if (gate > clamp) gate = clamp;
                if (up > clamp) up = clamp;
                if (up < -clamp) up = -clamp;
            }
            const uint64_t off = (uint64_t)pair * expert_mid_dim + row;
            gate_out[off] = gate;
            up_out[off] = up;
            mid_out[off] = (gate / (1.0f + expf(-gate))) * up * weights[(uint64_t)tok * n_expert + slot];
        }
    }
}

__global__ static void moe_gate_up_mid_expert_tile4_row32_kernel(
        float *gate_out,
        float *up_out,
        float *mid_out,
        const char *gate_base,
        const char *up_base,
        const cuda_block_q8_K *xq,
        const uint32_t *sorted_pairs,
        const uint32_t *offsets,
        const uint32_t *counts,
        const uint32_t *tile_total,
        const uint32_t *tile_experts,
        const uint32_t *tile_starts,
        const float *weights,
        uint64_t gate_expert_bytes,
        uint64_t gate_row_bytes,
        uint32_t xq_blocks,
        uint32_t expert_mid_dim,
        uint32_t n_expert,
        uint32_t max_count,
        uint32_t write_aux,
        float clamp) {
    uint32_t tile = blockIdx.y;
    if (tile >= *tile_total) return;
    uint32_t lane = threadIdx.x & 7u;
    uint32_t row = blockIdx.x * 32u + (threadIdx.x >> 3u);
    uint32_t expert = tile_experts[tile];
    uint32_t count = counts[expert];
    if (max_count != 0u && count >= max_count) return;
    uint32_t local_start = tile_starts[tile];
    __shared__ cuda_block_q8_K sxq[4][16];
    uint32_t pair[4] = {0, 0, 0, 0};
    uint32_t tok[4] = {0, 0, 0, 0};
    uint32_t slot[4] = {0, 0, 0, 0};
    const cuda_block_q8_K *xqb[4] = {NULL, NULL, NULL, NULL};
    uint32_t np = 0;
    for (; np < 4u; np++) {
        uint32_t local_pair = local_start + np;
        if (local_pair >= count) break;
        pair[np] = sorted_pairs[offsets[expert] + local_pair];
        tok[np] = pair[np] / n_expert;
        slot[np] = pair[np] - tok[np] * n_expert;
        xqb[np] = xq + (uint64_t)tok[np] * xq_blocks;
    }
    if (xq_blocks <= 16u) {
        for (uint32_t i = threadIdx.x; i < np * xq_blocks; i += blockDim.x) {
            uint32_t p = i / xq_blocks;
            uint32_t b = i - p * xq_blocks;
            sxq[p][b] = xqb[p][b];
        }
        __syncthreads();
        for (uint32_t p = 0; p < np; p++) xqb[p] = sxq[p];
    }
    if (row >= expert_mid_dim) return;
    const cuda_block_iq2_xxs *gr = (const cuda_block_iq2_xxs *)(gate_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes);
    const cuda_block_iq2_xxs *ur = (const cuda_block_iq2_xxs *)(up_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes);
    float gate[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float up[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (uint32_t b = lane; b < xq_blocks; b += 8u) {
        dev_dot_iq2_xxs_q8_K_block4(gr + b, xqb[0] ? xqb[0] + b : NULL, xqb[1] ? xqb[1] + b : NULL,
                                    xqb[2] ? xqb[2] + b : NULL, xqb[3] ? xqb[3] + b : NULL, np, gate);
        dev_dot_iq2_xxs_q8_K_block4(ur + b, xqb[0] ? xqb[0] + b : NULL, xqb[1] ? xqb[1] + b : NULL,
                                    xqb[2] ? xqb[2] + b : NULL, xqb[3] ? xqb[3] + b : NULL, np, up);
    }
    for (uint32_t p = 0; p < np; p++) {
        gate[p] = quarter_warp_sum_f32(gate[p], lane);
        up[p] = quarter_warp_sum_f32(up[p], lane);
        if (lane == 0) {
            if (clamp > 1.0e-6f) {
                if (gate[p] > clamp) gate[p] = clamp;
                if (up[p] > clamp) up[p] = clamp;
                if (up[p] < -clamp) up[p] = -clamp;
            }
            const uint64_t off = (uint64_t)pair[p] * expert_mid_dim + row;
            if (write_aux) {
                gate_out[off] = gate[p];
                up_out[off] = up[p];
            }
            mid_out[off] = (gate[p] / (1.0f + expf(-gate[p]))) * up[p] * weights[(uint64_t)tok[p] * n_expert + slot[p]];
        }
    }
}

__global__ static void moe_gate_up_mid_expert_tile8_row32_kernel(
        float *gate_out,
        float *up_out,
        float *mid_out,
        const char *gate_base,
        const char *up_base,
        const cuda_block_q8_K *xq,
        const uint32_t *sorted_pairs,
        const uint32_t *offsets,
        const uint32_t *counts,
        const uint32_t *tile_total,
        const uint32_t *tile_experts,
        const uint32_t *tile_starts,
        const float *weights,
        uint64_t gate_expert_bytes,
        uint64_t gate_row_bytes,
        uint32_t xq_blocks,
        uint32_t expert_mid_dim,
        uint32_t n_expert,
        uint32_t max_count,
        uint32_t write_aux,
        float clamp) {
    uint32_t tile = blockIdx.y;
    if (tile >= *tile_total) return;
    uint32_t lane = threadIdx.x & 7u;
    uint32_t row = blockIdx.x * 32u + (threadIdx.x >> 3u);
    uint32_t expert = tile_experts[tile];
    uint32_t count = counts[expert];
    if (max_count != 0u && count >= max_count) return;
    uint32_t local_start = tile_starts[tile];
    __shared__ cuda_block_q8_K sxq[8][16];
    __shared__ uint64_t s_iq2_grid[256];
    __shared__ uint8_t s_iq2_signs[128];
    uint32_t pair[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    uint32_t tok[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    uint32_t slot[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    const cuda_block_q8_K *xqb[8] = {NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL};
    uint32_t np = 0;
    for (; np < 8u; np++) {
        uint32_t local_pair = local_start + np;
        if (local_pair >= count) break;
        pair[np] = sorted_pairs[offsets[expert] + local_pair];
        tok[np] = pair[np] / n_expert;
        slot[np] = pair[np] - tok[np] * n_expert;
        xqb[np] = xq + (uint64_t)tok[np] * xq_blocks;
    }
    if (xq_blocks <= 16u) {
        for (uint32_t i = threadIdx.x; i < np * xq_blocks; i += blockDim.x) {
            uint32_t p = i / xq_blocks;
            uint32_t b = i - p * xq_blocks;
            sxq[p][b] = xqb[p][b];
        }
    }
    for (uint32_t i = threadIdx.x; i < 256u; i += blockDim.x) s_iq2_grid[i] = cuda_iq2xxs_grid[i];
    for (uint32_t i = threadIdx.x; i < 128u; i += blockDim.x) s_iq2_signs[i] = cuda_ksigns_iq2xs[i];
    __syncthreads();
    if (xq_blocks <= 16u) {
        for (uint32_t p = 0; p < np; p++) xqb[p] = sxq[p];
    }
    if (row >= expert_mid_dim) return;
    const cuda_block_iq2_xxs *gr = (const cuda_block_iq2_xxs *)(gate_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes);
    const cuda_block_iq2_xxs *ur = (const cuda_block_iq2_xxs *)(up_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes);
    float gate[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    float up[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    for (uint32_t b = lane; b < xq_blocks; b += 8u) {
        dev_dot_iq2_xxs_q8_K_block8_deq_lut(gr + b, xqb[0] ? xqb[0] + b : NULL, xqb[1] ? xqb[1] + b : NULL,
                                            xqb[2] ? xqb[2] + b : NULL, xqb[3] ? xqb[3] + b : NULL,
                                            xqb[4] ? xqb[4] + b : NULL, xqb[5] ? xqb[5] + b : NULL,
                                            xqb[6] ? xqb[6] + b : NULL, xqb[7] ? xqb[7] + b : NULL, np, gate,
                                            s_iq2_grid, s_iq2_signs);
        dev_dot_iq2_xxs_q8_K_block8_deq_lut(ur + b, xqb[0] ? xqb[0] + b : NULL, xqb[1] ? xqb[1] + b : NULL,
                                            xqb[2] ? xqb[2] + b : NULL, xqb[3] ? xqb[3] + b : NULL,
                                            xqb[4] ? xqb[4] + b : NULL, xqb[5] ? xqb[5] + b : NULL,
                                            xqb[6] ? xqb[6] + b : NULL, xqb[7] ? xqb[7] + b : NULL, np, up,
                                            s_iq2_grid, s_iq2_signs);
    }
    for (uint32_t p = 0; p < np; p++) {
        gate[p] = quarter_warp_sum_f32(gate[p], lane);
        up[p] = quarter_warp_sum_f32(up[p], lane);
        if (lane == 0) {
            if (clamp > 1.0e-6f) {
                if (gate[p] > clamp) gate[p] = clamp;
                if (up[p] > clamp) up[p] = clamp;
                if (up[p] < -clamp) up[p] = -clamp;
            }
            const uint64_t off = (uint64_t)pair[p] * expert_mid_dim + row;
            if (write_aux) {
                gate_out[off] = gate[p];
                up_out[off] = up[p];
            }
            mid_out[off] = (gate[p] / (1.0f + expf(-gate[p]))) * up[p] * weights[(uint64_t)tok[p] * n_expert + slot[p]];
        }
    }
}

__global__ static void moe_gate_up_mid_expert_tile8_row2048_kernel(
        float *gate_out,
        float *up_out,
        float *mid_out,
        const char *gate_base,
        const char *up_base,
        const cuda_block_q8_K *xq,
        const uint32_t *sorted_pairs,
        const uint32_t *offsets,
        const uint32_t *counts,
        const uint32_t *tile_total,
        const uint32_t *tile_experts,
        const uint32_t *tile_starts,
        const float *weights,
        uint64_t gate_expert_bytes,
        uint64_t gate_row_bytes,
        uint32_t xq_blocks,
        uint32_t expert_mid_dim,
        uint32_t n_expert,
        uint32_t max_count,
        uint32_t write_aux,
        float clamp) {
    uint32_t tile = blockIdx.y;
    if (tile >= *tile_total) return;
    uint32_t lane = threadIdx.x & 7u;
    uint32_t row_lane = threadIdx.x >> 3u;
    uint32_t expert = tile_experts[tile];
    uint32_t count = counts[expert];
    if (max_count != 0u && count >= max_count) return;
    uint32_t local_start = tile_starts[tile];
    __shared__ cuda_block_q8_K sxq[8][16];
    __shared__ uint64_t s_iq2_grid[256];
    __shared__ uint8_t s_iq2_signs[128];
    uint32_t pair[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    uint32_t tok[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    uint32_t slot[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    const cuda_block_q8_K *xqb[8] = {NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL};
    uint32_t np = 0;
    for (; np < 8u; np++) {
        uint32_t local_pair = local_start + np;
        if (local_pair >= count) break;
        pair[np] = sorted_pairs[offsets[expert] + local_pair];
        tok[np] = pair[np] / n_expert;
        slot[np] = pair[np] - tok[np] * n_expert;
        xqb[np] = xq + (uint64_t)tok[np] * xq_blocks;
    }
    if (xq_blocks <= 16u) {
        for (uint32_t i = threadIdx.x; i < np * xq_blocks; i += blockDim.x) {
            uint32_t p = i / xq_blocks;
            uint32_t b = i - p * xq_blocks;
            sxq[p][b] = xqb[p][b];
        }
    }
    for (uint32_t i = threadIdx.x; i < 256u; i += blockDim.x) s_iq2_grid[i] = cuda_iq2xxs_grid[i];
    for (uint32_t i = threadIdx.x; i < 128u; i += blockDim.x) s_iq2_signs[i] = cuda_ksigns_iq2xs[i];
    __syncthreads();
    if (xq_blocks <= 16u) {
        for (uint32_t p = 0; p < np; p++) xqb[p] = sxq[p];
    }
    for (uint32_t rr = 0; rr < 64u; rr++) {
        uint32_t row = blockIdx.x * 2048u + row_lane + rr * 32u;
        if (row >= expert_mid_dim) continue;
        const cuda_block_iq2_xxs *gr = (const cuda_block_iq2_xxs *)(gate_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes);
        const cuda_block_iq2_xxs *ur = (const cuda_block_iq2_xxs *)(up_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes);
        float gate[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
        float up[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
        for (uint32_t b = lane; b < xq_blocks; b += 8u) {
            dev_dot_iq2_xxs_q8_K_block8_deq_lut(gr + b, xqb[0] ? xqb[0] + b : NULL, xqb[1] ? xqb[1] + b : NULL,
                                                xqb[2] ? xqb[2] + b : NULL, xqb[3] ? xqb[3] + b : NULL,
                                                xqb[4] ? xqb[4] + b : NULL, xqb[5] ? xqb[5] + b : NULL,
                                                xqb[6] ? xqb[6] + b : NULL, xqb[7] ? xqb[7] + b : NULL, np, gate,
                                                s_iq2_grid, s_iq2_signs);
            dev_dot_iq2_xxs_q8_K_block8_deq_lut(ur + b, xqb[0] ? xqb[0] + b : NULL, xqb[1] ? xqb[1] + b : NULL,
                                                xqb[2] ? xqb[2] + b : NULL, xqb[3] ? xqb[3] + b : NULL,
                                                xqb[4] ? xqb[4] + b : NULL, xqb[5] ? xqb[5] + b : NULL,
                                                xqb[6] ? xqb[6] + b : NULL, xqb[7] ? xqb[7] + b : NULL, np, up,
                                                s_iq2_grid, s_iq2_signs);
        }
        for (uint32_t p = 0; p < np; p++) {
            gate[p] = quarter_warp_sum_f32(gate[p], lane);
            up[p] = quarter_warp_sum_f32(up[p], lane);
            if (lane == 0) {
                if (clamp > 1.0e-6f) {
                    if (gate[p] > clamp) gate[p] = clamp;
                    if (up[p] > clamp) up[p] = clamp;
                    if (up[p] < -clamp) up[p] = -clamp;
                }
                const uint64_t off = (uint64_t)pair[p] * expert_mid_dim + row;
                if (write_aux) {
                    gate_out[off] = gate[p];
                    up_out[off] = up[p];
                }
                mid_out[off] = (gate[p] / (1.0f + expf(-gate[p]))) * up[p] * weights[(uint64_t)tok[p] * n_expert + slot[p]];
            }
        }
    }
}

template <uint32_t ROW_SPAN>
__global__ static void moe_gate_up_mid_expert_tile8_rowspan_kernel(
        float *gate_out,
        float *up_out,
        float *mid_out,
        const char *gate_base,
        const char *up_base,
        const cuda_block_q8_K *xq,
        const uint32_t *sorted_pairs,
        const uint32_t *offsets,
        const uint32_t *counts,
        const uint32_t *tile_total,
        const uint32_t *tile_experts,
        const uint32_t *tile_starts,
        const float *weights,
        uint64_t gate_expert_bytes,
        uint64_t gate_row_bytes,
        uint32_t xq_blocks,
        uint32_t expert_mid_dim,
        uint32_t n_expert,
        uint32_t max_count,
        uint32_t write_aux,
        float clamp) {
    uint32_t tile = blockIdx.y;
    if (tile >= *tile_total) return;
    uint32_t lane = threadIdx.x & 7u;
    uint32_t row_lane = threadIdx.x >> 3u;
    uint32_t expert = tile_experts[tile];
    uint32_t count = counts[expert];
    if (max_count != 0u && count >= max_count) return;
    uint32_t local_start = tile_starts[tile];
    __shared__ cuda_block_q8_K sxq[8][16];
    __shared__ uint64_t s_iq2_grid[256];
    __shared__ uint8_t s_iq2_signs[128];
    uint32_t pair[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    uint32_t tok[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    uint32_t slot[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    const cuda_block_q8_K *xqb[8] = {NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL};
    uint32_t np = 0;
    for (; np < 8u; np++) {
        uint32_t local_pair = local_start + np;
        if (local_pair >= count) break;
        pair[np] = sorted_pairs[offsets[expert] + local_pair];
        tok[np] = pair[np] / n_expert;
        slot[np] = pair[np] - tok[np] * n_expert;
        xqb[np] = xq + (uint64_t)tok[np] * xq_blocks;
    }
    if (xq_blocks <= 16u) {
        for (uint32_t i = threadIdx.x; i < np * xq_blocks; i += blockDim.x) {
            uint32_t p = i / xq_blocks;
            uint32_t b = i - p * xq_blocks;
            sxq[p][b] = xqb[p][b];
        }
    }
    for (uint32_t i = threadIdx.x; i < 256u; i += blockDim.x) s_iq2_grid[i] = cuda_iq2xxs_grid[i];
    for (uint32_t i = threadIdx.x; i < 128u; i += blockDim.x) s_iq2_signs[i] = cuda_ksigns_iq2xs[i];
    __syncthreads();
    if (xq_blocks <= 16u) {
        for (uint32_t p = 0; p < np; p++) xqb[p] = sxq[p];
    }
    for (uint32_t rr = 0; rr < ROW_SPAN / 32u; rr++) {
        uint32_t row = blockIdx.x * ROW_SPAN + row_lane + rr * 32u;
        if (row >= expert_mid_dim) continue;
        const cuda_block_iq2_xxs *gr = (const cuda_block_iq2_xxs *)(gate_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes);
        const cuda_block_iq2_xxs *ur = (const cuda_block_iq2_xxs *)(up_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes);
        float gate[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
        float up[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
        for (uint32_t b = lane; b < xq_blocks; b += 8u) {
            dev_dot_iq2_xxs_q8_K_block8_deq_lut(gr + b, xqb[0] ? xqb[0] + b : NULL, xqb[1] ? xqb[1] + b : NULL,
                                                xqb[2] ? xqb[2] + b : NULL, xqb[3] ? xqb[3] + b : NULL,
                                                xqb[4] ? xqb[4] + b : NULL, xqb[5] ? xqb[5] + b : NULL,
                                                xqb[6] ? xqb[6] + b : NULL, xqb[7] ? xqb[7] + b : NULL, np, gate,
                                                s_iq2_grid, s_iq2_signs);
            dev_dot_iq2_xxs_q8_K_block8_deq_lut(ur + b, xqb[0] ? xqb[0] + b : NULL, xqb[1] ? xqb[1] + b : NULL,
                                                xqb[2] ? xqb[2] + b : NULL, xqb[3] ? xqb[3] + b : NULL,
                                                xqb[4] ? xqb[4] + b : NULL, xqb[5] ? xqb[5] + b : NULL,
                                                xqb[6] ? xqb[6] + b : NULL, xqb[7] ? xqb[7] + b : NULL, np, up,
                                                s_iq2_grid, s_iq2_signs);
        }
        for (uint32_t p = 0; p < np; p++) {
            gate[p] = quarter_warp_sum_f32(gate[p], lane);
            up[p] = quarter_warp_sum_f32(up[p], lane);
            if (lane == 0) {
                if (clamp > 1.0e-6f) {
                    if (gate[p] > clamp) gate[p] = clamp;
                    if (up[p] > clamp) up[p] = clamp;
                    if (up[p] < -clamp) up[p] = -clamp;
                }
                const uint64_t off = (uint64_t)pair[p] * expert_mid_dim + row;
                if (write_aux) {
                    gate_out[off] = gate[p];
                    up_out[off] = up[p];
                }
                mid_out[off] = (gate[p] / (1.0f + expf(-gate[p]))) * up[p] * weights[(uint64_t)tok[p] * n_expert + slot[p]];
            }
        }
    }
}

__global__ static void moe_gate_up_mid_sorted_p2_qwarp32_kernel(
        float *gate_out,
        float *up_out,
        float *mid_out,
        const char *gate_base,
        const char *up_base,
        const cuda_block_q8_K *xq,
        const uint32_t *sorted_pairs,
        const int32_t *selected,
        const float *weights,
        uint64_t gate_expert_bytes,
        uint64_t gate_row_bytes,
        uint32_t xq_blocks,
        uint32_t expert_mid_dim,
        uint32_t n_expert,
        uint32_t pair_count,
        float clamp) {
    uint32_t lane = threadIdx.x & 7u;
    uint32_t pair_lane = (threadIdx.x >> 3u) & 1u;
    uint32_t row = blockIdx.x * 16u + (threadIdx.x >> 4u);
    uint32_t sorted_idx = blockIdx.y * 2u + pair_lane;
    if (row >= expert_mid_dim || sorted_idx >= pair_count) return;
    uint32_t pair = sorted_pairs[sorted_idx];
    uint32_t tok = pair / n_expert;
    uint32_t slot = pair - tok * n_expert;
    int32_t expert_i = selected[(uint64_t)tok * n_expert + slot];
    if (expert_i < 0) expert_i = 0;
    uint32_t expert = (uint32_t)expert_i;
    const cuda_block_iq2_xxs *gr = (const cuda_block_iq2_xxs *)(gate_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes);
    const cuda_block_iq2_xxs *ur = (const cuda_block_iq2_xxs *)(up_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes);
    const cuda_block_q8_K *xqb = xq + (uint64_t)tok * xq_blocks;
    float gate = 0.0f;
    float up = 0.0f;
    for (uint32_t b = lane; b < xq_blocks; b += 8u) {
        gate += dev_dot_iq2_xxs_q8_K_block(gr + b, xqb + b);
        up += dev_dot_iq2_xxs_q8_K_block(ur + b, xqb + b);
    }
    gate = quarter_warp_sum_f32(gate, lane);
    up = quarter_warp_sum_f32(up, lane);
    if (lane == 0) {
        if (clamp > 1.0e-6f) {
            if (gate > clamp) gate = clamp;
            if (up > clamp) up = clamp;
            if (up < -clamp) up = -clamp;
        }
        const uint64_t off = (uint64_t)pair * expert_mid_dim + row;
        gate_out[off] = gate;
        up_out[off] = up;
        mid_out[off] = (gate / (1.0f + expf(-gate))) * up * weights[(uint64_t)tok * n_expert + slot];
    }
}

__global__ static void moe_gate_up_mid_q4K_sorted_qwarp32_kernel(
        float *gate_out,
        float *up_out,
        float *mid_out,
        const char *gate_base,
        const char *up_base,
        const cuda_block_q8_K *xq,
        const uint32_t *sorted_pairs,
        const int32_t *selected,
        const float *weights,
        uint64_t gate_expert_bytes,
        uint64_t gate_row_bytes,
        uint32_t xq_blocks,
        uint32_t expert_mid_dim,
        uint32_t n_expert,
        float clamp) {
    uint32_t lane = threadIdx.x & 7u;
    uint32_t row = blockIdx.x * 32u + (threadIdx.x >> 3u);
    uint32_t pair = sorted_pairs[blockIdx.y];
    if (row >= expert_mid_dim) return;
    uint32_t tok = pair / n_expert;
    uint32_t slot = pair - tok * n_expert;
    int32_t expert_i = selected[(uint64_t)tok * n_expert + slot];
    if (expert_i < 0) expert_i = 0;
    uint32_t expert = (uint32_t)expert_i;
    const cuda_block_q4_K *gr = (const cuda_block_q4_K *)(gate_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes);
    const cuda_block_q4_K *ur = (const cuda_block_q4_K *)(up_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes);
    const cuda_block_q8_K *xqb = xq + (uint64_t)tok * xq_blocks;
    float gate = 0.0f;
    float up = 0.0f;
    for (uint32_t b = lane; b < xq_blocks; b += 8u) {
        gate += dev_dot_q4_K_q8_K_block(gr + b, xqb + b);
        up += dev_dot_q4_K_q8_K_block(ur + b, xqb + b);
    }
    gate = quarter_warp_sum_f32(gate, lane);
    up = quarter_warp_sum_f32(up, lane);
    if (lane == 0) {
        if (clamp > 1.0e-6f) {
            if (gate > clamp) gate = clamp;
            if (up > clamp) up = clamp;
            if (up < -clamp) up = -clamp;
        }
        const uint64_t off = (uint64_t)pair * expert_mid_dim + row;
        gate_out[off] = gate;
        up_out[off] = up;
        mid_out[off] = (gate / (1.0f + expf(-gate))) * up * weights[(uint64_t)tok * n_expert + slot];
    }
}

__global__ static void moe_gate_up_mid_q4K_expert_tile4_row32_kernel(
        float *gate_out,
        float *up_out,
        float *mid_out,
        const char *gate_base,
        const char *up_base,
        const cuda_block_q8_K *xq,
        const uint32_t *sorted_pairs,
        const uint32_t *offsets,
        const uint32_t *counts,
        const uint32_t *tile_total,
        const uint32_t *tile_experts,
        const uint32_t *tile_starts,
        const float *weights,
        uint64_t gate_expert_bytes,
        uint64_t gate_row_bytes,
        uint32_t xq_blocks,
        uint32_t expert_mid_dim,
        uint32_t n_expert,
        uint32_t max_count,
        uint32_t write_aux,
        float clamp) {
    uint32_t tile = blockIdx.y;
    if (tile >= *tile_total) return;
    uint32_t lane = threadIdx.x & 7u;
    uint32_t row = blockIdx.x * 32u + (threadIdx.x >> 3u);
    uint32_t expert = tile_experts[tile];
    uint32_t count = counts[expert];
    if (max_count != 0u && count >= max_count) return;
    uint32_t local_start = tile_starts[tile];
    __shared__ cuda_block_q8_K sxq[4][16];
    uint32_t pair[4] = {0, 0, 0, 0};
    uint32_t tok[4] = {0, 0, 0, 0};
    uint32_t slot[4] = {0, 0, 0, 0};
    const cuda_block_q8_K *xqb[4] = {NULL, NULL, NULL, NULL};
    uint32_t np = 0;
    for (; np < 4u; np++) {
        uint32_t local_pair = local_start + np;
        if (local_pair >= count) break;
        pair[np] = sorted_pairs[offsets[expert] + local_pair];
        tok[np] = pair[np] / n_expert;
        slot[np] = pair[np] - tok[np] * n_expert;
        xqb[np] = xq + (uint64_t)tok[np] * xq_blocks;
    }
    if (xq_blocks <= 16u) {
        for (uint32_t i = threadIdx.x; i < np * xq_blocks; i += blockDim.x) {
            uint32_t p = i / xq_blocks;
            uint32_t b = i - p * xq_blocks;
            sxq[p][b] = xqb[p][b];
        }
        __syncthreads();
        for (uint32_t p = 0; p < np; p++) xqb[p] = sxq[p];
    }
    if (row >= expert_mid_dim) return;
    const cuda_block_q4_K *gr = (const cuda_block_q4_K *)(gate_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes);
    const cuda_block_q4_K *ur = (const cuda_block_q4_K *)(up_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes);
    float gate[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float up[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (uint32_t b = lane; b < xq_blocks; b += 8u) {
        dev_dot_q4_K_q8_K_block4(gr + b, xqb[0] ? xqb[0] + b : NULL, xqb[1] ? xqb[1] + b : NULL,
                                 xqb[2] ? xqb[2] + b : NULL, xqb[3] ? xqb[3] + b : NULL, np, gate);
        dev_dot_q4_K_q8_K_block4(ur + b, xqb[0] ? xqb[0] + b : NULL, xqb[1] ? xqb[1] + b : NULL,
                                 xqb[2] ? xqb[2] + b : NULL, xqb[3] ? xqb[3] + b : NULL, np, up);
    }
    for (uint32_t p = 0; p < np; p++) {
        gate[p] = quarter_warp_sum_f32(gate[p], lane);
        up[p] = quarter_warp_sum_f32(up[p], lane);
        if (lane == 0) {
            if (clamp > 1.0e-6f) {
                if (gate[p] > clamp) gate[p] = clamp;
                if (up[p] > clamp) up[p] = clamp;
                if (up[p] < -clamp) up[p] = -clamp;
            }
            const uint64_t off = (uint64_t)pair[p] * expert_mid_dim + row;
            if (write_aux) {
                gate_out[off] = gate[p];
                up_out[off] = up[p];
            }
            mid_out[off] = (gate[p] / (1.0f + expf(-gate[p]))) * up[p] * weights[(uint64_t)tok[p] * n_expert + slot[p]];
        }
    }
}

__global__ static void moe_gate_up_mid_q4K_expert_tile8_row32_kernel(
        float *gate_out,
        float *up_out,
        float *mid_out,
        const char *gate_base,
        const char *up_base,
        const cuda_block_q8_K *xq,
        const uint32_t *sorted_pairs,
        const uint32_t *offsets,
        const uint32_t *counts,
        const uint32_t *tile_total,
        const uint32_t *tile_experts,
        const uint32_t *tile_starts,
        const float *weights,
        uint64_t gate_expert_bytes,
        uint64_t gate_row_bytes,
        uint32_t xq_blocks,
        uint32_t expert_mid_dim,
        uint32_t n_expert,
        uint32_t max_count,
        uint32_t write_aux,
        float clamp) {
    uint32_t tile = blockIdx.y;
    if (tile >= *tile_total) return;
    uint32_t lane = threadIdx.x & 7u;
    uint32_t row = blockIdx.x * 32u + (threadIdx.x >> 3u);
    uint32_t expert = tile_experts[tile];
    uint32_t count = counts[expert];
    if (max_count != 0u && count >= max_count) return;
    uint32_t local_start = tile_starts[tile];
    __shared__ cuda_block_q8_K sxq[8][16];
    uint32_t pair[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    uint32_t tok[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    uint32_t slot[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    const cuda_block_q8_K *xqb[8] = {NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL};
    uint32_t np = 0;
    for (; np < 8u; np++) {
        uint32_t local_pair = local_start + np;
        if (local_pair >= count) break;
        pair[np] = sorted_pairs[offsets[expert] + local_pair];
        tok[np] = pair[np] / n_expert;
        slot[np] = pair[np] - tok[np] * n_expert;
        xqb[np] = xq + (uint64_t)tok[np] * xq_blocks;
    }
    if (xq_blocks <= 16u) {
        for (uint32_t i = threadIdx.x; i < np * xq_blocks; i += blockDim.x) {
            uint32_t p = i / xq_blocks;
            uint32_t b = i - p * xq_blocks;
            sxq[p][b] = xqb[p][b];
        }
        __syncthreads();
        for (uint32_t p = 0; p < np; p++) xqb[p] = sxq[p];
    }
    if (row >= expert_mid_dim) return;
    const cuda_block_q4_K *gr = (const cuda_block_q4_K *)(gate_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes);
    const cuda_block_q4_K *ur = (const cuda_block_q4_K *)(up_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes);
    float gate[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    float up[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    for (uint32_t b = lane; b < xq_blocks; b += 8u) {
        dev_dot_q4_K_q8_K_block8(gr + b, xqb[0] ? xqb[0] + b : NULL, xqb[1] ? xqb[1] + b : NULL,
                                 xqb[2] ? xqb[2] + b : NULL, xqb[3] ? xqb[3] + b : NULL,
                                 xqb[4] ? xqb[4] + b : NULL, xqb[5] ? xqb[5] + b : NULL,
                                 xqb[6] ? xqb[6] + b : NULL, xqb[7] ? xqb[7] + b : NULL, np, gate);
        dev_dot_q4_K_q8_K_block8(ur + b, xqb[0] ? xqb[0] + b : NULL, xqb[1] ? xqb[1] + b : NULL,
                                 xqb[2] ? xqb[2] + b : NULL, xqb[3] ? xqb[3] + b : NULL,
                                 xqb[4] ? xqb[4] + b : NULL, xqb[5] ? xqb[5] + b : NULL,
                                 xqb[6] ? xqb[6] + b : NULL, xqb[7] ? xqb[7] + b : NULL, np, up);
    }
    for (uint32_t p = 0; p < np; p++) {
        gate[p] = quarter_warp_sum_f32(gate[p], lane);
        up[p] = quarter_warp_sum_f32(up[p], lane);
        if (lane == 0) {
            if (clamp > 1.0e-6f) {
                if (gate[p] > clamp) gate[p] = clamp;
                if (up[p] > clamp) up[p] = clamp;
                if (up[p] < -clamp) up[p] = -clamp;
            }
            const uint64_t off = (uint64_t)pair[p] * expert_mid_dim + row;
            if (write_aux) {
                gate_out[off] = gate[p];
                up_out[off] = up[p];
            }
            mid_out[off] = (gate[p] / (1.0f + expf(-gate[p]))) * up[p] * weights[(uint64_t)tok[p] * n_expert + slot[p]];
        }
    }
}

__global__ static void moe_gate_up_mid_mxfp4_expert_tile8_row32_kernel(
        float *gate_out,
        float *up_out,
        float *mid_out,
        const char *gate_base,
        const char *up_base,
        const cuda_block_q8_K *xq,
        const uint32_t *sorted_pairs,
        const uint32_t *offsets,
        const uint32_t *counts,
        const uint32_t *tile_total,
        const uint32_t *tile_experts,
        const uint32_t *tile_starts,
        const float *weights,
        uint64_t gate_expert_bytes,
        uint64_t gate_row_bytes,
        uint32_t xq_blocks,
        uint32_t expert_mid_dim,
        uint32_t n_expert,
        uint32_t max_count,
        uint32_t write_aux,
        float clamp) {
    /* Grid is (tile, row-block): consecutive blocks re-walk the same
     * 32-row weight slice across an expert's token tiles, so the slice
     * stays L2-resident and the expert's weights stream from DRAM ~once
     * instead of once per 8-token tile.  Pure scheduling change; every
     * output element's arithmetic is untouched. */
    uint32_t tile = blockIdx.x;
    if (tile >= *tile_total) return;
    uint32_t lane = threadIdx.x & 7u;
    uint32_t row = blockIdx.y * 32u + (threadIdx.x >> 3u);
    uint32_t expert = tile_experts[tile];
    uint32_t count = counts[expert];
    if (max_count != 0u && count >= max_count) return;
    uint32_t local_start = tile_starts[tile];
    /* Staged activations live as 16-byte-aligned quant slices with the
     * Q8_K scales in a separate array: the dot helper is issue-bound, and
     * the aligned layout lets it load b128 instead of four dwords.  The
     * layout also drops the unused 32-byte bsums from every staged chunk.
     * Sized by the launch to 8 * xq_blocks * (256 + 4) bytes. */
    extern __shared__ int4 sxq4[];
    int8_t *sqs = (int8_t *)sxq4;
    float *sds = (float *)(sqs + (uint64_t)8u * xq_blocks * 256u);
    uint32_t pair[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    /* Fixed-count predicated setup keeps pair/xqb in registers instead of
     * dynamically indexed scratch.  The tail xqb entries stay at valid
     * dummy rows and are never read past np. */
    const cuda_block_q8_K *xqb[8] = { xq, xq, xq, xq, xq, xq, xq, xq };
    uint32_t np = count - local_start;
    if (np > 8u) np = 8u;
    #pragma unroll
    for (uint32_t p = 0; p < 8u; p++) {
        if (p < np) {
            pair[p] = sorted_pairs[offsets[expert] + local_start + p];
            xqb[p] = xq + (uint64_t)(pair[p] / n_expert) * xq_blocks;
        }
    }
    const int staged = xq_blocks <= 28u;
    if (staged) {
        for (uint32_t i = threadIdx.x; i < np * xq_blocks * 16u; i += blockDim.x) {
            const uint32_t u = i / (xq_blocks * 16u);
            const uint32_t rem = i - u * (xq_blocks * 16u);
            const uint32_t b = rem >> 4u;
            const uint32_t k = rem & 15u;
            const uint32_t sp = sorted_pairs[offsets[expert] + local_start + u];
            const int32_t *src = (const int32_t *)
                xq[(uint64_t)(sp / n_expert) * xq_blocks + b].qs + k * 4u;
            ((int4 *)sqs)[(u * xq_blocks + b) * 16u + k] =
                make_int4(src[0], src[1], src[2], src[3]);
        }
        for (uint32_t i = threadIdx.x; i < np * xq_blocks; i += blockDim.x) {
            const uint32_t u = i / xq_blocks;
            const uint32_t b = i - u * xq_blocks;
            const uint32_t sp = sorted_pairs[offsets[expert] + local_start + u];
            sds[i] = xq[(uint64_t)(sp / n_expert) * xq_blocks + b].d;
        }
        __syncthreads();
    }
    if (row >= expert_mid_dim) return;
    const char *gate_row = gate_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes;
    const char *up_row = up_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes;
    const uint64_t gate_chunk_bytes = gate_row_bytes / xq_blocks;
    float gate[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    float up[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    const uint64_t sq = (uint64_t)xq_blocks << 8u;
    for (uint32_t b = lane; b < xq_blocks; b += 8u) {
        const cuda_block_mxfp4 *gb = (const cuda_block_mxfp4 *)(gate_row + (uint64_t)b * gate_chunk_bytes);
        const cuda_block_mxfp4 *ub = (const cuda_block_mxfp4 *)(up_row + (uint64_t)b * gate_chunk_bytes);
        if (staged) {
            const int8_t *qb = sqs + ((uint64_t)b << 8u);
            dev_dot_mxfp4_q8a_pair_block8(gb, ub,
                qb, qb + sq, qb + 2u * sq, qb + 3u * sq,
                qb + 4u * sq, qb + 5u * sq, qb + 6u * sq, qb + 7u * sq,
                sds[b], sds[xq_blocks + b], sds[2u * xq_blocks + b], sds[3u * xq_blocks + b],
                sds[4u * xq_blocks + b], sds[5u * xq_blocks + b], sds[6u * xq_blocks + b], sds[7u * xq_blocks + b],
                np, gate, up);
        } else {
            dev_dot_mxfp4_q8_K_block8(gb, xqb[0] + b, xqb[1] + b, xqb[2] + b, xqb[3] + b,
                                      xqb[4] + b, xqb[5] + b, xqb[6] + b, xqb[7] + b, np, gate);
            dev_dot_mxfp4_q8_K_block8(ub, xqb[0] + b, xqb[1] + b, xqb[2] + b, xqb[3] + b,
                                      xqb[4] + b, xqb[5] + b, xqb[6] + b, xqb[7] + b, np, up);
        }
    }
    /* pair == tok * n_expert + slot, so it indexes weights directly. */
    #pragma unroll
    for (uint32_t p = 0; p < 8u; p++) {
        if (p >= np) continue;
        gate[p] = quarter_warp_sum_f32(gate[p], lane);
        up[p] = quarter_warp_sum_f32(up[p], lane);
        if (lane == 0) {
            if (clamp > 1.0e-6f) {
                if (gate[p] > clamp) gate[p] = clamp;
                if (up[p] > clamp) up[p] = clamp;
                if (up[p] < -clamp) up[p] = -clamp;
            }
            const uint64_t off = (uint64_t)pair[p] * expert_mid_dim + row;
            if (write_aux) {
                gate_out[off] = gate[p];
                up_out[off] = up[p];
            }
            mid_out[off] = (gate[p] / (1.0f + expf(-gate[p]))) * up[p] * weights[pair[p]];
        }
    }
}

/* Generic-tile MXFP4 gate/up kernel: TILE_TOKENS tokens per expert tile,
 * QWARPS 32-row slots per block (blockDim = QWARPS*8).  Lets the launch
 * trade staging footprint against resident-warp count: tile4 stages
 * TILE_TOKENS*xq_blocks*260 B (29.1 KiB at 7168 dims -> 4 blocks/WGP),
 * row64 keeps 8-token staging but runs 512 threads (2 blocks/WGP -> 32
 * resident warps).  Both target the tile8 kernel's latency-bound profile
 * (1-2 blocks/WGP today) without touching arithmetic.
 *
 * Exactness: per (token,row) output the accumulation order is identical
 * to moe_gate_up_mid_mxfp4_expert_tile8_row32_kernel - same 8-lane split
 * of the q8_K block loop, same dev_dot helpers, same quarter-warp
 * reduction, same epilogue.  Only grid/tile geometry changes. */
template <uint32_t TILE_TOKENS, uint32_t QWARPS>
__global__ static void moe_gate_up_mid_mxfp4_expert_tileN_kernel(
        float *gate_out,
        float *up_out,
        float *mid_out,
        const char *gate_base,
        const char *up_base,
        const cuda_block_q8_K *xq,
        const uint32_t *sorted_pairs,
        const uint32_t *offsets,
        const uint32_t *counts,
        const uint32_t *tile_total,
        const uint32_t *tile_experts,
        const uint32_t *tile_starts,
        const float *weights,
        uint64_t gate_expert_bytes,
        uint64_t gate_row_bytes,
        uint32_t xq_blocks,
        uint32_t expert_mid_dim,
        uint32_t n_expert,
        uint32_t max_count,
        uint32_t write_aux,
        float clamp) {
    uint32_t tile = blockIdx.x;
    if (tile >= *tile_total) return;
    uint32_t lane = threadIdx.x & 7u;
    uint32_t row = blockIdx.y * QWARPS + (threadIdx.x >> 3u);
    uint32_t expert = tile_experts[tile];
    uint32_t count = counts[expert];
    if (max_count != 0u && count >= max_count) return;
    uint32_t local_start = tile_starts[tile];
    extern __shared__ int4 sxq4[];
    int8_t *sqs = (int8_t *)sxq4;
    float *sds = (float *)(sqs + (uint64_t)TILE_TOKENS * xq_blocks * 256u);
    uint32_t pair[TILE_TOKENS] = {0};
    const cuda_block_q8_K *xqb[TILE_TOKENS];
    #pragma unroll
    for (uint32_t p = 0; p < TILE_TOKENS; p++) xqb[p] = xq;
    uint32_t np = count - local_start;
    if (np > TILE_TOKENS) np = TILE_TOKENS;
    #pragma unroll
    for (uint32_t p = 0; p < TILE_TOKENS; p++) {
        if (p < np) {
            pair[p] = sorted_pairs[offsets[expert] + local_start + p];
            xqb[p] = xq + (uint64_t)(pair[p] / n_expert) * xq_blocks;
        }
    }
    const int staged = xq_blocks <= 28u;
    if (staged) {
        for (uint32_t i = threadIdx.x; i < np * xq_blocks * 16u; i += blockDim.x) {
            const uint32_t u = i / (xq_blocks * 16u);
            const uint32_t rem = i - u * (xq_blocks * 16u);
            const uint32_t b = rem >> 4u;
            const uint32_t k = rem & 15u;
            const uint32_t sp = sorted_pairs[offsets[expert] + local_start + u];
            const int32_t *src = (const int32_t *)
                xq[(uint64_t)(sp / n_expert) * xq_blocks + b].qs + k * 4u;
            ((int4 *)sqs)[(u * xq_blocks + b) * 16u + k] =
                make_int4(src[0], src[1], src[2], src[3]);
        }
        for (uint32_t i = threadIdx.x; i < np * xq_blocks; i += blockDim.x) {
            const uint32_t u = i / xq_blocks;
            const uint32_t b = i - u * xq_blocks;
            const uint32_t sp = sorted_pairs[offsets[expert] + local_start + u];
            sds[i] = xq[(uint64_t)(sp / n_expert) * xq_blocks + b].d;
        }
        __syncthreads();
    }
    if (row >= expert_mid_dim) return;
    const char *gate_row = gate_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes;
    const char *up_row = up_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes;
    const uint64_t gate_chunk_bytes = gate_row_bytes / xq_blocks;
    float gate[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    float up[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    const uint64_t sq = (uint64_t)xq_blocks << 8u;
    for (uint32_t b = lane; b < xq_blocks; b += 8u) {
        const cuda_block_mxfp4 *gb = (const cuda_block_mxfp4 *)(gate_row + (uint64_t)b * gate_chunk_bytes);
        const cuda_block_mxfp4 *ub = (const cuda_block_mxfp4 *)(up_row + (uint64_t)b * gate_chunk_bytes);
        if (staged) {
            const int8_t *qb = sqs + ((uint64_t)b << 8u);
            const int8_t *qp[8];
            float sd[8];
            #pragma unroll
            for (uint32_t p = 0; p < 8u; p++) {
                qp[p] = qb + (p < TILE_TOKENS ? p : 0u) * sq;
                sd[p] = sds[(p < TILE_TOKENS ? p : 0u) * xq_blocks + b];
            }
            dev_dot_mxfp4_q8a_pair_block8(gb, ub,
                qp[0], qp[1], qp[2], qp[3], qp[4], qp[5], qp[6], qp[7],
                sd[0], sd[1], sd[2], sd[3], sd[4], sd[5], sd[6], sd[7],
                np, gate, up);
        } else {
            const cuda_block_q8_K *xq8[8];
            #pragma unroll
            for (uint32_t p = 0; p < 8u; p++) xq8[p] = xqb[p < TILE_TOKENS ? p : 0u] + b;
            dev_dot_mxfp4_q8_K_block8(gb, xq8[0], xq8[1], xq8[2], xq8[3],
                                      xq8[4], xq8[5], xq8[6], xq8[7], np, gate);
            dev_dot_mxfp4_q8_K_block8(ub, xq8[0], xq8[1], xq8[2], xq8[3],
                                      xq8[4], xq8[5], xq8[6], xq8[7], np, up);
        }
    }
    /* pair == tok * n_expert + slot, so it indexes weights directly. */
    #pragma unroll
    for (uint32_t p = 0; p < TILE_TOKENS; p++) {
        if (p >= np) continue;
        gate[p] = quarter_warp_sum_f32(gate[p], lane);
        up[p] = quarter_warp_sum_f32(up[p], lane);
        if (lane == 0) {
            if (clamp > 1.0e-6f) {
                if (gate[p] > clamp) gate[p] = clamp;
                if (up[p] > clamp) up[p] = clamp;
                if (up[p] < -clamp) up[p] = -clamp;
            }
            const uint64_t off = (uint64_t)pair[p] * expert_mid_dim + row;
            if (write_aux) {
                gate_out[off] = gate[p];
                up_out[off] = up[p];
            }
            mid_out[off] = (gate[p] / (1.0f + expf(-gate[p]))) * up[p] * weights[pair[p]];
        }
    }
}

/* Prefill-oriented MXFP4 gate/up kernel: one block covers 8 output rows
 * for up to 128 tokens of a single expert, with the 8 gate+up weight
 * rows staged in LDS (16 * gate_row_bytes; 60.9 KiB at row_bytes=3808).
 * Each expert's weights are then read from DRAM ~once per 128-token
 * tile instead of once per 8-token tile (16x re-read reduction at
 * balanced routing), while the quantized activations stream from global
 * (the expert's ~1 MiB xq working set stays L2-resident across its row
 * blocks).
 *
 * Exactness: per (token,row) output the accumulation order is identical
 * to moe_gate_up_mid_mxfp4_expert_tile8_row32_kernel - same 8-lane split
 * of the q8_K block loop, same dev_dot_mxfp4_q8_K_block8 helper, same
 * quarter-warp reduction, same epilogue.  Only data movement changes. */
__global__ static void moe_gate_up_mid_mxfp4_expert_row8_ldsB_kernel(
        float *gate_out,
        float *up_out,
        float *mid_out,
        const char *gate_base,
        const char *up_base,
        const cuda_block_q8_K *xq,
        const uint32_t *sorted_pairs,
        const uint32_t *offsets,
        const uint32_t *counts,
        const uint32_t *tile_total,
        const uint32_t *tile_experts,
        const uint32_t *tile_starts,
        const float *weights,
        uint64_t gate_expert_bytes,
        uint64_t gate_row_bytes,
        uint32_t xq_blocks,
        uint32_t expert_mid_dim,
        uint32_t n_expert,
        uint32_t max_count,
        uint32_t write_aux,
        float clamp) {
    uint32_t tile = blockIdx.y;
    if (tile >= *tile_total) return;
    const uint32_t expert = tile_experts[tile];
    const uint32_t count = counts[expert];
    if (max_count != 0u && count >= max_count) return;
    const uint32_t local_start = tile_starts[tile];
    const uint32_t row0 = blockIdx.x * 8u;
    if (row0 >= expert_mid_dim) return;

    /* LDS layout: 16 segments (gate rows 0..7 then up rows 0..7), each
     * gate_row_bytes long, laid out exactly like the global weight row so
     * the dot helper can point into it directly. */
    extern __shared__ char sB[];
    {
        const uint32_t nseg4 = (uint32_t)(gate_row_bytes >> 4u); /* 16 B units per row */
        const int4 *g4 = (const int4 *)(gate_base + (uint64_t)expert * gate_expert_bytes) + (uint64_t)row0 * nseg4;
        const int4 *u4 = (const int4 *)(up_base + (uint64_t)expert * gate_expert_bytes) + (uint64_t)row0 * nseg4;
        int4 *dst = (int4 *)sB;
        for (uint32_t m = 0; m < 2u; m++) {
            const int4 *src = m == 0u ? g4 : u4;
            for (uint32_t r = 0; r < 8u; r++) {
                for (uint32_t i = threadIdx.x; i < nseg4; i += blockDim.x) {
                    dst[(m * 8u + r) * nseg4 + i] = src[r * nseg4 + i];
                }
            }
        }
        __syncthreads();
    }

    const uint32_t lane = threadIdx.x & 7u;
    const uint32_t qwarp = threadIdx.x >> 3u;          /* 0..31 */
    const uint32_t row_in = qwarp & 7u;                /* row within the block's 8 */
    const uint32_t tok_grp = qwarp >> 3u;              /* 4 token groups of 8 per round */
    const uint32_t row = row0 + row_in;
    const uint32_t avail = count - local_start;

    for (uint32_t round = 0u; round * 32u < avail; round++) {
        const uint32_t t0 = local_start + round * 32u + tok_grp * 8u;
        uint32_t np = (t0 < count) ? count - t0 : 0u;
        if (np > 8u) np = 8u;
        uint32_t pair[8] = {0, 0, 0, 0, 0, 0, 0, 0};
        const cuda_block_q8_K *xqb[8] = { xq, xq, xq, xq, xq, xq, xq, xq };
        #pragma unroll
        for (uint32_t p = 0; p < 8u; p++) {
            if (p < np) {
                pair[p] = sorted_pairs[offsets[expert] + t0 + p];
                xqb[p] = xq + (uint64_t)(pair[p] / n_expert) * xq_blocks;
            }
        }
        const char *sB_gate = sB + (uint64_t)row_in * gate_row_bytes;
        const char *sB_up   = sB + (uint64_t)(8u + row_in) * gate_row_bytes;
        const uint64_t chunk_bytes = gate_row_bytes / xq_blocks;
        float gate[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
        float up[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
        for (uint32_t b = lane; b < xq_blocks; b += 8u) {
            const cuda_block_mxfp4 *gb = (const cuda_block_mxfp4 *)(sB_gate + (uint64_t)b * chunk_bytes);
            const cuda_block_mxfp4 *ub = (const cuda_block_mxfp4 *)(sB_up + (uint64_t)b * chunk_bytes);
            dev_dot_mxfp4_q8_K_block8(gb, xqb[0] + b, xqb[1] + b, xqb[2] + b, xqb[3] + b,
                                      xqb[4] + b, xqb[5] + b, xqb[6] + b, xqb[7] + b, np, gate);
            dev_dot_mxfp4_q8_K_block8(ub, xqb[0] + b, xqb[1] + b, xqb[2] + b, xqb[3] + b,
                                      xqb[4] + b, xqb[5] + b, xqb[6] + b, xqb[7] + b, np, up);
        }
        #pragma unroll
        for (uint32_t p = 0; p < 8u; p++) {
            if (p >= np) continue;
            gate[p] = quarter_warp_sum_f32(gate[p], lane);
            up[p] = quarter_warp_sum_f32(up[p], lane);
            if (lane == 0) {
                if (clamp > 1.0e-6f) {
                    if (gate[p] > clamp) gate[p] = clamp;
                    if (up[p] > clamp) up[p] = clamp;
                    if (up[p] < -clamp) up[p] = -clamp;
                }
                const uint64_t off = (uint64_t)pair[p] * expert_mid_dim + row;
                if (write_aux) {
                    gate_out[off] = gate[p];
                    up_out[off] = up[p];
                }
                mid_out[off] = (gate[p] / (1.0f + expf(-gate[p]))) * up[p] * weights[pair[p]];
            }
        }
    }
}

/* MXFP4 prefill gate/up tile32 kernel: each quarter-warp owns one output
 * row and FOUR 8-token groups (32 tokens), so the 17-byte weight chunks it
 * loads per q8_K block are reused across 32 tokens instead of 8.  At
 * balanced routing (128 tokens/expert) this cuts the expert-weight DRAM
 * re-read factor from 16x to 4x; the production tile8 path is bound by
 * exactly that traffic (~64 GB per 10.5k-token prefill).
 *
 * Exactness: identical to the tile8 kernel per (token,row) output - same
 * 8-lane split of the q8_K block loop, same dev_dot_mxfp4_q8_K_block8
 * helper per 8-token group, same quarter-warp reduction and epilogue.
 * Only the weight-load reuse changes. */
__global__ static void moe_gate_up_mid_mxfp4_expert_tile32_row32_kernel(
        float *gate_out,
        float *up_out,
        float *mid_out,
        const char *gate_base,
        const char *up_base,
        const cuda_block_q8_K *xq,
        const uint32_t *sorted_pairs,
        const uint32_t *offsets,
        const uint32_t *counts,
        const uint32_t *tile_total,
        const uint32_t *tile_experts,
        const uint32_t *tile_starts,
        const float *weights,
        uint64_t gate_expert_bytes,
        uint64_t gate_row_bytes,
        uint32_t xq_blocks,
        uint32_t expert_mid_dim,
        uint32_t n_expert,
        uint32_t max_count,
        uint32_t write_aux,
        float clamp) {
    const uint32_t tile = blockIdx.y;
    if (tile >= *tile_total) return;
    const uint32_t lane = threadIdx.x & 7u;
    const uint32_t row = blockIdx.x * 32u + (threadIdx.x >> 3u);
    const uint32_t expert = tile_experts[tile];
    const uint32_t count = counts[expert];
    if (max_count != 0u && count >= max_count) return;
    const uint32_t local_start = tile_starts[tile];
    if (row >= expert_mid_dim) return;
    const char *gate_row = gate_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes;
    const char *up_row = up_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes;
    const uint64_t chunk_bytes = gate_row_bytes / xq_blocks;

    uint32_t pair[32];
    uint32_t np4[4] = {0u, 0u, 0u, 0u};
    #pragma unroll
    for (uint32_t g = 0; g < 4u; g++) {
        const uint32_t t0 = local_start + g * 8u;
        uint32_t np = (t0 < count) ? count - t0 : 0u;
        if (np > 8u) np = 8u;
        np4[g] = np;
        #pragma unroll
        for (uint32_t p = 0; p < 8u; p++) {
            pair[g * 8u + p] = (p < np)
                ? sorted_pairs[offsets[expert] + t0 + p]
                : 0u;
        }
    }
    float gate[32];
    float up[32];
    #pragma unroll
    for (uint32_t i = 0; i < 32u; i++) { gate[i] = 0.0f; up[i] = 0.0f; }

    for (uint32_t b = lane; b < xq_blocks; b += 8u) {
        const cuda_block_mxfp4 *gb = (const cuda_block_mxfp4 *)(gate_row + (uint64_t)b * chunk_bytes);
        const cuda_block_mxfp4 *ub = (const cuda_block_mxfp4 *)(up_row + (uint64_t)b * chunk_bytes);
        #pragma unroll
        for (uint32_t g = 0; g < 4u; g++) {
            if (np4[g] == 0u) continue;
            const uint32_t tok0 = pair[g * 8u + 0] / n_expert;
            const cuda_block_q8_K *xg = xq + (uint64_t)tok0 * xq_blocks;
            /* Groups are 8 consecutive sorted pairs of one expert; pairs
             * share the expert, but the tokens are arbitrary, so the xq
             * base must be per-token.  Build the 8 pointers for this
             * group (registers reused across the b loop). */
            const cuda_block_q8_K *xp[8];
            #pragma unroll
            for (uint32_t p = 0; p < 8u; p++) {
                xp[p] = (p < np4[g])
                    ? xq + (uint64_t)(pair[g * 8u + p] / n_expert) * xq_blocks + b
                    : xg + b;
            }
            dev_dot_mxfp4_q8_K_block8(gb, xp[0], xp[1], xp[2], xp[3],
                                      xp[4], xp[5], xp[6], xp[7], np4[g], gate + g * 8u);
            dev_dot_mxfp4_q8_K_block8(ub, xp[0], xp[1], xp[2], xp[3],
                                      xp[4], xp[5], xp[6], xp[7], np4[g], up + g * 8u);
        }
    }
    #pragma unroll
    for (uint32_t g = 0; g < 4u; g++) {
        #pragma unroll
        for (uint32_t p = 0; p < 8u; p++) {
            if (p >= np4[g]) continue;
            const uint32_t i = g * 8u + p;
            gate[i] = quarter_warp_sum_f32(gate[i], lane);
            up[i] = quarter_warp_sum_f32(up[i], lane);
            if (lane == 0) {
                if (clamp > 1.0e-6f) {
                    if (gate[i] > clamp) gate[i] = clamp;
                    if (up[i] > clamp) up[i] = clamp;
                    if (up[i] < -clamp) up[i] = -clamp;
                }
                const uint64_t off = (uint64_t)pair[i] * expert_mid_dim + row;
                if (write_aux) {
                    gate_out[off] = gate[i];
                    up_out[off] = up[i];
                }
                mid_out[off] = (gate[i] / (1.0f + expf(-gate[i]))) * up[i] * weights[pair[i]];
            }
        }
    }
}

__global__ static DS4_ROCM_UNUSED void moe_down_kernel(
        float *down_out,
        const char *down_base,
        const cuda_block_q8_K *midq,
        const int32_t *selected,
        uint64_t down_expert_bytes,
        uint64_t down_row_bytes,
        uint32_t midq_blocks,
        uint32_t out_dim,
        uint32_t n_expert) {
    uint32_t row = blockIdx.x;
    uint32_t pair = blockIdx.y;
    if (row >= out_dim) return;
    uint32_t tok = pair / n_expert;
    uint32_t slot = pair - tok * n_expert;
    int32_t expert_i = selected[(uint64_t)tok * n_expert + slot];
    if (expert_i < 0) expert_i = 0;
    const cuda_block_q2_K *wr = (const cuda_block_q2_K *)(down_base + (uint64_t)(uint32_t)expert_i * down_expert_bytes + (uint64_t)row * down_row_bytes);
    const cuda_block_q8_K *xq = midq + (uint64_t)pair * midq_blocks;
    float acc = 0.0f;
    for (uint32_t b = threadIdx.x; b < midq_blocks; b += blockDim.x) acc += dev_dot_q2_K_q8_K_block(wr + b, xq + b);
    __shared__ float partial[256];
    partial[threadIdx.x] = acc;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) down_out[(uint64_t)pair * out_dim + row] = partial[0];
}

__global__ static DS4_ROCM_UNUSED void moe_down_warp8_kernel(
        float *down_out,
        const char *down_base,
        const cuda_block_q8_K *midq,
        const int32_t *selected,
        uint64_t down_expert_bytes,
        uint64_t down_row_bytes,
        uint32_t midq_blocks,
        uint32_t out_dim,
        uint32_t n_expert) {
    uint32_t lane = threadIdx.x & 31u;
    uint32_t warp = threadIdx.x >> 5u;
    uint32_t row = blockIdx.x * 8u + warp;
    uint32_t pair = blockIdx.y;
    if (row >= out_dim) return;
    uint32_t tok = pair / n_expert;
    uint32_t slot = pair - tok * n_expert;
    int32_t expert_i = selected[(uint64_t)tok * n_expert + slot];
    if (expert_i < 0) expert_i = 0;
    const cuda_block_q2_K *wr = (const cuda_block_q2_K *)(down_base + (uint64_t)(uint32_t)expert_i * down_expert_bytes + (uint64_t)row * down_row_bytes);
    const cuda_block_q8_K *xq = midq + (uint64_t)pair * midq_blocks;
    float acc = 0.0f;
    for (uint32_t b = lane; b < midq_blocks; b += 32u) acc += dev_dot_q2_K_q8_K_block(wr + b, xq + b);
    acc = warp_sum_f32(acc);
    if (lane == 0) down_out[(uint64_t)pair * out_dim + row] = acc;
}

__global__ static DS4_ROCM_UNUSED void moe_down_hwarp16_kernel(
        float *down_out,
        const char *down_base,
        const cuda_block_q8_K *midq,
        const int32_t *selected,
        uint64_t down_expert_bytes,
        uint64_t down_row_bytes,
        uint32_t midq_blocks,
        uint32_t out_dim,
        uint32_t n_expert) {
    uint32_t lane = threadIdx.x & 15u;
    uint32_t row = blockIdx.x * 16u + (threadIdx.x >> 4u);
    uint32_t pair = blockIdx.y;
    if (row >= out_dim) return;
    uint32_t tok = pair / n_expert;
    uint32_t slot = pair - tok * n_expert;
    int32_t expert_i = selected[(uint64_t)tok * n_expert + slot];
    if (expert_i < 0) expert_i = 0;
    const cuda_block_q2_K *wr = (const cuda_block_q2_K *)(down_base + (uint64_t)(uint32_t)expert_i * down_expert_bytes + (uint64_t)row * down_row_bytes);
    const cuda_block_q8_K *xq = midq + (uint64_t)pair * midq_blocks;
    float acc = 0.0f;
    for (uint32_t b = lane; b < midq_blocks; b += 16u) acc += dev_dot_q2_K_q8_K_block(wr + b, xq + b);
    acc = half_warp_sum_f32(acc, lane);
    if (lane == 0) down_out[(uint64_t)pair * out_dim + row] = acc;
}

__global__ static void moe_down_qwarp32_kernel(
        float *down_out,
        const char *down_base,
        const cuda_block_q8_K *midq,
        const int32_t *selected,
        uint64_t down_expert_bytes,
        uint64_t down_row_bytes,
        uint32_t midq_blocks,
        uint32_t out_dim,
        uint32_t n_expert) {
    uint32_t lane = threadIdx.x & 7u;
    uint32_t row = blockIdx.x * 32u + (threadIdx.x >> 3u);
    uint32_t pair = blockIdx.y;
    if (row >= out_dim) return;
    uint32_t tok = pair / n_expert;
    uint32_t slot = pair - tok * n_expert;
    int32_t expert_i = selected[(uint64_t)tok * n_expert + slot];
    if (expert_i < 0) expert_i = 0;
    const cuda_block_q2_K *wr = (const cuda_block_q2_K *)(down_base + (uint64_t)(uint32_t)expert_i * down_expert_bytes + (uint64_t)row * down_row_bytes);
    const cuda_block_q8_K *xq = midq + (uint64_t)pair * midq_blocks;
    float acc = 0.0f;
    for (uint32_t b = lane; b < midq_blocks; b += 8u) acc += dev_dot_q2_K_q8_K_block(wr + b, xq + b);
    acc = quarter_warp_sum_f32(acc, lane);
    if (lane == 0) down_out[(uint64_t)pair * out_dim + row] = acc;
}

__global__ static void moe_gate_up_mid_decode_q4K_qwarp32_kernel(
        float *gate_out,
        float *up_out,
        float *mid_out,
        const char *gate_base,
        const char *up_base,
        const cuda_block_q8_K *xq,
        const int32_t *selected,
        const float *weights,
        uint64_t gate_expert_bytes,
        uint64_t gate_row_bytes,
        uint32_t xq_blocks,
        uint32_t expert_mid_dim,
        uint32_t n_expert,
        uint32_t write_aux,
        float clamp) {
    uint32_t lane = threadIdx.x & 7u;
    uint32_t row_lane = threadIdx.x >> 3u;
    uint32_t pair = blockIdx.y;
    uint32_t tok = pair / n_expert;
    uint32_t slot = pair - tok * n_expert;
    int32_t expert_i = selected[(uint64_t)tok * n_expert + slot];
    if (expert_i < 0) expert_i = 0;
    uint32_t expert = (uint32_t)expert_i;
    const cuda_block_q8_K *xqb = xq + (uint64_t)tok * xq_blocks;
    for (uint32_t rr = 0; rr < 4u; rr++) {
        uint32_t row = blockIdx.x * 128u + row_lane + rr * 32u;
        if (row >= expert_mid_dim) continue;
        const cuda_block_q4_K *gr = (const cuda_block_q4_K *)(gate_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes);
        const cuda_block_q4_K *ur = (const cuda_block_q4_K *)(up_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes);
        float gate = 0.0f;
        float up = 0.0f;
        for (uint32_t b = lane; b < xq_blocks; b += 8u) {
            gate += dev_dot_q4_K_q8_K_block(gr + b, xqb + b);
            up += dev_dot_q4_K_q8_K_block(ur + b, xqb + b);
        }
        gate = quarter_warp_sum_f32(gate, lane);
        up = quarter_warp_sum_f32(up, lane);
        if (lane == 0) {
            if (clamp > 1.0e-6f) {
                if (gate > clamp) gate = clamp;
                if (up > clamp) up = clamp;
                if (up < -clamp) up = -clamp;
            }
            const uint64_t off = (uint64_t)pair * expert_mid_dim + row;
            if (write_aux) {
                gate_out[off] = gate;
                up_out[off] = up;
            }
            mid_out[off] = (gate / (1.0f + expf(-gate))) * up * weights[(uint64_t)tok * n_expert + slot];
        }
    }
}

__global__ static void moe_gate_up_mid_decode_mxfp4_qwarp32_kernel(
        float *gate_out,
        float *up_out,
        float *mid_out,
        const char *gate_base,
        const char *up_base,
        const cuda_block_q8_K *xq,
        const int32_t *selected,
        const float *weights,
        uint64_t gate_expert_bytes,
        uint64_t gate_row_bytes,
        uint32_t xq_blocks,
        uint32_t expert_mid_dim,
        uint32_t n_expert,
        uint32_t write_aux,
        float clamp) {
    constexpr uint32_t rows_per_wave = 1u;
    constexpr uint32_t waves_per_block = 8u;
    constexpr uint32_t rows_per_block = rows_per_wave * waves_per_block;
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t wave = threadIdx.x >> 5u;
    const uint32_t block_lane = lane >> 1u;
    const uint32_t half = lane & 1u;
    const uint32_t first_row = blockIdx.x * rows_per_block + wave * rows_per_wave;
    const uint32_t pair = blockIdx.y;
    const uint32_t tok = pair / n_expert;
    const uint32_t slot = pair - tok * n_expert;
    int32_t expert_i = selected[(uint64_t)tok * n_expert + slot];
    if (expert_i < 0) expert_i = 0;
    const uint32_t expert = (uint32_t)expert_i;
    const cuda_block_q8_K *xqb = xq + (uint64_t)tok * xq_blocks;
    float gate[rows_per_wave] = {0.0f};
    float up[rows_per_wave] = {0.0f};
    const uint32_t mxfp4_blocks = xq_blocks * 8u;

    for (uint32_t mb = block_lane; mb < mxfp4_blocks; mb += 16u) {
        const cuda_block_q8_K *yb = xqb + (mb >> 3u);
        #pragma unroll
        for (uint32_t rr = 0; rr < rows_per_wave; rr++) {
            const uint32_t row = first_row + rr;
            if (row >= expert_mid_dim) continue;
            const cuda_block_mxfp4 *gate_blocks =
                (const cuda_block_mxfp4 *)(gate_base +
                    (uint64_t)expert * gate_expert_bytes +
                    (uint64_t)row * gate_row_bytes);
            const cuda_block_mxfp4 *up_blocks =
                (const cuda_block_mxfp4 *)(up_base +
                    (uint64_t)expert * gate_expert_bytes +
                    (uint64_t)row * gate_row_bytes);
            const uint32_t subblock = mb & 7u;
            gate[rr] += dev_dot_mxfp4_q8_K_half_block(
                gate_blocks + mb, yb, subblock, half);
            up[rr] += dev_dot_mxfp4_q8_K_half_block(
                up_blocks + mb, yb, subblock, half);
        }
    }

    #pragma unroll
    for (uint32_t rr = 0; rr < rows_per_wave; rr++) {
        const uint32_t row = first_row + rr;
        if (row >= expert_mid_dim) continue;
        gate[rr] = warp_sum_f32(gate[rr]);
        up[rr] = warp_sum_f32(up[rr]);
        if (lane == 0u) {
            float g = gate[rr];
            float u = up[rr];
            if (clamp > 1.0e-6f) {
                if (g > clamp) g = clamp;
                if (u > clamp) u = clamp;
                if (u < -clamp) u = -clamp;
            }
            const uint64_t off = (uint64_t)pair * expert_mid_dim + row;
            if (write_aux) {
                gate_out[off] = g;
                up_out[off] = u;
            }
            mid_out[off] = (g / (1.0f + expf(-g))) * u *
                weights[(uint64_t)tok * n_expert + slot];
        }
    }
}

__global__ static void moe_gate_up_mid_q2K_decode_q8_qwarp32_kernel(
        float *gate_out,
        float *up_out,
        float *mid_out,
        const char *gate_base,
        const char *up_base,
        const cuda_block_q8_K *xq,
        const int32_t *selected,
        const float *weights,
        uint64_t gate_expert_bytes,
        uint64_t gate_row_bytes,
        uint32_t xq_blocks,
        uint32_t expert_mid_dim,
        uint32_t n_expert,
        uint32_t write_aux,
        float clamp) {
    const uint32_t lane = threadIdx.x & 15u;
    const uint32_t row_lane = threadIdx.x >> 4u;
    const uint32_t pair = blockIdx.y;
    const uint32_t tok = pair / n_expert;
    const uint32_t slot = pair - tok * n_expert;
    int32_t expert_i = selected[(uint64_t)tok * n_expert + slot];
    if (expert_i < 0) expert_i = 0;
    const uint32_t expert = (uint32_t)expert_i;
    const cuda_block_q8_K *xqb = xq + (uint64_t)tok * xq_blocks;
    __shared__ cuda_block_q8_K sxq[16];
    if (xq_blocks <= 16u) {
        for (uint32_t i = threadIdx.x; i < xq_blocks; i += blockDim.x) sxq[i] = xqb[i];
        __syncthreads();
        xqb = sxq;
    }
    for (uint32_t rr = 0; rr < 16u; rr++) {
        const uint32_t row = blockIdx.x * 256u + row_lane + rr * 16u;
        if (row >= expert_mid_dim) continue;
        const cuda_block_q2_K *gr = (const cuda_block_q2_K *)(gate_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes);
        const cuda_block_q2_K *ur = (const cuda_block_q2_K *)(up_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes);
        float gate = 0.0f;
        float up = 0.0f;
        for (uint32_t b = lane; b < xq_blocks; b += 16u) {
            gate += dev_dot_q2_K_q8_K_block(gr + b, xqb + b);
            up += dev_dot_q2_K_q8_K_block(ur + b, xqb + b);
        }
        gate = half_warp_sum_f32(gate, lane);
        up = half_warp_sum_f32(up, lane);
        if (lane == 0u) {
            if (clamp > 1.0e-6f) {
                if (gate > clamp) gate = clamp;
                if (up > clamp) up = clamp;
                if (up < -clamp) up = -clamp;
            }
            const uint64_t off = (uint64_t)pair * expert_mid_dim + row;
            if (write_aux) {
                gate_out[off] = gate;
                up_out[off] = up;
            }
            mid_out[off] = (gate / (1.0f + expf(-gate))) * up * weights[(uint64_t)tok * n_expert + slot];
        }
    }
}

__global__ static void moe_down_sum6_qwarp32_kernel(
        float *out,
        const char *down_base,
        const cuda_block_q8_K *midq,
        const int32_t *selected,
        uint64_t down_expert_bytes,
        uint64_t down_row_bytes,
        uint32_t midq_blocks,
        uint32_t out_dim,
        uint32_t n_expert) {
    uint32_t lane = threadIdx.x & 7u;
    uint32_t row = blockIdx.x * 32u + (threadIdx.x >> 3u);
    if (row >= out_dim) return;
    float total = 0.0f;
    #pragma unroll
    for (uint32_t slot = 0; slot < DS4_ROCM_N_EXPERT_USED; slot++) {
        if (slot >= n_expert) continue;
        int32_t expert_i = selected[slot];
        if (expert_i < 0) expert_i = 0;
        const cuda_block_q2_K *wr = (const cuda_block_q2_K *)(down_base + (uint64_t)(uint32_t)expert_i * down_expert_bytes + (uint64_t)row * down_row_bytes);
        const cuda_block_q8_K *xq = midq + (uint64_t)slot * midq_blocks;
        float acc = 0.0f;
        for (uint32_t b = lane; b < midq_blocks; b += 8u) acc += dev_dot_q2_K_q8_K_block(wr + b, xq + b);
        acc = quarter_warp_sum_f32(acc, lane);
        if (lane == 0) total += acc;
    }
    if (lane == 0) out[row] = total;
}

__global__ static void moe_down_sum6_qwarp32_ptrs_kernel(
        float *out,
        const char * const *down_slots,
        const cuda_block_q8_K *midq,
        uint64_t down_row_bytes,
        uint32_t midq_blocks,
        uint32_t out_dim,
        uint32_t n_expert) {
    uint32_t lane = threadIdx.x & 7u;
    uint32_t row = blockIdx.x * 32u + (threadIdx.x >> 3u);
    if (row >= out_dim) return;
    float total = 0.0f;
    #pragma unroll
    for (uint32_t slot = 0; slot < DS4_ROCM_N_EXPERT_USED; slot++) {
        if (slot >= n_expert) continue;
        const char *down_base = down_slots[slot];
        if (!down_base) continue;
        const cuda_block_q2_K *wr = (const cuda_block_q2_K *)(down_base + (uint64_t)row * down_row_bytes);
        const cuda_block_q8_K *xq = midq + (uint64_t)slot * midq_blocks;
        float acc = 0.0f;
        for (uint32_t b = lane; b < midq_blocks; b += 8u) acc += dev_dot_q2_K_q8_K_block(wr + b, xq + b);
        acc = quarter_warp_sum_f32(acc, lane);
        if (lane == 0) total += acc;
    }
    if (lane == 0) out[row] = total;
}

__global__ static void moe_down_sum6_qwarp32_ptrs_batch_kernel(
        float *out,
        const char * const *down_slots,
        const cuda_block_q8_K *midq,
        const int32_t *selected,
        uint64_t down_row_bytes,
        uint32_t midq_blocks,
        uint32_t out_dim,
        uint32_t n_expert,
        uint32_t n_tokens) {
    uint32_t lane = threadIdx.x & 7u;
    uint32_t row = blockIdx.x * 32u + (threadIdx.x >> 3u);
    uint32_t tok = blockIdx.y;
    if (row >= out_dim || tok >= n_tokens) return;
    float total = 0.0f;
    #pragma unroll
    for (uint32_t slot = 0; slot < DS4_ROCM_N_EXPERT_USED; slot++) {
        if (slot >= n_expert) continue;
        int32_t compact_i = selected[(uint64_t)tok * n_expert + slot];
        if (compact_i < 0) compact_i = 0;
        const char *down_base = down_slots[(uint32_t)compact_i];
        if (!down_base) continue;
        const cuda_block_q2_K *wr = (const cuda_block_q2_K *)(down_base + (uint64_t)row * down_row_bytes);
        const cuda_block_q8_K *xq = midq + ((uint64_t)tok * n_expert + slot) * midq_blocks;
        float acc = 0.0f;
        for (uint32_t b = lane; b < midq_blocks; b += 8u) acc += dev_dot_q2_K_q8_K_block(wr + b, xq + b);
        acc = quarter_warp_sum_f32(acc, lane);
        if (lane == 0) total += acc;
    }
    if (lane == 0) out[(uint64_t)tok * out_dim + row] = total;
}

/* GLM's routed-IQ2 layout uses IQ2_XXS for the down projection as well as
 * gate/up.  One 8-lane subgroup owns an output row and accumulates all routed
 * experts for a token.  This mirrors the Q2_K direct-sum path above while
 * using the existing IQ2_XXS x Q8_K dot primitive. */
__global__ static void moe_down_iq2_sum_qwarp32_batch_kernel(
        float *out,
        const char *down_base,
        const cuda_block_q8_K *midq,
        const int32_t *selected,
        uint64_t down_expert_bytes,
        uint64_t down_row_bytes,
        uint32_t midq_blocks,
        uint32_t out_dim,
        uint32_t n_expert,
        uint32_t n_tokens) {
    const uint32_t lane = threadIdx.x & 7u;
    const uint32_t row = blockIdx.x * 32u + (threadIdx.x >> 3u);
    const uint32_t tok = blockIdx.y;
    if (row >= out_dim || tok >= n_tokens) return;
    float total = 0.0f;
#pragma unroll
    for (uint32_t slot = 0; slot < DS4_ROCM_N_EXPERT_USED; slot++) {
        if (slot >= n_expert) continue;
        int32_t expert_i = selected[(uint64_t)tok * n_expert + slot];
        if (expert_i < 0) expert_i = 0;
        const cuda_block_iq2_xxs *wr =
            (const cuda_block_iq2_xxs *)(down_base +
                (uint64_t)(uint32_t)expert_i * down_expert_bytes +
                (uint64_t)row * down_row_bytes);
        const cuda_block_q8_K *xq =
            midq + ((uint64_t)tok * n_expert + slot) * midq_blocks;
        float acc = 0.0f;
        for (uint32_t b = lane; b < midq_blocks; b += 8u) {
            acc += dev_dot_iq2_xxs_q8_K_block(wr + b, xq + b);
        }
        acc = quarter_warp_sum_f32(acc, lane);
        if (lane == 0u) total += acc;
    }
    if (lane == 0u) out[(uint64_t)tok * out_dim + row] = total;
}

__global__ static void moe_down_iq2_sum_qwarp32_ptrs_batch_kernel(
        float *out,
        const char * const *down_slots,
        const cuda_block_q8_K *midq,
        const int32_t *selected,
        uint64_t down_row_bytes,
        uint32_t midq_blocks,
        uint32_t out_dim,
        uint32_t n_expert,
        uint32_t n_tokens) {
    const uint32_t lane = threadIdx.x & 7u;
    const uint32_t row = blockIdx.x * 32u + (threadIdx.x >> 3u);
    const uint32_t tok = blockIdx.y;
    if (row >= out_dim || tok >= n_tokens) return;
    float total = 0.0f;
#pragma unroll
    for (uint32_t slot = 0; slot < DS4_ROCM_N_EXPERT_USED; slot++) {
        if (slot >= n_expert) continue;
        int32_t compact_i = selected[(uint64_t)tok * n_expert + slot];
        if (compact_i < 0) compact_i = 0;
        const char *down = down_slots[(uint32_t)compact_i];
        if (!down) continue;
        const cuda_block_iq2_xxs *wr =
            (const cuda_block_iq2_xxs *)(down +
                (uint64_t)row * down_row_bytes);
        const cuda_block_q8_K *xq =
            midq + ((uint64_t)tok * n_expert + slot) * midq_blocks;
        float acc = 0.0f;
        for (uint32_t b = lane; b < midq_blocks; b += 8u) {
            acc += dev_dot_iq2_xxs_q8_K_block(wr + b, xq + b);
        }
        acc = quarter_warp_sum_f32(acc, lane);
        if (lane == 0u) total += acc;
    }
    if (lane == 0u) out[(uint64_t)tok * out_dim + row] = total;
}

__global__ static void moe_down_q4K_sum6_qwarp32_kernel(
        float *out,
        const char *down_base,
        const cuda_block_q8_K *midq,
        const int32_t *selected,
        uint64_t down_expert_bytes,
        uint64_t down_row_bytes,
        uint32_t midq_blocks,
        uint32_t out_dim,
        uint32_t n_expert) {
    uint32_t lane = threadIdx.x & 7u;
    uint32_t row = blockIdx.x * 32u + (threadIdx.x >> 3u);
    if (row >= out_dim) return;
    float total = 0.0f;
    #pragma unroll
    for (uint32_t slot = 0; slot < DS4_ROCM_N_EXPERT_USED; slot++) {
        if (slot >= n_expert) continue;
        int32_t expert_i = selected[slot];
        if (expert_i < 0) expert_i = 0;
        const cuda_block_q4_K *wr = (const cuda_block_q4_K *)(down_base + (uint64_t)(uint32_t)expert_i * down_expert_bytes + (uint64_t)row * down_row_bytes);
        const cuda_block_q8_K *xq = midq + (uint64_t)slot * midq_blocks;
        float acc = 0.0f;
        for (uint32_t b = lane; b < midq_blocks; b += 8u) acc += dev_dot_q4_K_q8_K_block(wr + b, xq + b);
        acc = quarter_warp_sum_f32(acc, lane);
        if (lane == 0) total += acc;
    }
    if (lane == 0) out[row] = total;
}

template <bool Batch>
__global__ static void moe_down_mxfp4_sum6_qwarp32_kernel(
        float *out,
        const char *down_base,
        const cuda_block_q8_K *midq,
        const int32_t *selected,
        uint64_t down_expert_bytes,
        uint64_t down_row_bytes,
        uint32_t midq_blocks,
        uint32_t out_dim,
        uint32_t n_expert) {
    constexpr uint32_t rows_per_wave = 1u;
    constexpr uint32_t waves_per_block = 8u;
    constexpr uint32_t rows_per_block = rows_per_wave * waves_per_block;
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t wave = threadIdx.x >> 5u;
    const uint32_t block_lane = lane >> 1u;
    const uint32_t half = lane & 1u;
    const uint32_t first_row = blockIdx.x * rows_per_block + wave * rows_per_wave;
    const uint32_t tok = Batch ? blockIdx.y : 0u;
    const int32_t *token_selected = selected + (uint64_t)tok * n_expert;
    const cuda_block_q8_K *token_midq =
        midq + (uint64_t)tok * n_expert * midq_blocks;
    float *token_out = out + (uint64_t)tok * out_dim;
    float total[rows_per_wave] = {0.0f};
    const uint32_t mxfp4_blocks = midq_blocks * 8u;

    #pragma unroll
    for (uint32_t slot = 0; slot < DS4_ROCM_N_EXPERT_USED; slot++) {
        if (slot >= n_expert) continue;
        int32_t expert_i = token_selected[slot];
        if (expert_i < 0) expert_i = 0;
        const cuda_block_q8_K *xq = token_midq + (uint64_t)slot * midq_blocks;
        for (uint32_t mb = block_lane; mb < mxfp4_blocks; mb += 16u) {
            const cuda_block_q8_K *yb = xq + (mb >> 3u);
            #pragma unroll
            for (uint32_t rr = 0; rr < rows_per_wave; rr++) {
                const uint32_t row = first_row + rr;
                if (row >= out_dim) continue;
                const cuda_block_mxfp4 *down_blocks =
                    (const cuda_block_mxfp4 *)(down_base +
                        (uint64_t)(uint32_t)expert_i * down_expert_bytes +
                        (uint64_t)row * down_row_bytes);
                total[rr] += dev_dot_mxfp4_q8_K_half_block(
                    down_blocks + mb, yb, mb & 7u, half);
            }
        }
    }

    #pragma unroll
    for (uint32_t rr = 0; rr < rows_per_wave; rr++) {
        const uint32_t row = first_row + rr;
        if (row >= out_dim) continue;
        total[rr] = warp_sum_f32(total[rr]);
        if (lane == 0u) token_out[row] = total[rr];
    }
}

__global__ static void moe_down_q4K_qwarp32_kernel(
        float *down_out,
        const char *down_base,
        const cuda_block_q8_K *midq,
        const int32_t *selected,
        uint64_t down_expert_bytes,
        uint64_t down_row_bytes,
        uint32_t midq_blocks,
        uint32_t out_dim,
        uint32_t n_expert) {
    uint32_t lane = threadIdx.x & 7u;
    uint32_t row = blockIdx.x * 32u + (threadIdx.x >> 3u);
    uint32_t pair = blockIdx.y;
    if (row >= out_dim) return;
    uint32_t tok = pair / n_expert;
    uint32_t slot = pair - tok * n_expert;
    int32_t expert_i = selected[(uint64_t)tok * n_expert + slot];
    if (expert_i < 0) expert_i = 0;
    const cuda_block_q4_K *wr = (const cuda_block_q4_K *)(down_base + (uint64_t)(uint32_t)expert_i * down_expert_bytes + (uint64_t)row * down_row_bytes);
    const cuda_block_q8_K *xq = midq + (uint64_t)pair * midq_blocks;
    float acc = 0.0f;
    for (uint32_t b = lane; b < midq_blocks; b += 8u) acc += dev_dot_q4_K_q8_K_block(wr + b, xq + b);
    acc = quarter_warp_sum_f32(acc, lane);
    if (lane == 0) down_out[(uint64_t)pair * out_dim + row] = acc;
}

__global__ static void moe_down_q4K_sorted_qwarp32_kernel(
        float *down_out,
        const char *down_base,
        const cuda_block_q8_K *midq,
        const uint32_t *sorted_pairs,
        const int32_t *selected,
        uint64_t down_expert_bytes,
        uint64_t down_row_bytes,
        uint32_t midq_blocks,
        uint32_t out_dim,
        uint32_t n_expert) {
    uint32_t lane = threadIdx.x & 7u;
    uint32_t row = blockIdx.x * 32u + (threadIdx.x >> 3u);
    uint32_t pair = sorted_pairs[blockIdx.y];
    if (row >= out_dim) return;
    uint32_t tok = pair / n_expert;
    uint32_t slot = pair - tok * n_expert;
    int32_t expert_i = selected[(uint64_t)tok * n_expert + slot];
    if (expert_i < 0) expert_i = 0;
    const cuda_block_q4_K *wr = (const cuda_block_q4_K *)(down_base + (uint64_t)(uint32_t)expert_i * down_expert_bytes + (uint64_t)row * down_row_bytes);
    const cuda_block_q8_K *xq = midq + (uint64_t)pair * midq_blocks;
    float acc = 0.0f;
    for (uint32_t b = lane; b < midq_blocks; b += 8u) acc += dev_dot_q4_K_q8_K_block(wr + b, xq + b);
    acc = quarter_warp_sum_f32(acc, lane);
    if (lane == 0) down_out[(uint64_t)pair * out_dim + row] = acc;
}

__global__ static void moe_down_q4K_expert_tile4_row32_kernel(
        float *down_out,
        const char *down_base,
        const cuda_block_q8_K *midq,
        const uint32_t *sorted_pairs,
        const uint32_t *offsets,
        const uint32_t *counts,
        const uint32_t *tile_total,
        const uint32_t *tile_experts,
        const uint32_t *tile_starts,
        uint64_t down_expert_bytes,
        uint64_t down_row_bytes,
        uint32_t midq_blocks,
        uint32_t out_dim,
        uint32_t n_expert,
        uint32_t atomic_out) {
    uint32_t tile = blockIdx.y;
    if (tile >= *tile_total) return;
    uint32_t lane = threadIdx.x & 7u;
    uint32_t row = blockIdx.x * 32u + (threadIdx.x >> 3u);
    uint32_t expert = tile_experts[tile];
    uint32_t local_start = tile_starts[tile];
    __shared__ cuda_block_q8_K sxq[4][8];
    uint32_t pair[4] = {0, 0, 0, 0};
    const cuda_block_q8_K *xqb[4] = {NULL, NULL, NULL, NULL};
    uint32_t np = 0;
    for (; np < 4u; np++) {
        uint32_t local_pair = local_start + np;
        if (local_pair >= counts[expert]) break;
        pair[np] = sorted_pairs[offsets[expert] + local_pair];
        xqb[np] = midq + (uint64_t)pair[np] * midq_blocks;
    }
    if (midq_blocks <= 8u) {
        for (uint32_t i = threadIdx.x; i < np * midq_blocks; i += blockDim.x) {
            uint32_t p = i / midq_blocks;
            uint32_t b = i - p * midq_blocks;
            sxq[p][b] = xqb[p][b];
        }
        __syncthreads();
        for (uint32_t p = 0; p < np; p++) xqb[p] = sxq[p];
    }
    if (row >= out_dim) return;
    const cuda_block_q4_K *wr = (const cuda_block_q4_K *)(down_base + (uint64_t)expert * down_expert_bytes + (uint64_t)row * down_row_bytes);
    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (uint32_t b = lane; b < midq_blocks; b += 8u) {
        dev_dot_q4_K_q8_K_block4(wr + b, xqb[0] ? xqb[0] + b : NULL, xqb[1] ? xqb[1] + b : NULL,
                                 xqb[2] ? xqb[2] + b : NULL, xqb[3] ? xqb[3] + b : NULL, np, acc);
    }
    for (uint32_t p = 0; p < np; p++) {
        acc[p] = quarter_warp_sum_f32(acc[p], lane);
        if (lane == 0) {
            if (atomic_out) {
                uint32_t tok = pair[p] / n_expert;
                atomicAdd(down_out + (uint64_t)tok * out_dim + row, acc[p]);
            } else {
                down_out[(uint64_t)pair[p] * out_dim + row] = acc[p];
            }
        }
    }
}

__global__ static void moe_down_q4K_expert_tile8_row32_kernel(
        float *down_out,
        const char *down_base,
        const cuda_block_q8_K *midq,
        const uint32_t *sorted_pairs,
        const uint32_t *offsets,
        const uint32_t *counts,
        const uint32_t *tile_total,
        const uint32_t *tile_experts,
        const uint32_t *tile_starts,
        uint64_t down_expert_bytes,
        uint64_t down_row_bytes,
        uint32_t midq_blocks,
        uint32_t out_dim,
        uint32_t n_expert,
        uint32_t atomic_out) {
    uint32_t tile = blockIdx.y;
    if (tile >= *tile_total) return;
    uint32_t lane = threadIdx.x & 7u;
    uint32_t row = blockIdx.x * 32u + (threadIdx.x >> 3u);
    uint32_t expert = tile_experts[tile];
    uint32_t local_start = tile_starts[tile];
    __shared__ cuda_block_q8_K sxq[8][8];
    uint32_t pair[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    const cuda_block_q8_K *xqb[8] = {NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL};
    uint32_t np = 0;
    for (; np < 8u; np++) {
        uint32_t local_pair = local_start + np;
        if (local_pair >= counts[expert]) break;
        pair[np] = sorted_pairs[offsets[expert] + local_pair];
        xqb[np] = midq + (uint64_t)pair[np] * midq_blocks;
    }
    if (midq_blocks <= 8u) {
        for (uint32_t i = threadIdx.x; i < np * midq_blocks; i += blockDim.x) {
            uint32_t p = i / midq_blocks;
            uint32_t b = i - p * midq_blocks;
            sxq[p][b] = xqb[p][b];
        }
        __syncthreads();
        for (uint32_t p = 0; p < np; p++) xqb[p] = sxq[p];
    }
    if (row >= out_dim) return;
    const cuda_block_q4_K *wr = (const cuda_block_q4_K *)(down_base + (uint64_t)expert * down_expert_bytes + (uint64_t)row * down_row_bytes);
    float acc[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    for (uint32_t b = lane; b < midq_blocks; b += 8u) {
        dev_dot_q4_K_q8_K_block8(wr + b, xqb[0] ? xqb[0] + b : NULL, xqb[1] ? xqb[1] + b : NULL,
                                 xqb[2] ? xqb[2] + b : NULL, xqb[3] ? xqb[3] + b : NULL,
                                 xqb[4] ? xqb[4] + b : NULL, xqb[5] ? xqb[5] + b : NULL,
                                 xqb[6] ? xqb[6] + b : NULL, xqb[7] ? xqb[7] + b : NULL, np, acc);
    }
    for (uint32_t p = 0; p < np; p++) {
        acc[p] = quarter_warp_sum_f32(acc[p], lane);
        if (lane == 0) {
            if (atomic_out) {
                uint32_t tok = pair[p] / n_expert;
                atomicAdd(down_out + (uint64_t)tok * out_dim + row, acc[p]);
            } else {
                down_out[(uint64_t)pair[p] * out_dim + row] = acc[p];
            }
        }
    }
}

__global__ static void moe_down_mxfp4_expert_tile8_row32_kernel(
        float *down_out,
        const char *down_base,
        const cuda_block_q8_K *midq,
        const uint32_t *sorted_pairs,
        const uint32_t *offsets,
        const uint32_t *counts,
        const uint32_t *tile_total,
        const uint32_t *tile_experts,
        const uint32_t *tile_starts,
        uint64_t down_expert_bytes,
        uint64_t down_row_bytes,
        uint32_t midq_blocks,
        uint32_t out_dim,
        uint32_t n_expert,
        uint32_t atomic_out,
        uint32_t row_groups) {
    uint32_t tile = blockIdx.y;
    if (tile >= *tile_total) return;
    uint32_t lane = threadIdx.x & 7u;
    uint32_t expert = tile_experts[tile];
    uint32_t local_start = tile_starts[tile];
    /* Aligned quant slices plus separate scales, as in the gate/up tile
     * kernel, so the dot helper loads b128 from LDS. */
    __shared__ int4 sxq4[8u * 8u * 16u];
    __shared__ float sds[8u * 8u];
    int8_t *sqs = (int8_t *)sxq4;
    uint32_t pair[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    /* Fixed-count predicated setup keeps pair/xqb in registers; the tail
     * xqb entries stay at valid dummy rows and are never read past np. */
    const cuda_block_q8_K *xqb[8] = { midq, midq, midq, midq, midq, midq, midq, midq };
    uint32_t count = counts[expert];
    uint32_t np = count - local_start;
    if (np > 8u) np = 8u;
    #pragma unroll
    for (uint32_t p = 0; p < 8u; p++) {
        if (p < np) {
            pair[p] = sorted_pairs[offsets[expert] + local_start + p];
            xqb[p] = midq + (uint64_t)pair[p] * midq_blocks;
        }
    }
    const int staged = midq_blocks <= 8u;
    if (staged) {
        for (uint32_t i = threadIdx.x; i < np * midq_blocks * 16u; i += blockDim.x) {
            const uint32_t u = i / (midq_blocks * 16u);
            const uint32_t rem = i - u * (midq_blocks * 16u);
            const uint32_t b = rem >> 4u;
            const uint32_t k = rem & 15u;
            const uint32_t sp = sorted_pairs[offsets[expert] + local_start + u];
            const int32_t *src = (const int32_t *)
                midq[(uint64_t)sp * midq_blocks + b].qs + k * 4u;
            ((int4 *)sqs)[(u * midq_blocks + b) * 16u + k] =
                make_int4(src[0], src[1], src[2], src[3]);
        }
        for (uint32_t i = threadIdx.x; i < np * midq_blocks; i += blockDim.x) {
            const uint32_t u = i / midq_blocks;
            const uint32_t b = i - u * midq_blocks;
            const uint32_t sp = sorted_pairs[offsets[expert] + local_start + u];
            sds[i] = midq[(uint64_t)sp * midq_blocks + b].d;
        }
        __syncthreads();
    }
    /* Each block covers row_groups consecutive 32-row groups against one
     * staged copy of the tile's activations: activation staging traffic
     * and block-launch count both divide by row_groups.  Per (token,row)
     * the arithmetic is untouched (same helper, same lane-strided block
     * loop, same reduction), so results are bit-identical to
     * row_groups=1. */
    for (uint32_t rg = 0u; rg < row_groups; rg++) {
    uint32_t row = (blockIdx.x * row_groups + rg) * 32u + (threadIdx.x >> 3u);
    if (row >= out_dim) continue;
    const char *down_row = down_base + (uint64_t)expert * down_expert_bytes + (uint64_t)row * down_row_bytes;
    const uint64_t down_chunk_bytes = down_row_bytes / midq_blocks;
    float acc[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    const uint64_t sq = (uint64_t)midq_blocks << 8u;
    for (uint32_t b = lane; b < midq_blocks; b += 8u) {
        const cuda_block_mxfp4 *wb = (const cuda_block_mxfp4 *)(down_row + (uint64_t)b * down_chunk_bytes);
        if (staged) {
            const int8_t *qb = sqs + ((uint64_t)b << 8u);
            dev_dot_mxfp4_q8a_block8(wb,
                qb, qb + sq, qb + 2u * sq, qb + 3u * sq,
                qb + 4u * sq, qb + 5u * sq, qb + 6u * sq, qb + 7u * sq,
                sds[b], sds[midq_blocks + b], sds[2u * midq_blocks + b], sds[3u * midq_blocks + b],
                sds[4u * midq_blocks + b], sds[5u * midq_blocks + b], sds[6u * midq_blocks + b], sds[7u * midq_blocks + b],
                np, acc);
        } else {
            dev_dot_mxfp4_q8_K_block8(wb, xqb[0] + b, xqb[1] + b, xqb[2] + b, xqb[3] + b,
                                      xqb[4] + b, xqb[5] + b, xqb[6] + b, xqb[7] + b, np, acc);
        }
    }
    #pragma unroll
    for (uint32_t p = 0; p < 8u; p++) {
        if (p >= np) continue;
        acc[p] = quarter_warp_sum_f32(acc[p], lane);
        if (lane == 0) {
            down_out[(uint64_t)pair[p] * out_dim + row] = acc[p];
        }
    }
    }
}

__global__ static void moe_down_sorted_qwarp32_kernel(
        float *down_out,
        const char *down_base,
        const cuda_block_q8_K *midq,
        const uint32_t *sorted_pairs,
        const int32_t *selected,
        uint64_t down_expert_bytes,
        uint64_t down_row_bytes,
        uint32_t midq_blocks,
        uint32_t out_dim,
        uint32_t n_expert) {
    uint32_t lane = threadIdx.x & 7u;
    uint32_t row = blockIdx.x * 32u + (threadIdx.x >> 3u);
    uint32_t pair = sorted_pairs[blockIdx.y];
    if (row >= out_dim) return;
    uint32_t tok = pair / n_expert;
    uint32_t slot = pair - tok * n_expert;
    int32_t expert_i = selected[(uint64_t)tok * n_expert + slot];
    if (expert_i < 0) expert_i = 0;
    const cuda_block_q2_K *wr = (const cuda_block_q2_K *)(down_base + (uint64_t)(uint32_t)expert_i * down_expert_bytes + (uint64_t)row * down_row_bytes);
    const cuda_block_q8_K *xq = midq + (uint64_t)pair * midq_blocks;
    float acc = 0.0f;
    for (uint32_t b = lane; b < midq_blocks; b += 8u) acc += dev_dot_q2_K_q8_K_block(wr + b, xq + b);
    acc = quarter_warp_sum_f32(acc, lane);
    if (lane == 0) down_out[(uint64_t)pair * out_dim + row] = acc;
}

__global__ static DS4_ROCM_UNUSED void moe_down_expert_tile8_kernel(
        float *down_out,
        const char *down_base,
        const cuda_block_q8_K *midq,
        const uint32_t *sorted_pairs,
        const uint32_t *offsets,
        const uint32_t *counts,
        const uint32_t *tile_total,
        const uint32_t *tile_experts,
        const uint32_t *tile_starts,
        uint64_t down_expert_bytes,
        uint64_t down_row_bytes,
        uint32_t midq_blocks,
        uint32_t out_dim,
        uint32_t n_expert) {
    uint32_t tile = blockIdx.y;
    if (tile >= *tile_total) return;
    uint32_t group = threadIdx.x >> 3u;
    uint32_t lane = threadIdx.x & 7u;
    uint32_t pair_slot = group & 7u;
    uint32_t row_lane = group >> 3u;
    uint32_t expert = tile_experts[tile];
    uint32_t local_pair = tile_starts[tile] + pair_slot;
    if (local_pair >= counts[expert]) return;
    uint32_t sorted_idx = offsets[expert] + local_pair;
    uint32_t pair = sorted_pairs[sorted_idx];
    const cuda_block_q8_K *xq = midq + (uint64_t)pair * midq_blocks;

    for (uint32_t rr = 0; rr < 2u; rr++) {
        uint32_t row = blockIdx.x * 8u + row_lane + rr * 4u;
        if (row >= out_dim) continue;
        const cuda_block_q2_K *wr = (const cuda_block_q2_K *)(down_base + (uint64_t)expert * down_expert_bytes + (uint64_t)row * down_row_bytes);
        float acc = 0.0f;
        for (uint32_t b = lane; b < midq_blocks; b += 8u) acc += dev_dot_q2_K_q8_K_block(wr + b, xq + b);
        acc = quarter_warp_sum_f32(acc, lane);
        if (lane == 0) down_out[(uint64_t)pair * out_dim + row] = acc;
    }
}

__global__ static void moe_down_expert_tile4_row32_kernel(
        float *down_out,
        const char *down_base,
        const cuda_block_q8_K *midq,
        const uint32_t *sorted_pairs,
        const uint32_t *offsets,
        const uint32_t *counts,
        const uint32_t *tile_total,
        const uint32_t *tile_experts,
        const uint32_t *tile_starts,
        uint64_t down_expert_bytes,
        uint64_t down_row_bytes,
        uint32_t midq_blocks,
        uint32_t out_dim,
        uint32_t n_expert,
        uint32_t atomic_out) {
    uint32_t tile = blockIdx.y;
    if (tile >= *tile_total) return;
    uint32_t lane = threadIdx.x & 7u;
    uint32_t row = blockIdx.x * 32u + (threadIdx.x >> 3u);
    uint32_t expert = tile_experts[tile];
    uint32_t local_start = tile_starts[tile];
    __shared__ cuda_block_q8_K sxq[4][8];
    uint32_t pair[4] = {0, 0, 0, 0};
    const cuda_block_q8_K *xqb[4] = {NULL, NULL, NULL, NULL};
    uint32_t np = 0;
    for (; np < 4u; np++) {
        uint32_t local_pair = local_start + np;
        if (local_pair >= counts[expert]) break;
        pair[np] = sorted_pairs[offsets[expert] + local_pair];
        xqb[np] = midq + (uint64_t)pair[np] * midq_blocks;
    }
    if (midq_blocks <= 8u) {
        for (uint32_t i = threadIdx.x; i < np * midq_blocks; i += blockDim.x) {
            uint32_t p = i / midq_blocks;
            uint32_t b = i - p * midq_blocks;
            sxq[p][b] = xqb[p][b];
        }
        __syncthreads();
        for (uint32_t p = 0; p < np; p++) xqb[p] = sxq[p];
    }
    if (row >= out_dim) return;
    const cuda_block_q2_K *wr = (const cuda_block_q2_K *)(down_base + (uint64_t)expert * down_expert_bytes + (uint64_t)row * down_row_bytes);
    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (uint32_t b = lane; b < midq_blocks; b += 8u) {
        dev_dot_q2_K_q8_K_block4(wr + b, xqb[0] ? xqb[0] + b : NULL, xqb[1] ? xqb[1] + b : NULL,
                                 xqb[2] ? xqb[2] + b : NULL, xqb[3] ? xqb[3] + b : NULL, np, acc);
    }
    for (uint32_t p = 0; p < np; p++) {
        acc[p] = quarter_warp_sum_f32(acc[p], lane);
        if (lane == 0) {
            if (atomic_out) {
                uint32_t tok = pair[p] / n_expert;
                atomicAdd(down_out + (uint64_t)tok * out_dim + row, acc[p]);
            } else {
                down_out[(uint64_t)pair[p] * out_dim + row] = acc[p];
            }
        }
    }
}

__global__ static void moe_down_expert_tile8_row32_kernel(
        float *down_out,
        const char *down_base,
        const cuda_block_q8_K *midq,
        const uint32_t *sorted_pairs,
        const uint32_t *offsets,
        const uint32_t *counts,
        const uint32_t *tile_total,
        const uint32_t *tile_experts,
        const uint32_t *tile_starts,
        uint64_t down_expert_bytes,
        uint64_t down_row_bytes,
        uint32_t midq_blocks,
        uint32_t out_dim,
        uint32_t n_expert,
        uint32_t atomic_out) {
    uint32_t tile = blockIdx.y;
    if (tile >= *tile_total) return;
    uint32_t lane = threadIdx.x & 7u;
    uint32_t row = blockIdx.x * 32u + (threadIdx.x >> 3u);
    uint32_t expert = tile_experts[tile];
    uint32_t local_start = tile_starts[tile];
    __shared__ cuda_block_q8_K sxq[8][8];
    uint32_t pair[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    const cuda_block_q8_K *xqb[8] = {NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL};
    uint32_t np = 0;
    for (; np < 8u; np++) {
        uint32_t local_pair = local_start + np;
        if (local_pair >= counts[expert]) break;
        pair[np] = sorted_pairs[offsets[expert] + local_pair];
        xqb[np] = midq + (uint64_t)pair[np] * midq_blocks;
    }
    if (midq_blocks <= 8u) {
        for (uint32_t i = threadIdx.x; i < np * midq_blocks; i += blockDim.x) {
            uint32_t p = i / midq_blocks;
            uint32_t b = i - p * midq_blocks;
            sxq[p][b] = xqb[p][b];
        }
        __syncthreads();
        for (uint32_t p = 0; p < np; p++) xqb[p] = sxq[p];
    }
    if (row >= out_dim) return;
    const cuda_block_q2_K *wr = (const cuda_block_q2_K *)(down_base + (uint64_t)expert * down_expert_bytes + (uint64_t)row * down_row_bytes);
    float acc[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    for (uint32_t b = lane; b < midq_blocks; b += 8u) {
        dev_dot_q2_K_q8_K_block8(wr + b, xqb[0] ? xqb[0] + b : NULL, xqb[1] ? xqb[1] + b : NULL,
                                 xqb[2] ? xqb[2] + b : NULL, xqb[3] ? xqb[3] + b : NULL,
                                 xqb[4] ? xqb[4] + b : NULL, xqb[5] ? xqb[5] + b : NULL,
                                 xqb[6] ? xqb[6] + b : NULL, xqb[7] ? xqb[7] + b : NULL, np, acc);
    }
    for (uint32_t p = 0; p < np; p++) {
        acc[p] = quarter_warp_sum_f32(acc[p], lane);
        if (lane == 0) {
            if (atomic_out) {
                uint32_t tok = pair[p] / n_expert;
                atomicAdd(down_out + (uint64_t)tok * out_dim + row, acc[p]);
            } else {
                down_out[(uint64_t)pair[p] * out_dim + row] = acc[p];
            }
        }
    }
}

__global__ static void moe_down_expert_tile16_row32_kernel(
        float *down_out,
        const char *down_base,
        const cuda_block_q8_K *midq,
        const uint32_t *sorted_pairs,
        const uint32_t *offsets,
        const uint32_t *counts,
        const uint32_t *tile_total,
        const uint32_t *tile_experts,
        const uint32_t *tile_starts,
        uint64_t down_expert_bytes,
        uint64_t down_row_bytes,
        uint32_t midq_blocks,
        uint32_t out_dim,
        uint32_t n_expert,
        uint32_t atomic_out) {
    uint32_t tile = blockIdx.y;
    if (tile >= *tile_total) return;
    uint32_t local_start = tile_starts[tile];
    if (local_start & 8u) return;
    uint32_t lane = threadIdx.x & 7u;
    uint32_t row = blockIdx.x * 32u + (threadIdx.x >> 3u);
    uint32_t expert = tile_experts[tile];
    __shared__ cuda_block_q8_K sxq[16][8];
    uint32_t pair[16] = {0};
    const cuda_block_q8_K *xqb[16] = {NULL};
    uint32_t np = 0;
    for (; np < 16u; np++) {
        uint32_t local_pair = local_start + np;
        if (local_pair >= counts[expert]) break;
        pair[np] = sorted_pairs[offsets[expert] + local_pair];
        xqb[np] = midq + (uint64_t)pair[np] * midq_blocks;
    }
    if (midq_blocks <= 8u) {
        for (uint32_t i = threadIdx.x; i < np * midq_blocks; i += blockDim.x) {
            uint32_t p = i / midq_blocks;
            uint32_t b = i - p * midq_blocks;
            sxq[p][b] = xqb[p][b];
        }
        __syncthreads();
        for (uint32_t p = 0; p < np; p++) xqb[p] = sxq[p];
    }
    if (row >= out_dim) return;
    const cuda_block_q2_K *wr = (const cuda_block_q2_K *)(down_base + (uint64_t)expert * down_expert_bytes + (uint64_t)row * down_row_bytes);
    float acc[16] = {0.0f};
    for (uint32_t b = lane; b < midq_blocks; b += 8u) {
        dev_dot_q2_K_q8_K_block8(wr + b, xqb[0] ? xqb[0] + b : NULL, xqb[1] ? xqb[1] + b : NULL,
                                 xqb[2] ? xqb[2] + b : NULL, xqb[3] ? xqb[3] + b : NULL,
                                 xqb[4] ? xqb[4] + b : NULL, xqb[5] ? xqb[5] + b : NULL,
                                 xqb[6] ? xqb[6] + b : NULL, xqb[7] ? xqb[7] + b : NULL, np < 8u ? np : 8u, acc);
        if (np > 8u) {
            dev_dot_q2_K_q8_K_block8(wr + b, xqb[8] ? xqb[8] + b : NULL, xqb[9] ? xqb[9] + b : NULL,
                                     xqb[10] ? xqb[10] + b : NULL, xqb[11] ? xqb[11] + b : NULL,
                                     xqb[12] ? xqb[12] + b : NULL, xqb[13] ? xqb[13] + b : NULL,
                                     xqb[14] ? xqb[14] + b : NULL, xqb[15] ? xqb[15] + b : NULL, np - 8u, acc + 8);
        }
    }
    for (uint32_t p = 0; p < np; p++) {
        acc[p] = quarter_warp_sum_f32(acc[p], lane);
        if (lane == 0) {
            if (atomic_out) {
                uint32_t tok = pair[p] / n_expert;
                atomicAdd(down_out + (uint64_t)tok * out_dim + row, acc[p]);
            } else {
                down_out[(uint64_t)pair[p] * out_dim + row] = acc[p];
            }
        }
    }
}

__global__ static void moe_down_expert_tile16_row2048_kernel(
        float *down_out,
        const char *down_base,
        const cuda_block_q8_K *midq,
        const uint32_t *sorted_pairs,
        const uint32_t *offsets,
        const uint32_t *counts,
        const uint32_t *tile_total,
        const uint32_t *tile_experts,
        const uint32_t *tile_starts,
        uint64_t down_expert_bytes,
        uint64_t down_row_bytes,
        uint32_t midq_blocks,
        uint32_t out_dim,
        uint32_t n_expert,
        uint32_t atomic_out) {
    uint32_t tile = blockIdx.y;
    if (tile >= *tile_total) return;
    uint32_t local_start = tile_starts[tile];
    if (local_start & 8u) return;
    uint32_t lane = threadIdx.x & 7u;
    uint32_t row_lane = threadIdx.x >> 3u;
    uint32_t expert = tile_experts[tile];
    __shared__ cuda_block_q8_K sxq[16][8];
    uint32_t pair[16] = {0};
    const cuda_block_q8_K *xqb[16] = {NULL};
    uint32_t np = 0;
    for (; np < 16u; np++) {
        uint32_t local_pair = local_start + np;
        if (local_pair >= counts[expert]) break;
        pair[np] = sorted_pairs[offsets[expert] + local_pair];
        xqb[np] = midq + (uint64_t)pair[np] * midq_blocks;
    }
    if (midq_blocks <= 8u) {
        for (uint32_t i = threadIdx.x; i < np * midq_blocks; i += blockDim.x) {
            uint32_t p = i / midq_blocks;
            uint32_t b = i - p * midq_blocks;
            sxq[p][b] = xqb[p][b];
        }
        __syncthreads();
        for (uint32_t p = 0; p < np; p++) xqb[p] = sxq[p];
    }
    for (uint32_t rr = 0; rr < 64u; rr++) {
        uint32_t row = blockIdx.x * 2048u + row_lane + rr * 32u;
        if (row >= out_dim) continue;
        const cuda_block_q2_K *wr = (const cuda_block_q2_K *)(down_base + (uint64_t)expert * down_expert_bytes + (uint64_t)row * down_row_bytes);
        float acc[16] = {0.0f};
        for (uint32_t b = lane; b < midq_blocks; b += 8u) {
            dev_dot_q2_K_q8_K_block8(wr + b, xqb[0] ? xqb[0] + b : NULL, xqb[1] ? xqb[1] + b : NULL,
                                     xqb[2] ? xqb[2] + b : NULL, xqb[3] ? xqb[3] + b : NULL,
                                     xqb[4] ? xqb[4] + b : NULL, xqb[5] ? xqb[5] + b : NULL,
                                     xqb[6] ? xqb[6] + b : NULL, xqb[7] ? xqb[7] + b : NULL, np < 8u ? np : 8u, acc);
            if (np > 8u) {
                dev_dot_q2_K_q8_K_block8(wr + b, xqb[8] ? xqb[8] + b : NULL, xqb[9] ? xqb[9] + b : NULL,
                                         xqb[10] ? xqb[10] + b : NULL, xqb[11] ? xqb[11] + b : NULL,
                                         xqb[12] ? xqb[12] + b : NULL, xqb[13] ? xqb[13] + b : NULL,
                                         xqb[14] ? xqb[14] + b : NULL, xqb[15] ? xqb[15] + b : NULL, np - 8u, acc + 8);
            }
        }
        for (uint32_t p = 0; p < np; p++) {
            acc[p] = quarter_warp_sum_f32(acc[p], lane);
            if (lane == 0) {
                if (atomic_out) {
                    uint32_t tok = pair[p] / n_expert;
                    atomicAdd(down_out + (uint64_t)tok * out_dim + row, acc[p]);
                } else {
                    down_out[(uint64_t)pair[p] * out_dim + row] = acc[p];
                }
            }
        }
    }
}

template <uint32_t ROW_SPAN>
__global__ static void moe_down_expert_tile16_rowspan_kernel(
        float *down_out,
        const char *down_base,
        const cuda_block_q8_K *midq,
        const uint32_t *sorted_pairs,
        const uint32_t *offsets,
        const uint32_t *counts,
        const uint32_t *tile_total,
        const uint32_t *tile_experts,
        const uint32_t *tile_starts,
        uint64_t down_expert_bytes,
        uint64_t down_row_bytes,
        uint32_t midq_blocks,
        uint32_t out_dim,
        uint32_t n_expert,
        uint32_t atomic_out) {
    uint32_t tile = blockIdx.y;
    if (tile >= *tile_total) return;
    uint32_t local_start = tile_starts[tile];
    if (local_start & 8u) return;
    uint32_t lane = threadIdx.x & 7u;
    uint32_t row_lane = threadIdx.x >> 3u;
    uint32_t expert = tile_experts[tile];
    __shared__ cuda_block_q8_K sxq[16][8];
    uint32_t pair[16] = {0};
    const cuda_block_q8_K *xqb[16] = {NULL};
    uint32_t np = 0;
    for (; np < 16u; np++) {
        uint32_t local_pair = local_start + np;
        if (local_pair >= counts[expert]) break;
        pair[np] = sorted_pairs[offsets[expert] + local_pair];
        xqb[np] = midq + (uint64_t)pair[np] * midq_blocks;
    }
    if (midq_blocks <= 8u) {
        for (uint32_t i = threadIdx.x; i < np * midq_blocks; i += blockDim.x) {
            uint32_t p = i / midq_blocks;
            uint32_t b = i - p * midq_blocks;
            sxq[p][b] = xqb[p][b];
        }
        __syncthreads();
        for (uint32_t p = 0; p < np; p++) xqb[p] = sxq[p];
    }
    for (uint32_t rr = 0; rr < ROW_SPAN / 32u; rr++) {
        uint32_t row = blockIdx.x * ROW_SPAN + row_lane + rr * 32u;
        if (row >= out_dim) continue;
        const cuda_block_q2_K *wr = (const cuda_block_q2_K *)(down_base + (uint64_t)expert * down_expert_bytes + (uint64_t)row * down_row_bytes);
        float acc[16] = {0.0f};
        for (uint32_t b = lane; b < midq_blocks; b += 8u) {
            dev_dot_q2_K_q8_K_block8(wr + b, xqb[0] ? xqb[0] + b : NULL, xqb[1] ? xqb[1] + b : NULL,
                                     xqb[2] ? xqb[2] + b : NULL, xqb[3] ? xqb[3] + b : NULL,
                                     xqb[4] ? xqb[4] + b : NULL, xqb[5] ? xqb[5] + b : NULL,
                                     xqb[6] ? xqb[6] + b : NULL, xqb[7] ? xqb[7] + b : NULL, np < 8u ? np : 8u, acc);
            if (np > 8u) {
                dev_dot_q2_K_q8_K_block8(wr + b, xqb[8] ? xqb[8] + b : NULL, xqb[9] ? xqb[9] + b : NULL,
                                         xqb[10] ? xqb[10] + b : NULL, xqb[11] ? xqb[11] + b : NULL,
                                         xqb[12] ? xqb[12] + b : NULL, xqb[13] ? xqb[13] + b : NULL,
                                         xqb[14] ? xqb[14] + b : NULL, xqb[15] ? xqb[15] + b : NULL, np - 8u, acc + 8);
            }
        }
        for (uint32_t p = 0; p < np; p++) {
            acc[p] = quarter_warp_sum_f32(acc[p], lane);
            if (lane == 0) {
                if (atomic_out) {
                    uint32_t tok = pair[p] / n_expert;
                    atomicAdd(down_out + (uint64_t)tok * out_dim + row, acc[p]);
                } else {
                    down_out[(uint64_t)pair[p] * out_dim + row] = acc[p];
                }
            }
        }
    }
}

__global__ static void moe_down_sorted_p2_qwarp32_kernel(
        float *down_out,
        const char *down_base,
        const cuda_block_q8_K *midq,
        const uint32_t *sorted_pairs,
        const int32_t *selected,
        uint64_t down_expert_bytes,
        uint64_t down_row_bytes,
        uint32_t midq_blocks,
        uint32_t out_dim,
        uint32_t n_expert,
        uint32_t pair_count) {
    uint32_t lane = threadIdx.x & 7u;
    uint32_t pair_lane = (threadIdx.x >> 3u) & 1u;
    uint32_t row = blockIdx.x * 16u + (threadIdx.x >> 4u);
    uint32_t sorted_idx = blockIdx.y * 2u + pair_lane;
    if (row >= out_dim || sorted_idx >= pair_count) return;
    uint32_t pair = sorted_pairs[sorted_idx];
    uint32_t tok = pair / n_expert;
    uint32_t slot = pair - tok * n_expert;
    int32_t expert_i = selected[(uint64_t)tok * n_expert + slot];
    if (expert_i < 0) expert_i = 0;
    const cuda_block_q2_K *wr = (const cuda_block_q2_K *)(down_base + (uint64_t)(uint32_t)expert_i * down_expert_bytes + (uint64_t)row * down_row_bytes);
    const cuda_block_q8_K *xq = midq + (uint64_t)pair * midq_blocks;
    float acc = 0.0f;
    for (uint32_t b = lane; b < midq_blocks; b += 8u) acc += dev_dot_q2_K_q8_K_block(wr + b, xq + b);
    acc = quarter_warp_sum_f32(acc, lane);
    if (lane == 0) down_out[(uint64_t)pair * out_dim + row] = acc;
}

__global__ static void moe_sum_kernel(float *out, const float *down, uint32_t out_dim, uint32_t n_expert, uint32_t n_tokens) {
    uint64_t gid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t n = (uint64_t)n_tokens * out_dim;
    if (gid >= n) return;
    uint32_t tok = gid / out_dim;
    uint32_t row = gid - (uint64_t)tok * out_dim;
    float acc = 0.0f;
    for (uint32_t e = 0; e < n_expert; e++) acc += down[((uint64_t)tok * n_expert + e) * out_dim + row];
    out[gid] = acc;
}

__global__ static void moe_sum_f16_kernel(float *out, const half *down_h, uint32_t out_dim, uint32_t n_expert, uint32_t n_tokens) {
    uint64_t gid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t n = (uint64_t)n_tokens * out_dim;
    if (gid >= n) return;
    uint32_t tok = gid / out_dim;
    uint32_t row = gid - (uint64_t)tok * out_dim;
    float acc = 0.0f;
    for (uint32_t e = 0; e < n_expert; e++) acc += __half2float(down_h[((uint64_t)tok * n_expert + e) * out_dim + row]);
    out[gid] = acc;
}

__global__ static void moe_sum_f16x2_kernel(float *out, const half *down_h, uint32_t out_dim, uint32_t n_expert, uint32_t n_tokens) {
    const uint64_t gid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t out_dim2 = out_dim >> 1u;
    const uint64_t n2 = (uint64_t)n_tokens * out_dim2;
    if (gid >= n2) return;
    const uint32_t tok = gid / out_dim2;
    const uint32_t row = (uint32_t)(gid - (uint64_t)tok * out_dim2) << 1u;
    float acc0 = 0.0f;
    float acc1 = 0.0f;
    for (uint32_t e = 0; e < n_expert; e++) {
        const uint64_t off = ((uint64_t)tok * n_expert + e) * out_dim + row;
        const float2 v = __half22float2(*reinterpret_cast<const __half2 *>(down_h + off));
        acc0 += v.x;
        acc1 += v.y;
    }
    const uint64_t out_off = (uint64_t)tok * out_dim + row;
    out[out_off] = acc0;
    out[out_off + 1u] = acc1;
}

__device__ __forceinline__ static void q2_K_scale_broadcast_w32(const unsigned char *blk, float *d, float *dmin) {
    float vd = 0.0f;
    float vm = 0.0f;
    if ((threadIdx.x & 31u) == 0u) {
        const uint16_t d_bits = (uint16_t)blk[80] | ((uint16_t)blk[81] << 8);
        const uint16_t dmin_bits = (uint16_t)blk[82] | ((uint16_t)blk[83] << 8);
        vd = dev_f16_to_f32(d_bits);
        vm = dev_f16_to_f32(dmin_bits);
    }
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
    *d = __shfl(vd, 0, 32);
    *dmin = __shfl(vm, 0, 32);
#else
    *d = __shfl_sync(FULL_WARP_MASK, vd, 0, 32);
    *dmin = __shfl_sync(FULL_WARP_MASK, vm, 0, 32);
#endif
}

__device__ __forceinline__ static float q2_K_dequant_256_scaled_w32(
        const unsigned char *blk,
        uint32_t lane,
        uint32_t kk,
        float d,
        float dmin) {
    const unsigned char *sc = blk;
    const unsigned char *qs = blk + 16u;
    const uint32_t g = (lane >> 4u) + (kk << 1u);
    const uint32_t within = g & 7u;
    const uint32_t qi = (g >> 3u) * 32u + (within & 1u) * 16u + (lane & 15u);
    const uint32_t shift = (within >> 1u) * 2u;
    const float q = (float)((qs[qi] >> shift) & 3u);
    const float scale = (float)(sc[g] & 0x0fu);
    const float mn = (float)(sc[g] >> 4u);
    return d * scale * q - dmin * mn;
}

template <int BN, int BK>
__device__ __forceinline__ static void q2_K_dequant_tile_half_rowwise(
        half *shB,
        const unsigned char *base,
        uint64_t row_bytes,
        uint32_t n0,
        uint32_t k0,
        uint32_t out_dim,
        uint32_t tid) {
    const uint32_t g = (k0 & 255u) >> 4u;
    const uint32_t within = g & 7u;
    const uint32_t qbase = (g >> 3u) * 32u + (within & 1u) * 16u;
    const uint32_t shift = (within >> 1u) * 2u;
    constexpr uint32_t KG = 2u;
    for (uint32_t j = tid; j < (uint32_t)(BN * (BK / KG)); j += blockDim.x) {
        const uint32_t nn = j / (uint32_t)(BK / KG);
        const uint32_t kk0 = (j - nn * (uint32_t)(BK / KG)) * KG;
        const uint32_t row = n0 + nn;
        if (row < out_dim) {
            const unsigned char *blk = base + (uint64_t)row * row_bytes + (uint64_t)(k0 >> 8u) * 84u;
            const float d = dev_f16_to_f32((uint16_t)blk[80] | ((uint16_t)blk[81] << 8));
            const float dm = dev_f16_to_f32((uint16_t)blk[82] | ((uint16_t)blk[83] << 8));
            const float s = (float)(blk[g] & 0x0fu);
            const float m = (float)(blk[g] >> 4u);
#pragma unroll
            for (uint32_t u = 0; u < KG; u++) {
                const uint32_t kk = kk0 + u;
                const float q = (float)((blk[16u + qbase + kk] >> shift) & 3u);
                shB[kk * (uint32_t)BN + nn] = __float2half(d * s * q - dm * m);
            }
        } else {
#pragma unroll
            for (uint32_t u = 0; u < KG; u++) {
                const uint32_t kk = kk0 + u;
                shB[kk * (uint32_t)BN + nn] = __float2half(0.0f);
            }
        }
    }
}

template <int BN, int BK>
__device__ __forceinline__ static void q2_K_dequant_pair_tile_half_rowwise_staged(
        half *shB0,
        half *shB1,
        const uint32_t *raw_rows,
        uint32_t k0,
        uint32_t tid) {
    const uint32_t g = (k0 & 255u) >> 4u;
    const uint32_t within = g & 7u;
    const uint32_t qbase = (g >> 3u) * 32u + (within & 1u) * 16u;
    const uint32_t shift = (within >> 1u) * 2u;
    constexpr uint32_t KG = 4u;
    constexpr uint32_t RAW_DWORDS = 84u / sizeof(uint32_t);
    constexpr uint32_t UNITS_PER_TILE = (uint32_t)(BN * (BK / KG));
    for (uint32_t j = tid; j < 2u * UNITS_PER_TILE; j += blockDim.x) {
        const uint32_t tile = j / UNITS_PER_TILE;
        const uint32_t rem = j - tile * UNITS_PER_TILE;
        const uint32_t nn = rem / (uint32_t)(BK / KG);
        const uint32_t kk0 = (rem - nn * (uint32_t)(BK / KG)) * KG;
        const uint32_t row = tile * (uint32_t)BN + nn;
        const unsigned char *blk =
                reinterpret_cast<const unsigned char *>(raw_rows + row * RAW_DWORDS);
        const uint32_t dm_bits = *reinterpret_cast<const uint32_t *>(blk + 80u);
        const float d = dev_f16_to_f32((uint16_t)dm_bits);
        const float dm = dev_f16_to_f32((uint16_t)(dm_bits >> 16u));
        const float s = (float)(blk[g] & 0x0fu);
        const float m = (float)(blk[g] >> 4u);
        const uint32_t qbits = *reinterpret_cast<const uint32_t *>(blk + 16u + qbase + kk0);
        const uint32_t q0 = (qbits >> shift) & 3u;
        const uint32_t q1 = (qbits >> (8u + shift)) & 3u;
        const uint32_t q2 = (qbits >> (16u + shift)) & 3u;
        const uint32_t q3 = (qbits >> (24u + shift)) & 3u;
        const float ds = d * s;
        const float dmm = dm * m;
        half *shB = tile == 0u ? shB0 : shB1;
        half *dst = shB + nn * (uint32_t)BK + kk0;
        *reinterpret_cast<uint32_t *>(dst) =
                dev_pack_half2_bits(ds * (float)q0 - dmm, ds * (float)q1 - dmm);
        *reinterpret_cast<uint32_t *>(dst + 2u) =
                dev_pack_half2_bits(ds * (float)q2 - dmm, ds * (float)q3 - dmm);
    }
}

template <int BN, int BK>
__device__ __forceinline__ static void q2_K_dequant_dual_pair_tile_half_rowwise(
        half *shB0g,
        half *shB0u,
        half *shB1g,
        half *shB1u,
        const unsigned char *gate_base,
        const unsigned char *up_base,
        uint64_t row_bytes,
        uint32_t n0,
        uint32_t k0,
        uint32_t out_dim,
        uint32_t tid) {
    const uint32_t g = (k0 & 255u) >> 4u;
    const uint32_t within = g & 7u;
    const uint32_t qbase = (g >> 3u) * 32u + (within & 1u) * 16u;
    const uint32_t shift = (within >> 1u) * 2u;
    constexpr uint32_t KG = 4u;
    constexpr uint32_t UNITS_PER_TILE = (uint32_t)(BN * (BK / KG));
    for (uint32_t j = tid; j < 2u * UNITS_PER_TILE; j += blockDim.x) {
        const uint32_t tile = j / UNITS_PER_TILE;
        const uint32_t rem = j - tile * UNITS_PER_TILE;
        const uint32_t nn = rem / (uint32_t)(BK / KG);
        const uint32_t kk0 = (rem - nn * (uint32_t)(BK / KG)) * KG;
        const uint32_t row = n0 + tile * (uint32_t)BN + nn;
        half *shBg = tile == 0u ? shB0g : shB1g;
        half *shBu = tile == 0u ? shB0u : shB1u;
        uint32_t gv0 = 0u;
        uint32_t gv1 = 0u;
        uint32_t uv0 = 0u;
        uint32_t uv1 = 0u;
        if (row < out_dim) {
            const unsigned char *gb = gate_base + (uint64_t)row * row_bytes + (uint64_t)(k0 >> 8u) * 84u;
            const unsigned char *ub = up_base + (uint64_t)row * row_bytes + (uint64_t)(k0 >> 8u) * 84u;
            const float gd = dev_f16_to_f32((uint16_t)gb[80] | ((uint16_t)gb[81] << 8));
            const float gm = dev_f16_to_f32((uint16_t)gb[82] | ((uint16_t)gb[83] << 8));
            const float ud = dev_f16_to_f32((uint16_t)ub[80] | ((uint16_t)ub[81] << 8));
            const float um = dev_f16_to_f32((uint16_t)ub[82] | ((uint16_t)ub[83] << 8));
            const float gs = (float)(gb[g] & 0x0fu);
            const float gmn = (float)(gb[g] >> 4u);
            const float us = (float)(ub[g] & 0x0fu);
            const float umn = (float)(ub[g] >> 4u);
            const uint32_t gqbits = *reinterpret_cast<const uint32_t *>(gb + 16u + qbase + kk0);
            const uint32_t uqbits = *reinterpret_cast<const uint32_t *>(ub + 16u + qbase + kk0);
            const uint32_t gq0 = (gqbits >> shift) & 3u;
            const uint32_t gq1 = (gqbits >> (8u + shift)) & 3u;
            const uint32_t gq2 = (gqbits >> (16u + shift)) & 3u;
            const uint32_t gq3 = (gqbits >> (24u + shift)) & 3u;
            const uint32_t uq0 = (uqbits >> shift) & 3u;
            const uint32_t uq1 = (uqbits >> (8u + shift)) & 3u;
            const uint32_t uq2 = (uqbits >> (16u + shift)) & 3u;
            const uint32_t uq3 = (uqbits >> (24u + shift)) & 3u;
            const float gds = gd * gs;
            const float gmm = gm * gmn;
            const float uds = ud * us;
            const float umm = um * umn;
            gv0 = dev_pack_half2_bits(gds * (float)gq0 - gmm,
                                      gds * (float)gq1 - gmm);
            gv1 = dev_pack_half2_bits(gds * (float)gq2 - gmm,
                                      gds * (float)gq3 - gmm);
            uv0 = dev_pack_half2_bits(uds * (float)uq0 - umm,
                                      uds * (float)uq1 - umm);
            uv1 = dev_pack_half2_bits(uds * (float)uq2 - umm,
                                      uds * (float)uq3 - umm);
        }
        const uint32_t sj = nn * (uint32_t)BK + kk0;
        *reinterpret_cast<uint32_t *>(shBg + sj) = gv0;
        *reinterpret_cast<uint32_t *>(shBg + sj + 2u) = gv1;
        *reinterpret_cast<uint32_t *>(shBu + sj) = uv0;
        *reinterpret_cast<uint32_t *>(shBu + sj + 2u) = uv1;
    }
}

__device__ __forceinline__ static float moe_silu_oldhip(float x) {
    return x * (1.0f / (1.0f + expf(-x)));
}

__global__ __launch_bounds__(32) static void moe_gate_up_mid_q2K_rows_rpb1_w32_kernel(
        float *gate_out,
        float *up_out,
        float *mid_out,
        const char *gate_base,
        const char *up_base,
        const float *x,
        const int32_t *selected,
        const float *weights,
        uint64_t gate_expert_bytes,
        uint64_t gate_row_bytes,
        uint32_t expert_in_dim,
        uint32_t expert_mid_dim,
        uint32_t n_expert,
        float clamp,
        int store_gate_up) {
    const uint32_t lane = threadIdx.x;
    const uint32_t row = blockIdx.x;
    const uint32_t pair = blockIdx.y;
    if (row >= expert_mid_dim || lane >= 32u) return;
    const uint32_t tok = pair / n_expert;
    const uint32_t slot = pair - tok * n_expert;
    int32_t expert_i = selected[(uint64_t)tok * n_expert + slot];
    if (expert_i < 0) expert_i = 0;
    const uint32_t expert = (uint32_t)expert_i;
    const float *xr = x + (uint64_t)tok * expert_in_dim;
    const unsigned char *gr = (const unsigned char *)gate_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes;
    const unsigned char *ur = (const unsigned char *)up_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes;

    float gate = 0.0f;
    float up = 0.0f;
    const uint32_t nb = expert_in_dim >> 8u;
    for (uint32_t b = 0; b < nb; b++) {
        const unsigned char *gb = gr + (uint64_t)b * 84u;
        const unsigned char *ub = ur + (uint64_t)b * 84u;
        float gd, gdmin, ud, udmin;
        q2_K_scale_broadcast_w32(gb, &gd, &gdmin);
        q2_K_scale_broadcast_w32(ub, &ud, &udmin);
        const uint64_t xbase = (uint64_t)b * 256u;
#pragma unroll
        for (uint32_t kk = 0; kk < 8u; kk++) {
            const uint32_t i = lane + (kk << 5u);
            const float xv = xr[xbase + i];
            gate += q2_K_dequant_256_scaled_w32(gb, lane, kk, gd, gdmin) * xv;
            up += q2_K_dequant_256_scaled_w32(ub, lane, kk, ud, udmin) * xv;
        }
    }

    gate = warp_sum_f32(gate);
    up = warp_sum_f32(up);
    if (lane == 0u) {
        if (clamp > 1.0e-6f) {
            if (gate > clamp) gate = clamp;
            if (up > clamp) up = clamp;
            if (up < -clamp) up = -clamp;
        }
        const uint64_t off = (uint64_t)pair * expert_mid_dim + row;
        if (store_gate_up) {
            gate_out[off] = gate;
            up_out[off] = up;
        }
        mid_out[off] = moe_silu_oldhip(gate) * up * weights[(uint64_t)tok * n_expert + slot];
    }
}

__global__ static void moe_gate_up_mid_q2K_rows_w32_kernel(
        float *gate_out,
        float *up_out,
        float *mid_out,
        const char *gate_base,
        const char *up_base,
        const float *x,
        const int32_t *selected,
        const float *weights,
        uint64_t gate_expert_bytes,
        uint64_t gate_row_bytes,
        uint32_t expert_in_dim,
        uint32_t expert_mid_dim,
        uint32_t n_expert,
        float clamp,
        int store_gate_up) {
    const uint32_t tid = threadIdx.x;
    const uint32_t lane = tid & 31u;
    const uint32_t wave = tid >> 5u;
    const uint32_t rows_per_block = blockDim.x >> 5u;
    const uint32_t row = blockIdx.x * rows_per_block + wave;
    const uint32_t pair = blockIdx.y;
    if (row >= expert_mid_dim) return;
    const uint32_t tok = pair / n_expert;
    const uint32_t slot = pair - tok * n_expert;
    int32_t expert_i = selected[(uint64_t)tok * n_expert + slot];
    if (expert_i < 0) expert_i = 0;
    const uint32_t expert = (uint32_t)expert_i;
    const float *xr = x + (uint64_t)tok * expert_in_dim;
    const unsigned char *gr = (const unsigned char *)gate_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes;
    const unsigned char *ur = (const unsigned char *)up_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes;

    float gate = 0.0f;
    float up = 0.0f;
    const uint32_t nb = expert_in_dim >> 8u;
    for (uint32_t b = 0; b < nb; b++) {
        const unsigned char *gb = gr + (uint64_t)b * 84u;
        const unsigned char *ub = ur + (uint64_t)b * 84u;
        float gd, gdmin, ud, udmin;
        q2_K_scale_broadcast_w32(gb, &gd, &gdmin);
        q2_K_scale_broadcast_w32(ub, &ud, &udmin);
        const uint64_t xbase = (uint64_t)b * 256u;
#pragma unroll
        for (uint32_t kk = 0; kk < 8u; kk++) {
            const uint32_t i = lane + (kk << 5u);
            const float xv = xr[xbase + i];
            gate += q2_K_dequant_256_scaled_w32(gb, lane, kk, gd, gdmin) * xv;
            up += q2_K_dequant_256_scaled_w32(ub, lane, kk, ud, udmin) * xv;
        }
    }

    gate = warp_sum_f32(gate);
    up = warp_sum_f32(up);
    if (lane == 0u) {
        if (clamp > 1.0e-6f) {
            if (gate > clamp) gate = clamp;
            if (up > clamp) up = clamp;
            if (up < -clamp) up = -clamp;
        }
        const uint64_t off = (uint64_t)pair * expert_mid_dim + row;
        if (store_gate_up) {
            gate_out[off] = gate;
            up_out[off] = up;
        }
        mid_out[off] = moe_silu_oldhip(gate) * up * weights[(uint64_t)tok * n_expert + slot];
    }
}

/* Routed Q2_K down projection of float mid activations, one output row per
 * wave: lane l sums the weights l, l + 32, ... of every 256-weight block of
 * every selected expert, and the lanes reduce pairwise.
 *
 * Decode of this projection is bound by arithmetic and by the number of
 * loads, so a lane fetches the sub-block scales of a block as two words and
 * its two quant bytes once, and the per-lane offsets live in the base
 * pointers so that the loads need no per-lane address arithmetic.
 *
 * The values and their order are pinned: a weight is
 * (d * scale) * q - dmin * min, fused into the product with the activation,
 * and the eight weights of a block are added in the order the fast-math
 * build gave the loop this replaces, so the result is bit-identical to it. */
__global__ static void moe_down_q2K_sum_rows_w32_kernel(
        float *out,
        const char *down_base,
        const float *mid,
        const int32_t *selected,
        uint32_t n_tokens,
        uint32_t expert_mid_dim,
        uint32_t out_dim,
        uint64_t down_expert_bytes,
        uint64_t down_row_bytes,
        uint32_t n_expert) {
#pragma clang fp reassociate(off)
    const uint32_t tid = threadIdx.x;
    const uint32_t lane = tid & 31u;
    const uint32_t wave = tid >> 5u;
    const uint32_t rows_per_block = blockDim.x >> 5u;
    const uint32_t row = blockIdx.x * rows_per_block + wave;
    const uint32_t tok = blockIdx.y;
    if (row >= out_dim || tok >= n_tokens) return;

    /* Lanes 0..15 hold the weights of the even 16-weight sub-blocks, lanes
     * 16..31 those of the odd ones. */
    const uint32_t odd = lane >> 4u;
    float acc = 0.0f;
    const uint32_t nb = expert_mid_dim >> 8u;
#pragma unroll
    for (uint32_t slot = 0; slot < DS4_ROCM_N_EXPERT_USED; slot++) {
        if (slot >= n_expert) continue;
        int32_t expert_i = selected[(uint64_t)tok * n_expert + slot];
        if (expert_i < 0) expert_i = 0;
        const unsigned char *dr = (const unsigned char *)down_base + (uint64_t)(uint32_t)expert_i * down_expert_bytes + (uint64_t)row * down_row_bytes;
        const unsigned char *quants = dr + 16u + lane;
        const float *mr = mid + ((uint64_t)tok * n_expert + slot) * expert_mid_dim + lane;
        for (uint32_t b = 0; b < nb; b++) {
            const uint32_t off = b * 84u;
            uint64_t sc64[2];
            uint32_t dm_bits;
            __builtin_memcpy(sc64, dr + off, sizeof(sc64));
            __builtin_memcpy(&dm_bits, dr + off + 80u, sizeof(dm_bits));
            const uint32_t q[2] = { quants[off], quants[off + 32u] };
            const float *m = mr + (b << 8u);
            const float d = dev_f16_to_f32((uint16_t)dm_bits);
            const float dmin = dev_f16_to_f32((uint16_t)(dm_bits >> 16u));
            /* This lane's eight (scale, min) bytes, two per dword. */
            const uint64_t s0 = sc64[0] >> (odd << 3u), s1 = sc64[1] >> (odd << 3u);
            const uint32_t sc[4] = { (uint32_t)s0, (uint32_t)(s0 >> 32u), (uint32_t)s1, (uint32_t)(s1 >> 32u) };
#pragma unroll
            for (uint32_t k = 0; k < 8u; k++) {
                /* 7, 3, 5, 1, 6, 2, 4, 0: the order of the replaced kernel. */
                const uint32_t kk = 7u - (((k & 1u) << 2u) | (k & 2u) | ((k >> 2u) & 1u));
                const uint32_t s = sc[kk >> 1u] >> ((kk & 1u) << 4u);
                const float ds = d * (float)(s & 15u);
                const float dmm = dmin * (float)((s >> 4u) & 15u);
                const float qf = (float)((q[kk >> 2u] >> ((kk & 3u) << 1u)) & 3u);
                acc = __builtin_fmaf(__builtin_fmaf(ds, qf, -dmm), m[kk << 5u], acc);
            }
        }
    }
    acc = warp_sum_f32(acc);
    if (lane == 0u) out[(uint64_t)tok * out_dim + row] = acc;
}

__global__ static void moe_gate_up_mid_q2K_rows_w32_ptrs_kernel(
        float *gate_out,
        float *up_out,
        float *mid_out,
        const char * const *gate_slots,
        const char * const *up_slots,
        const uint8_t *pair_missing,
        uint32_t want_missing,
        const float *x,
        const int32_t *selected,
        const float *weights,
        uint64_t gate_row_bytes,
        uint32_t expert_in_dim,
        uint32_t expert_mid_dim,
        uint32_t n_expert,
        uint32_t active_mask,
        float clamp,
        int store_gate_up) {
    const uint32_t tid = threadIdx.x;
    const uint32_t lane = tid & 31u;
    const uint32_t wave = tid >> 5u;
    const uint32_t rows_per_block = blockDim.x >> 5u;
    const uint32_t row = blockIdx.x * rows_per_block + wave;
    const uint32_t pair = blockIdx.y;
    if (row >= expert_mid_dim || rows_per_block == 0u) return;
    const uint32_t tok = pair / n_expert;
    const uint32_t slot = pair - tok * n_expert;
    if ((active_mask & (1u << slot)) == 0u) return;
    if (pair_missing &&
        ((uint32_t)(pair_missing[pair] != 0) != want_missing)) {
        return;
    }
    int32_t compact_i = selected[(uint64_t)tok * n_expert + slot];
    if (compact_i < 0) compact_i = 0;
    const char *gate_base = gate_slots[(uint32_t)compact_i];
    const char *up_base = up_slots[(uint32_t)compact_i];
    if (!gate_base || !up_base) return;
    const float *xr = x + (uint64_t)tok * expert_in_dim;
    const unsigned char *gr =
        (const unsigned char *)gate_base + (uint64_t)row * gate_row_bytes;
    const unsigned char *ur =
        (const unsigned char *)up_base + (uint64_t)row * gate_row_bytes;

    float gate = 0.0f;
    float up = 0.0f;
    const uint32_t nb = expert_in_dim >> 8u;
    for (uint32_t b = 0; b < nb; b++) {
        const unsigned char *gb = gr + (uint64_t)b * 84u;
        const unsigned char *ub = ur + (uint64_t)b * 84u;
        float gd, gdmin, ud, udmin;
        q2_K_scale_broadcast_w32(gb, &gd, &gdmin);
        q2_K_scale_broadcast_w32(ub, &ud, &udmin);
        const uint64_t xbase = (uint64_t)b * 256u;
#pragma unroll
        for (uint32_t kk = 0; kk < 8u; kk++) {
            const uint32_t i = lane + (kk << 5u);
            const float xv = xr[xbase + i];
            gate += q2_K_dequant_256_scaled_w32(gb, lane, kk, gd, gdmin) * xv;
            up += q2_K_dequant_256_scaled_w32(ub, lane, kk, ud, udmin) * xv;
        }
    }

    gate = warp_sum_f32(gate);
    up = warp_sum_f32(up);
    if (lane == 0u) {
        if (clamp > 1.0e-6f) {
            if (gate > clamp) gate = clamp;
            if (up > clamp) up = clamp;
            if (up < -clamp) up = -clamp;
        }
        const uint64_t off = (uint64_t)pair * expert_mid_dim + row;
        if (store_gate_up) {
            gate_out[off] = gate;
            up_out[off] = up;
        }
        mid_out[off] =
            moe_silu_oldhip(gate) * up *
            weights[(uint64_t)tok * n_expert + slot];
    }
}

__global__ static void moe_down_q2K_sum_rows_w32_ptrs_batch_kernel(
        float *out,
        const char * const *down_slots,
        const float *mid,
        const int32_t *selected,
        uint32_t n_tokens,
        uint32_t expert_mid_dim,
        uint32_t out_dim,
        uint64_t down_row_bytes,
        uint32_t n_expert) {
    const uint32_t tid = threadIdx.x;
    const uint32_t lane = tid & 31u;
    const uint32_t wave = tid >> 5u;
    const uint32_t rows_per_block = blockDim.x >> 5u;
    const uint32_t row = blockIdx.x * rows_per_block + wave;
    const uint32_t tok = blockIdx.y;
    if (row >= out_dim || tok >= n_tokens || rows_per_block == 0u) return;

    float acc = 0.0f;
    const uint32_t nb = expert_mid_dim >> 8u;
#pragma unroll
    for (uint32_t slot = 0; slot < DS4_ROCM_N_EXPERT_USED; slot++) {
        if (slot >= n_expert) continue;
        int32_t compact_i = selected[(uint64_t)tok * n_expert + slot];
        if (compact_i < 0) compact_i = 0;
        const char *down_base = down_slots[(uint32_t)compact_i];
        if (!down_base) continue;
        const unsigned char *dr =
            (const unsigned char *)down_base + (uint64_t)row * down_row_bytes;
        const float *mr =
            mid + ((uint64_t)tok * n_expert + slot) * expert_mid_dim;
        for (uint32_t b = 0; b < nb; b++) {
            const unsigned char *db = dr + (uint64_t)b * 84u;
            float d, dmin;
            q2_K_scale_broadcast_w32(db, &d, &dmin);
            const uint64_t mbase = (uint64_t)b * 256u;
#pragma unroll
            for (uint32_t kk = 0; kk < 8u; kk++) {
                const uint32_t i = lane + (kk << 5u);
                acc += q2_K_dequant_256_scaled_w32(db, lane, kk, d, dmin) *
                       mr[mbase + i];
            }
        }
    }
    acc = warp_sum_f32(acc);
    if (lane == 0u) out[(uint64_t)tok * out_dim + row] = acc;
}

template <uint32_t PAIR_TILE, bool OUT_F16=false>
__global__ static void moe_gate_up_mid_q2K_expert_batch_sharedx_kernel(
        float *mid_out,
        half *mid_out_h,
        const char *gate_base,
        const char *up_base,
        const float *x,
        const float *weights,
        const uint32_t *counts,
        const uint32_t *offsets,
        const uint32_t *pairs,
        uint32_t min_count,
        uint32_t max_count,
        uint32_t expert_in_dim,
        uint32_t expert_mid_dim,
        uint64_t gate_expert_bytes,
        uint64_t gate_row_bytes,
        uint32_t n_expert,
        float clamp) {
    extern __shared__ float shx[];
    const uint32_t tid = threadIdx.x;
    const uint32_t lane = tid & 31u;
    const uint32_t wave = tid >> 5u;
    const uint32_t rows_per_block = blockDim.x >> 5u;
    const uint32_t row = blockIdx.x * rows_per_block + wave;
    const uint32_t expert = blockIdx.y;
    const bool row_valid = row < expert_mid_dim;
    const uint32_t count = counts[expert];
    if (count == 0u || count < min_count || (max_count != 0u && count >= max_count)) return;
    const uint32_t first = offsets[expert];
    const unsigned char *grow = (const unsigned char *)gate_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)(row_valid ? row : 0u) * gate_row_bytes;
    const unsigned char *urow = (const unsigned char *)up_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)(row_valid ? row : 0u) * gate_row_bytes;
    const uint32_t nb = expert_in_dim >> 8u;
    for (uint32_t p0 = 0; p0 < count; p0 += PAIR_TILE) {
        uint32_t pair[PAIR_TILE];
        float g_acc[PAIR_TILE];
        float u_acc[PAIR_TILE];
#pragma unroll
        for (uint32_t u = 0; u < PAIR_TILE; u++) {
            pair[u] = (p0 + u < count) ? pairs[first + p0 + u] : UINT32_MAX;
            g_acc[u] = 0.0f;
            u_acc[u] = 0.0f;
        }
        for (uint32_t b = 0; b < nb; b++) {
            const uint64_t xbase = (uint64_t)b * 256u;
            for (uint32_t j = tid; j < PAIR_TILE * 256u; j += blockDim.x) {
                const uint32_t u = j >> 8u;
                const uint32_t k = j & 255u;
                if (pair[u] != UINT32_MAX) {
                    const uint32_t tok = pair[u] / n_expert;
                    shx[j] = x[(uint64_t)tok * expert_in_dim + xbase + k];
                } else {
                    shx[j] = 0.0f;
                }
            }
            __syncthreads();
            if (row_valid) {
                const unsigned char *gb = grow + (uint64_t)b * 84u;
                const unsigned char *ub = urow + (uint64_t)b * 84u;
                float gd, gdmin, ud, udmin;
                q2_K_scale_broadcast_w32(gb, &gd, &gdmin);
                q2_K_scale_broadcast_w32(ub, &ud, &udmin);
#pragma unroll
                for (uint32_t kk = 0; kk < 8u; kk++) {
                    const uint32_t i = lane + (kk << 5u);
                    const float gwv = q2_K_dequant_256_scaled_w32(gb, lane, kk, gd, gdmin);
                    const float uwv = q2_K_dequant_256_scaled_w32(ub, lane, kk, ud, udmin);
#pragma unroll
                    for (uint32_t u = 0; u < PAIR_TILE; u++) {
                        const float xv = shx[(u << 8u) + i];
                        g_acc[u] += gwv * xv;
                        u_acc[u] += uwv * xv;
                    }
                }
            }
            __syncthreads();
        }
#pragma unroll
        for (uint32_t u = 0; u < PAIR_TILE; u++) {
            g_acc[u] = warp_sum_f32(g_acc[u]);
            u_acc[u] = warp_sum_f32(u_acc[u]);
        }
        if (lane == 0u && row_valid) {
#pragma unroll
            for (uint32_t u = 0; u < PAIR_TILE; u++) {
                if (pair[u] != UINT32_MAX) {
                    float g = g_acc[u];
                    float upv = u_acc[u];
                    if (clamp > 1.0e-6f) {
                        if (g > clamp) g = clamp;
                        if (upv > clamp) upv = clamp;
                        if (upv < -clamp) upv = -clamp;
                    }
                    const float v = moe_silu_oldhip(g) * upv * weights[pair[u]];
                    if (OUT_F16) mid_out_h[(uint64_t)pair[u] * expert_mid_dim + row] = __float2half(v);
                    else mid_out[(uint64_t)pair[u] * expert_mid_dim + row] = v;
                }
            }
        }
    }
}

template <uint32_t PAIR_TILE, bool MID_F16=false, bool OUT_F16=false, bool SLOT_MAJOR=false>
__global__ static void moe_down_q2K_expert_batch_sharedmid_kernel(
        float *down_out,
        half *down_out_h,
        const char *down_base,
        const float *mid,
        const half *mid_h,
        const uint32_t *counts,
        const uint32_t *offsets,
        const uint32_t *pairs,
        uint32_t min_count,
        uint32_t max_count,
        uint32_t expert_mid_dim,
        uint32_t out_dim,
        uint64_t down_expert_bytes,
        uint64_t down_row_bytes,
        uint32_t n_expert,
        uint32_t n_tokens = 0u,
        const uint32_t *active_count = NULL,
        const uint32_t *active_experts = NULL) {
    extern __shared__ float shmid[];
    const uint32_t tid = threadIdx.x;
    const uint32_t lane = tid & 31u;
    const uint32_t wave = tid >> 5u;
    const uint32_t rows_per_block = blockDim.x >> 5u;
    const uint32_t row = blockIdx.x * rows_per_block + wave;
    const uint32_t expert_index = blockIdx.y;
    if (active_experts && expert_index >= *active_count) return;
    const uint32_t expert = active_experts ? active_experts[expert_index] : expert_index;
    const bool row_valid = row < out_dim;
    const uint32_t count = counts[expert];
    if (count == 0u || count < min_count || (max_count != 0u && count >= max_count)) return;
    const uint32_t first = offsets[expert];
    const unsigned char *drow = (const unsigned char *)down_base + (uint64_t)expert * down_expert_bytes + (uint64_t)(row_valid ? row : 0u) * down_row_bytes;
    const uint32_t nb = expert_mid_dim >> 8u;
    for (uint32_t p0 = 0; p0 < count; p0 += PAIR_TILE) {
        uint32_t pair[PAIR_TILE];
        float acc[PAIR_TILE];
#pragma unroll
        for (uint32_t u = 0; u < PAIR_TILE; u++) {
            pair[u] = (p0 + u < count) ? pairs[first + p0 + u] : UINT32_MAX;
            acc[u] = 0.0f;
        }
        for (uint32_t b = 0; b < nb; b++) {
            const uint64_t mbase = (uint64_t)b * 256u;
            for (uint32_t j = tid; j < PAIR_TILE * 256u; j += blockDim.x) {
                const uint32_t u = j >> 8u;
                const uint32_t k = j & 255u;
                if (pair[u] != UINT32_MAX) {
                    const uint64_t moff = (uint64_t)pair[u] * expert_mid_dim + mbase + k;
                    shmid[j] = MID_F16 ? __half2float(mid_h[moff]) : mid[moff];
                } else {
                    shmid[j] = 0.0f;
                }
            }
            __syncthreads();
            if (row_valid) {
                const unsigned char *db = drow + (uint64_t)b * 84u;
                float d, dmin;
                q2_K_scale_broadcast_w32(db, &d, &dmin);
#pragma unroll
                for (uint32_t kk = 0; kk < 8u; kk++) {
                    const uint32_t i = lane + (kk << 5u);
                    const float wv = q2_K_dequant_256_scaled_w32(db, lane, kk, d, dmin);
#pragma unroll
                    for (uint32_t u = 0; u < PAIR_TILE; u++) acc[u] += wv * shmid[(u << 8u) + i];
                }
            }
            __syncthreads();
        }
#pragma unroll
        for (uint32_t u = 0; u < PAIR_TILE; u++) acc[u] = warp_sum_f32(acc[u]);
        if (lane == 0u && row_valid) {
#pragma unroll
            for (uint32_t u = 0; u < PAIR_TILE; u++) {
                if (pair[u] != UINT32_MAX) {
                    if (OUT_F16) {
                        uint64_t dst = (uint64_t)pair[u] * out_dim + row;
                        if (SLOT_MAJOR) {
                            const uint32_t tok = pair[u] / n_expert;
                            const uint32_t slot = pair[u] - tok * n_expert;
                            dst = ((uint64_t)slot * n_tokens + tok) * out_dim + row;
                        }
                        down_out_h[dst] = __float2half(acc[u]);
                    } else {
                        uint64_t dst = (uint64_t)pair[u] * out_dim + row;
                        if (SLOT_MAJOR) {
                            const uint32_t tok = pair[u] / n_expert;
                            const uint32_t slot = pair[u] - tok * n_expert;
                            dst = ((uint64_t)slot * n_tokens + tok) * out_dim + row;
                        }
                        down_out[dst] = acc[u];
                    }
                }
            }
        }
    }
}

#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
__device__ __forceinline__ static float iq2_xxs_dequant_256_direct(const unsigned char *blk, uint32_t i) {
    const uint16_t d_bits = (uint16_t)blk[0] | ((uint16_t)blk[1] << 8);
    const uint16_t *q2 = reinterpret_cast<const uint16_t *>(blk + 2u);
    const uint32_t ib32 = i >> 5u;
    const uint32_t j = i & 31u;
    const uint32_t aux_g = (uint32_t)q2[ib32 * 4u + 0u] | ((uint32_t)q2[ib32 * 4u + 1u] << 16);
    const uint32_t aux_s = (uint32_t)q2[ib32 * 4u + 2u] | ((uint32_t)q2[ib32 * 4u + 3u] << 16);
    const uint32_t half = j >> 4u;
    const uint32_t g = (j >> 3u) & 1u;
    const uint32_t ii = j & 7u;
    const uint32_t grid_idx = (aux_g >> (8u * ((half << 1u) + g))) & 0xffu;
    const uint32_t sign_idx = (aux_s >> (14u * half + 7u * g)) & 127u;
    const uint64_t grid = cuda_iq2xxs_grid[grid_idx];
    const uint32_t signs = dev_unpack_iq2_signs(cuda_ksigns_iq2xs[sign_idx]);
    float w = (float)((grid >> (8u * ii)) & 0xffu);
    if (signs & (1u << ii)) w = -w;
    return dev_f16_to_f32(d_bits) * (0.125f + 0.25f * (float)(aux_s >> 28u)) * w;
}

template <int BN, int BK>
__device__ __forceinline__ static void iq2_xxs_dequant_dual_pair_tile_half_rowwise(
        half *shB0g,
        half *shB0u,
        half *shB1g,
        half *shB1u,
        const unsigned char *gate_base,
        const unsigned char *up_base,
        uint64_t row_bytes,
        uint32_t n0,
        uint32_t k0,
        uint32_t out_dim,
        uint32_t tid) {
    constexpr uint32_t KG = 4u;
    constexpr uint32_t UNITS_PER_TILE = (uint32_t)(BN * (BK / KG));
    for (uint32_t j = tid; j < 2u * UNITS_PER_TILE; j += blockDim.x) {
        const uint32_t tile = j / UNITS_PER_TILE;
        const uint32_t rem = j - tile * UNITS_PER_TILE;
        const uint32_t nn = rem / (uint32_t)(BK / KG);
        const uint32_t kk0 = (rem - nn * (uint32_t)(BK / KG)) * KG;
        const uint32_t row = n0 + tile * (uint32_t)BN + nn;
        half *shBg = tile == 0u ? shB0g : shB1g;
        half *shBu = tile == 0u ? shB0u : shB1u;
        uint32_t gv0 = 0u, gv1 = 0u, uv0 = 0u, uv1 = 0u;
        if (row < out_dim) {
            const unsigned char *gb = gate_base + (uint64_t)row * row_bytes + (uint64_t)(k0 >> 8u) * 66u;
            const unsigned char *ub = up_base + (uint64_t)row * row_bytes + (uint64_t)(k0 >> 8u) * 66u;
            const uint32_t kk_base = k0 & 255u;
            const float g0 = iq2_xxs_dequant_256_direct(gb, kk_base + kk0 + 0u);
            const float g1 = iq2_xxs_dequant_256_direct(gb, kk_base + kk0 + 1u);
            const float g2 = iq2_xxs_dequant_256_direct(gb, kk_base + kk0 + 2u);
            const float g3 = iq2_xxs_dequant_256_direct(gb, kk_base + kk0 + 3u);
            const float u0 = iq2_xxs_dequant_256_direct(ub, kk_base + kk0 + 0u);
            const float u1 = iq2_xxs_dequant_256_direct(ub, kk_base + kk0 + 1u);
            const float u2 = iq2_xxs_dequant_256_direct(ub, kk_base + kk0 + 2u);
            const float u3 = iq2_xxs_dequant_256_direct(ub, kk_base + kk0 + 3u);
            gv0 = dev_pack_half2_bits(g0, g1);
            gv1 = dev_pack_half2_bits(g2, g3);
            uv0 = dev_pack_half2_bits(u0, u1);
            uv1 = dev_pack_half2_bits(u2, u3);
        }
        const uint32_t sj = nn * (uint32_t)BK + kk0;
        *reinterpret_cast<uint32_t *>(shBg + sj) = gv0;
        *reinterpret_cast<uint32_t *>(shBg + sj + 2u) = gv1;
        *reinterpret_cast<uint32_t *>(shBu + sj) = uv0;
        *reinterpret_cast<uint32_t *>(shBu + sj + 2u) = uv1;
    }
}

template <int MTILES=8, int BM=16, int BN=16, int BK=16, bool OUT_F16=false, bool X_F16=false>
__global__ static void moe_gate_up_mid_iq2_hotlist_wmma_n2_kernel(
        float *mid_out,
        half *mid_out_h,
        const char *gate_base,
        const char *up_base,
        const float *x,
        const half *x_h,
        const float *weights,
        const uint32_t *counts,
        const uint32_t *offsets,
        const uint32_t *pairs,
        const uint32_t *hot_experts,
        uint32_t hot_count,
        uint32_t expert_in_dim,
        uint32_t expert_mid_dim,
        uint64_t gate_expert_bytes,
        uint64_t gate_row_bytes,
        float clamp) {
    extern __shared__ unsigned char raw_sh[];
    half *shA = reinterpret_cast<half *>(raw_sh);
    half *shBg0 = shA + MTILES * BM * BK;
    half *shBu0 = shBg0 + BK * BN;
    half *shBg1 = shBu0 + BK * BN;
    half *shBu1 = shBg1 + BK * BN;
    float *shCg0 = reinterpret_cast<float *>(shBu1 + BK * BN);
    float *shCu0 = shCg0 + MTILES * BM * BN;
    float *shCg1 = shCu0 + MTILES * BM * BN;
    float *shCu1 = shCg1 + MTILES * BM * BN;
    const uint32_t hot_idx = (uint32_t)blockIdx.z;
    if (hot_idx >= hot_count) return;
    const uint32_t expert = hot_experts[hot_idx];
    const uint32_t count = counts[expert];
    const uint32_t m_group0 = (uint32_t)blockIdx.y * MTILES * BM;
    if (m_group0 >= count) return;
    const uint32_t n0 = (uint32_t)blockIdx.x * (2u * BN);
    const uint32_t tid = threadIdx.x;
    const uint32_t wave = tid >> 5u;
    const uint32_t first = offsets[expert];
    __shared__ uint32_t shPair[MTILES * BM];
    for (uint32_t j = tid; j < MTILES * BM; j += blockDim.x) {
        const uint32_t bucket_row = m_group0 + j;
        shPair[j] = (bucket_row < count) ? pairs[first + bucket_row] : UINT32_MAX;
    }
    __syncthreads();

    using frag_a = rocwmma::fragment<rocwmma::matrix_a, BM, BN, BK, half, rocwmma::row_major>;
    using frag_b = rocwmma::fragment<rocwmma::matrix_b, BM, BN, BK, half, rocwmma::col_major>;
    using frag_c = rocwmma::fragment<rocwmma::accumulator, BM, BN, BK, float>;
    frag_a a;
    frag_b bg0, bu0, bg1, bu1;
    frag_c accg0, accu0, accg1, accu1;
    if (wave < MTILES) {
        rocwmma::fill_fragment(accg0, 0.0f);
        rocwmma::fill_fragment(accu0, 0.0f);
        rocwmma::fill_fragment(accg1, 0.0f);
        rocwmma::fill_fragment(accu1, 0.0f);
    }

    const unsigned char *gew = (const unsigned char *)gate_base + (uint64_t)expert * gate_expert_bytes;
    const unsigned char *uew = (const unsigned char *)up_base + (uint64_t)expert * gate_expert_bytes;
    for (uint32_t k0 = 0; k0 < expert_in_dim; k0 += BK) {
        if (X_F16) {
            for (uint32_t j = tid; j < MTILES * BM * (BK / 2); j += blockDim.x) {
                const uint32_t pair_row = j / (BK / 2);
                const uint32_t kk2 = j - pair_row * (BK / 2);
                const uint32_t pair = shPair[pair_row];
                uint32_t v = 0u;
                if (pair != UINT32_MAX) {
                    const uint32_t token = pair / 6u;
                    const uint64_t xoff = (uint64_t)token * expert_in_dim + k0 + kk2 * 2u;
                    v = *reinterpret_cast<const uint32_t *>(x_h + xoff);
                }
                *reinterpret_cast<uint32_t *>(shA + pair_row * BK + kk2 * 2u) = v;
            }
        } else {
            for (uint32_t j = tid; j < MTILES * BM * BK; j += blockDim.x) {
                const uint32_t pair_row = j / BK;
                const uint32_t kk = j - pair_row * BK;
                const uint32_t pair = shPair[pair_row];
                shA[j] = pair != UINT32_MAX
                    ? __float2half(x[(uint64_t)(pair / 6u) * expert_in_dim + k0 + kk])
                    : __float2half(0.0f);
            }
        }
        iq2_xxs_dequant_dual_pair_tile_half_rowwise<BN, BK>(
                shBg0, shBu0, shBg1, shBu1, gew, uew, gate_row_bytes, n0, k0, expert_mid_dim, tid);
        __syncthreads();
        if (wave < MTILES) {
            rocwmma::load_matrix_sync(a, shA + wave * BM * BK, BK);
            rocwmma::load_matrix_sync(bg0, shBg0, BN);
            rocwmma::load_matrix_sync(bu0, shBu0, BN);
            rocwmma::load_matrix_sync(bg1, shBg1, BN);
            rocwmma::load_matrix_sync(bu1, shBu1, BN);
            rocwmma::mma_sync(accg0, a, bg0, accg0);
            rocwmma::mma_sync(accu0, a, bu0, accu0);
            rocwmma::mma_sync(accg1, a, bg1, accg1);
            rocwmma::mma_sync(accu1, a, bu1, accu1);
        }
        __syncthreads();
    }

    if (wave < MTILES) {
        rocwmma::store_matrix_sync(shCg0 + wave * BM * BN, accg0, BN, rocwmma::mem_row_major);
        rocwmma::store_matrix_sync(shCu0 + wave * BM * BN, accu0, BN, rocwmma::mem_row_major);
        rocwmma::store_matrix_sync(shCg1 + wave * BM * BN, accg1, BN, rocwmma::mem_row_major);
        rocwmma::store_matrix_sync(shCu1 + wave * BM * BN, accu1, BN, rocwmma::mem_row_major);
    }
    __syncthreads();

    for (uint32_t j = tid; j < MTILES * BM * BN; j += blockDim.x) {
        const uint32_t mt = j / (BM * BN);
        const uint32_t rem = j - mt * BM * BN;
        const uint32_t mm = rem / BN;
        const uint32_t nn = rem - mm * BN;
        const uint32_t pair = shPair[mt * BM + mm];
        if (pair != UINT32_MAX) {
            const uint32_t row0 = n0 + nn;
            const uint32_t row1 = n0 + BN + nn;
            const float wt = weights[pair];
            if (row0 < expert_mid_dim) {
                float g = shCg0[j], u = shCu0[j];
                if (clamp > 1.0e-6f) {
                    if (g > clamp) g = clamp;
                    if (u > clamp) u = clamp;
                    if (u < -clamp) u = -clamp;
                }
                const float v = moe_silu_oldhip(g) * u * wt;
                if (OUT_F16) mid_out_h[(uint64_t)pair * expert_mid_dim + row0] = __float2half(v);
                else mid_out[(uint64_t)pair * expert_mid_dim + row0] = v;
            }
            if (row1 < expert_mid_dim) {
                float g = shCg1[j], u = shCu1[j];
                if (clamp > 1.0e-6f) {
                    if (g > clamp) g = clamp;
                    if (u > clamp) u = clamp;
                    if (u < -clamp) u = -clamp;
                }
                const float v = moe_silu_oldhip(g) * u * wt;
                if (OUT_F16) mid_out_h[(uint64_t)pair * expert_mid_dim + row1] = __float2half(v);
                else mid_out[(uint64_t)pair * expert_mid_dim + row1] = v;
            }
        }
    }
}

template <int MTILES=8, int BM=16, int BN=16, int BK=16, bool OUT_F16=false, bool X_F16=false>
__global__ static void moe_gate_up_mid_q2K_hotlist_wmma_n2_kernel(
        float *mid_out,
        half *mid_out_h,
        const char *gate_base,
        const char *up_base,
        const float *x,
        const half *x_h,
        const float *weights,
        const uint32_t *counts,
        const uint32_t *offsets,
        const uint32_t *pairs,
        const uint32_t *hot_experts,
        uint32_t hot_count,
        uint32_t expert_in_dim,
        uint32_t expert_mid_dim,
        uint64_t gate_expert_bytes,
        uint64_t gate_row_bytes,
        uint32_t n_expert,
        float clamp) {
    extern __shared__ unsigned char raw_sh[];
    half *shA = reinterpret_cast<half *>(raw_sh);
    half *shBg0 = shA + MTILES * BM * BK;
    half *shBu0 = shBg0 + BK * BN;
    half *shBg1 = shBu0 + BK * BN;
    half *shBu1 = shBg1 + BK * BN;
    float *shCg0 = reinterpret_cast<float *>(shBu1 + BK * BN);
    float *shCu0 = shCg0 + MTILES * BM * BN;
    float *shCg1 = shCu0 + MTILES * BM * BN;
    float *shCu1 = shCg1 + MTILES * BM * BN;
    const uint32_t hot_idx = (uint32_t)blockIdx.z;
    if (hot_idx >= hot_count) return;
    const uint32_t expert = hot_experts[hot_idx];
    const uint32_t count = counts[expert];
    const uint32_t m_group0 = (uint32_t)blockIdx.y * MTILES * BM;
    if (m_group0 >= count) return;
    const uint32_t n0 = (uint32_t)blockIdx.x * (2u * BN);
    const uint32_t tid = threadIdx.x;
    const uint32_t wave = tid >> 5u;
    const uint32_t first = offsets[expert];
    __shared__ uint32_t shPair[MTILES * BM];
    for (uint32_t j = tid; j < MTILES * BM; j += blockDim.x) {
        const uint32_t bucket_row = m_group0 + j;
        shPair[j] = (bucket_row < count) ? pairs[first + bucket_row] : UINT32_MAX;
    }
    __syncthreads();

    using frag_a = rocwmma::fragment<rocwmma::matrix_a, BM, BN, BK, half, rocwmma::row_major>;
    using frag_b = rocwmma::fragment<rocwmma::matrix_b, BM, BN, BK, half, rocwmma::col_major>;
    using frag_c = rocwmma::fragment<rocwmma::accumulator, BM, BN, BK, float>;
    frag_a a;
    frag_b bg0;
    frag_b bu0;
    frag_b bg1;
    frag_b bu1;
    frag_c accg0;
    frag_c accu0;
    frag_c accg1;
    frag_c accu1;
    if (wave < MTILES) {
        rocwmma::fill_fragment(accg0, 0.0f);
        rocwmma::fill_fragment(accu0, 0.0f);
        rocwmma::fill_fragment(accg1, 0.0f);
        rocwmma::fill_fragment(accu1, 0.0f);
    }

    const unsigned char *gew = (const unsigned char *)gate_base + (uint64_t)expert * gate_expert_bytes;
    const unsigned char *uew = (const unsigned char *)up_base + (uint64_t)expert * gate_expert_bytes;
    for (uint32_t k0 = 0; k0 < expert_in_dim; k0 += BK) {
        if (X_F16) {
            for (uint32_t j = tid; j < MTILES * BM * (BK / 2); j += blockDim.x) {
                const uint32_t pair_row = j / (BK / 2);
                const uint32_t kk2 = j - pair_row * (BK / 2);
                const uint32_t pair = shPair[pair_row];
                uint32_t v = 0u;
                if (pair != UINT32_MAX) {
                    const uint32_t token = pair / n_expert;
                    const uint64_t xoff = (uint64_t)token * expert_in_dim + k0 + kk2 * 2u;
                    v = *reinterpret_cast<const uint32_t *>(x_h + xoff);
                }
                *reinterpret_cast<uint32_t *>(shA + pair_row * BK + kk2 * 2u) = v;
            }
        } else {
            for (uint32_t j = tid; j < MTILES * BM * BK; j += blockDim.x) {
                const uint32_t mt = j / (BM * BK);
                const uint32_t rem = j - mt * BM * BK;
                const uint32_t mm = rem / BK;
                const uint32_t kk = rem - mm * BK;
                const uint32_t pair = shPair[mt * BM + mm];
                if (pair != UINT32_MAX) {
                    const uint32_t token = pair / n_expert;
                    shA[j] = __float2half(x[(uint64_t)token * expert_in_dim + k0 + kk]);
                } else {
                    shA[j] = __float2half(0.0f);
                }
            }
        }
        q2_K_dequant_dual_pair_tile_half_rowwise<BN, BK>(
                shBg0, shBu0, shBg1, shBu1, gew, uew, gate_row_bytes, n0, k0, expert_mid_dim, tid);
        __syncthreads();
        if (wave < MTILES) {
            rocwmma::load_matrix_sync(a, shA + wave * BM * BK, BK);
            rocwmma::load_matrix_sync(bg0, shBg0, BN);
            rocwmma::load_matrix_sync(bu0, shBu0, BN);
            rocwmma::load_matrix_sync(bg1, shBg1, BN);
            rocwmma::load_matrix_sync(bu1, shBu1, BN);
            rocwmma::mma_sync(accg0, a, bg0, accg0);
            rocwmma::mma_sync(accu0, a, bu0, accu0);
            rocwmma::mma_sync(accg1, a, bg1, accg1);
            rocwmma::mma_sync(accu1, a, bu1, accu1);
        }
        __syncthreads();
    }

    if (wave < MTILES) {
        rocwmma::store_matrix_sync(shCg0 + wave * BM * BN, accg0, BN, rocwmma::mem_row_major);
        rocwmma::store_matrix_sync(shCu0 + wave * BM * BN, accu0, BN, rocwmma::mem_row_major);
        rocwmma::store_matrix_sync(shCg1 + wave * BM * BN, accg1, BN, rocwmma::mem_row_major);
        rocwmma::store_matrix_sync(shCu1 + wave * BM * BN, accu1, BN, rocwmma::mem_row_major);
    }
    __syncthreads();

    for (uint32_t j = tid; j < MTILES * BM * BN; j += blockDim.x) {
        const uint32_t mt = j / (BM * BN);
        const uint32_t rem = j - mt * BM * BN;
        const uint32_t mm = rem / BN;
        const uint32_t nn = rem - mm * BN;
        const uint32_t pair = shPair[mt * BM + mm];
        if (pair != UINT32_MAX) {
            const uint32_t row0 = n0 + nn;
            const uint32_t row1 = n0 + BN + nn;
            const float wt = weights[pair];
            if (row0 < expert_mid_dim) {
                float g = shCg0[j];
                float u = shCu0[j];
                if (clamp > 1.0e-6f) {
                    if (g > clamp) g = clamp;
                    if (u > clamp) u = clamp;
                    if (u < -clamp) u = -clamp;
                }
                const float v = moe_silu_oldhip(g) * u * wt;
                if (OUT_F16) mid_out_h[(uint64_t)pair * expert_mid_dim + row0] = __float2half(v);
                else mid_out[(uint64_t)pair * expert_mid_dim + row0] = v;
            }
            if (row1 < expert_mid_dim) {
                float g = shCg1[j];
                float u = shCu1[j];
                if (clamp > 1.0e-6f) {
                    if (g > clamp) g = clamp;
                    if (u > clamp) u = clamp;
                    if (u < -clamp) u = -clamp;
                }
                const float v = moe_silu_oldhip(g) * u * wt;
                if (OUT_F16) mid_out_h[(uint64_t)pair * expert_mid_dim + row1] = __float2half(v);
                else mid_out[(uint64_t)pair * expert_mid_dim + row1] = v;
            }
        }
    }
}

template <int MTILES=8, int BM=16, int BN=16, int BK=16>
__global__ static void moe_down_q2K_hotlist_wmma_kernel(
        float *down_out,
        const char *down_base,
        const float *mid,
        const uint32_t *counts,
        const uint32_t *offsets,
        const uint32_t *pairs,
        const uint32_t *hot_experts,
        uint32_t hot_count,
        uint32_t expert_mid_dim,
        uint32_t out_dim,
        uint64_t down_expert_bytes,
        uint64_t down_row_bytes) {
    extern __shared__ unsigned char raw_sh[];
    half *shA = reinterpret_cast<half *>(raw_sh);
    half *shB = shA + MTILES * BM * BK;
    float *shC = reinterpret_cast<float *>(shB + BK * BN);
    const uint32_t hot_idx = (uint32_t)blockIdx.z;
    if (hot_idx >= hot_count) return;
    const uint32_t expert = hot_experts[hot_idx];
    const uint32_t count = counts[expert];
    const uint32_t m_group0 = (uint32_t)blockIdx.y * MTILES * BM;
    if (m_group0 >= count) return;
    const uint32_t n0 = (uint32_t)blockIdx.x * BN;
    const uint32_t tid = threadIdx.x;
    const uint32_t wave = tid >> 5u;
    const uint32_t first = offsets[expert];

    using frag_a = rocwmma::fragment<rocwmma::matrix_a, BM, BN, BK, half, rocwmma::row_major>;
    using frag_b = rocwmma::fragment<rocwmma::matrix_b, BM, BN, BK, half, rocwmma::row_major>;
    using frag_c = rocwmma::fragment<rocwmma::accumulator, BM, BN, BK, float>;
    frag_a a;
    frag_b b;
    frag_c acc;
    if (wave < MTILES) rocwmma::fill_fragment(acc, 0.0f);

    const unsigned char *dew = (const unsigned char *)down_base + (uint64_t)expert * down_expert_bytes;
    for (uint32_t k0 = 0; k0 < expert_mid_dim; k0 += BK) {
        for (uint32_t j = tid; j < MTILES * BM * BK; j += blockDim.x) {
            const uint32_t mt = j / (BM * BK);
            const uint32_t rem = j - mt * BM * BK;
            const uint32_t mm = rem / BK;
            const uint32_t kk = rem - mm * BK;
            const uint32_t bucket_row = m_group0 + mt * BM + mm;
            if (bucket_row < count) {
                const uint32_t pair = pairs[first + bucket_row];
                shA[j] = __float2half(mid[(uint64_t)pair * expert_mid_dim + k0 + kk]);
            } else {
                shA[j] = __float2half(0.0f);
            }
        }
        q2_K_dequant_tile_half_rowwise<BN, BK>(
                shB, dew, down_row_bytes, n0, k0, out_dim, tid);
        __syncthreads();
        if (wave < MTILES) {
            rocwmma::load_matrix_sync(a, shA + wave * BM * BK, BK);
            rocwmma::load_matrix_sync(b, shB, BN);
            rocwmma::mma_sync(acc, a, b, acc);
        }
        __syncthreads();
    }

    if (wave < MTILES) rocwmma::store_matrix_sync(shC + wave * BM * BN, acc, BN, rocwmma::mem_row_major);
    __syncthreads();
    for (uint32_t j = tid; j < MTILES * BM * BN; j += blockDim.x) {
        const uint32_t mt = j / (BM * BN);
        const uint32_t rem = j - mt * BM * BN;
        const uint32_t mm = rem / BN;
        const uint32_t nn = rem - mm * BN;
        const uint32_t bucket_row = m_group0 + mt * BM + mm;
        const uint32_t row = n0 + nn;
        if (bucket_row < count && row < out_dim) {
            const uint32_t pair = pairs[first + bucket_row];
            down_out[(uint64_t)pair * out_dim + row] = shC[j];
        }
    }
}

/* One 32-weight IQ2_XXS group as stored: four grid indices, then four 7-bit
 * sign fields and the 4-bit group scale.  Groups start 2 bytes into a 66-byte
 * block, so the words are not 4-byte aligned. */
struct __attribute__((packed)) ds4_iq2_xxs_group {
    uint32_t idx;
    uint32_t aux;
};

/* Lane l of a wave32 reads the value of lane l ^ 16. */
__device__ __forceinline__ static uint32_t wave_swap_halves(uint32_t v) {
    return __builtin_amdgcn_permlanex16(v, v, 0x76543210u, 0xfedcba98u, false, false);
}

/* IQ2_XXS gate and up projections with their SwiGLU for resident prefill.
 * One block computes T*16 tokens of one expert against the same 256 rows of
 * both matrices and writes the weighted F16 mid rows the tiled down kernel
 * reads.
 *
 * Only the activations go through LDS (staged once per 64-wide K chunk and
 * read by all eight waves).  Every wave owns 32 rows of each matrix and
 * dequantizes them straight into the B operand: lane l needs 16 weights of
 * row l % 16, which is two grid entries read from an LDS table of exact F16
 * magnitudes, with the sign bits xored in.  A lane decodes one row tile and
 * takes the other of its pair from the opposite half of the wave, since the
 * instruction wants both halves to hold the same operand.  Each operand
 * then serves all T token tiles from registers.
 *
 * The group scale d*(2s+1)/8 is not folded into the operand.  It is constant
 * per weight row, hence per lane of an accumulator tile, so the accumulators
 * are kept in units of the current group's scale and rescaled by the ratio
 * when the group changes: eight multiplies per tile per 32 weights.
 *
 * Experts receive anything from one token to the whole batch.  The widest
 * tiles are bound by the WMMA rate.  Narrower ones are bound by reading the
 * expert's 4 MiB of weights, which every tile has to do once whatever its
 * width, so they fetch four groups of a row at a time: reading it group by
 * group touches every cache line eight times, and 256 rows per block do not
 * stay in the first-level cache in between.  The widest tile has no
 * registers left for that.
 * Operand layout of the w32 instruction: see attention_mixed_heads32. */
template <uint32_t T>
__global__ static void __launch_bounds__(256) moe_gate_up_iq2_tile_wmma_kernel(
        half *mid_h,
        const char *gate_base,
        const char *up_base,
        const half *x_h,
        const float *weights,
        const uint32_t *pairs,
        uint32_t in_dim,
        uint32_t mid_dim,
        uint64_t expert_bytes,
        uint32_t row_bytes,
        uint32_t n_expert,
        float clamp,
        const uint32_t *tiles) {
    constexpr uint32_t BM = T * 16u;
    constexpr uint32_t BN = 256u;                 /* rows of each matrix per block */
    constexpr uint32_t BK = 64u;
    constexpr uint32_t A_STRIDE = BK + 8u;        /* odd multiple of 16 bytes: no LDS bank conflicts */
    constexpr uint32_t WG = T <= 3u ? 4u : 1u;    /* groups per weight fetch */
    constexpr uint32_t SPAN = WG < 2u ? 2u : WG;  /* groups per unrolled loop body */
    const uint32_t expert = tiles[2u * blockIdx.y];
    const uint32_t first = tiles[2u * blockIdx.y + 1u] & 0x00ffffffu;
    const uint32_t count = (tiles[2u * blockIdx.y + 1u] >> 24u) + 1u;
    const uint32_t tid = threadIdx.x;
    const uint32_t wave = tid >> 5u;
    const uint32_t lane = tid & 31u;
    const uint32_t lane16 = lane & 15u;
    const uint32_t lane_hi = lane >> 4u;
    const uint32_t row0 = (uint32_t)blockIdx.x * BN + wave * 32u + lane16;

    __shared__ __align__(16) half shA[BM * A_STRIDE];
    __shared__ __align__(16) uint4 shGrid[256];

    {
        const uint64_t grid = cuda_iq2xxs_grid[tid];
        union {
            uint4 u;
            half h[8];
        } v;
#pragma unroll
        for (uint32_t j = 0; j < 8u; j++) v.h[j] = __float2half((float)((grid >> (8u * j)) & 0xffu));
        shGrid[tid] = v.u;
    }

    /* Thread tid stages 16 activations of token tid / 4 per chunk. */
    const uint32_t stage_row = tid >> 2u;
    const half *a_src = stage_row < count ?
        x_h + (uint64_t)(pairs[first + stage_row] / n_expert) * in_dim + (tid & 3u) * 16u : NULL;
    uint4 *a_dst = reinterpret_cast<uint4 *>(shA + (stage_row % BM) * A_STRIDE + (tid & 3u) * 16u);
    /* This lane's row of each matrix: the low half of the wave decodes the
     * first 16 rows and the high half the next 16. */
    const unsigned char *w_src[2];
    w_src[0] = (const unsigned char *)gate_base + (uint64_t)expert * expert_bytes +
               (uint64_t)(row0 + lane_hi * 16u) * row_bytes;
    w_src[1] = (const unsigned char *)up_base + (uint64_t)expert * expert_bytes +
               (uint64_t)(row0 + lane_hi * 16u) * row_bytes;

    /* Global loads run one step ahead of their use so their latency overlaps
     * the compute. */
    uint4 a_pre[2];
    ds4_iq2_xxs_group w_pre[2][WG], w_cur[2][WG];
    uint32_t d_pre[2];
    float d[2], scale[4];
    auto load_a = [&](uint32_t k0) {
        a_pre[0] = make_uint4(0u, 0u, 0u, 0u);
        a_pre[1] = a_pre[0];
        if (a_src && stage_row < BM) {
            const uint4 *src = reinterpret_cast<const uint4 *>(a_src + k0);
            a_pre[0] = src[0];
            a_pre[1] = src[1];
        }
    };
    auto load_w = [&](uint32_t k0) {
#pragma unroll
        for (uint32_t m = 0; m < 2u; m++) {
            const unsigned char *blk = w_src[m] + (k0 >> 8u) * 66u;
            if ((k0 & 255u) == 0u) d_pre[m] = *reinterpret_cast<const uint16_t *>(blk);
#pragma unroll
            for (uint32_t i = 0; i < WG; i++) {
                w_pre[m][i] = *reinterpret_cast<const ds4_iq2_xxs_group *>(
                        blk + 2u + (((k0 & 255u) >> 5u) + i) * 8u);
            }
        }
    };
    load_a(0u);
    load_w(0u);

    /* acc[2 * m + h]: matrix m, row tile h of this wave. */
    ds4_q8_float8_t acc[4][T];
#pragma unroll
    for (uint32_t rt = 0; rt < 4u; rt++) {
        scale[rt] = 1.0f;
#pragma unroll
        for (uint32_t t = 0; t < T; t++) {
#pragma unroll
            for (uint32_t j = 0; j < 8u; j++) acc[rt][t][j] = 0.0f;
        }
    }

    for (uint32_t kb = 0; kb < in_dim; kb += SPAN * 32u) {
#pragma unroll
        for (uint32_t gi = 0; gi < SPAN; gi++) {
            const uint32_t kg = kb + gi * 32u;
            if ((gi & 1u) == 0u) {
                __syncthreads();
                if (stage_row < BM) {
                    a_dst[0] = a_pre[0];
                    a_dst[1] = a_pre[1];
                }
                __syncthreads();
                if (kg + BK < in_dim) load_a(kg + BK);
            }
            if (gi % WG == 0u) {
#pragma unroll
                for (uint32_t m = 0; m < 2u; m++) {
                    if ((kg & 255u) == 0u) d[m] = dev_f16_to_f32((uint16_t)d_pre[m]);
#pragma unroll
                    for (uint32_t i = 0; i < WG; i++) w_cur[m][i] = w_pre[m][i];
                }
                if (kg + WG * 32u < in_dim) load_w(kg + WG * 32u);
            }
#pragma unroll
            for (uint32_t m = 0; m < 2u; m++) {
                /* A zero block scale contributes nothing either way; keep
                 * the ratio finite. */
                float s = d[m] * (float)(2u * (w_cur[m][gi % WG].aux >> 28u) + 1u) * 0.125f;
                if (s == 0.0f) s = 1.0e-20f;
                const float s_other = __uint_as_float(wave_swap_halves(__float_as_uint(s)));
#pragma unroll
                for (uint32_t h = 0; h < 2u; h++) {
                    const float s_rt = lane_hi == h ? s : s_other;
                    const float ratio = scale[2u * m + h] / s_rt;
                    scale[2u * m + h] = s_rt;
#pragma unroll
                    for (uint32_t t = 0; t < T; t++) {
#pragma unroll
                        for (uint32_t j = 0; j < 8u; j++) acc[2u * m + h][t][j] *= ratio;
                    }
                }
            }
#pragma unroll
            for (uint32_t ks = 0; ks < 2u; ks++) {
                ds4_q8_half16_t a[T];
#pragma unroll
                for (uint32_t t = 0; t < T; t++) {
                    a[t] = lds_load_half16(shA + (t * 16u + lane16) * A_STRIDE + (gi & 1u) * 32u + ks * 16u);
                }
#pragma unroll
                for (uint32_t m = 0; m < 2u; m++) {
                    const ds4_iq2_xxs_group w = w_cur[m][gi % WG];
                    uint32_t own[8], other[8];
#pragma unroll
                    for (uint32_t l = 0; l < 2u; l++) {
                        const uint4 mag = shGrid[(w.idx >> (16u * ks + 8u * l)) & 0xffu];
                        /* Seven stored sign bits plus their parity, spread
                         * to the sign bits of the eight halves: bits 2j and
                         * 2j+1 land on bits 15 and 31 of word j. */
                        const uint32_t f = (w.aux >> (14u * ks + 7u * l)) & 0x7fu;
                        const uint32_t s8 = f | ((__popc(f) & 1u) << 7u);
                        own[4u * l + 0u] = mag.x ^ ((s8 * 0x40008000u) & 0x80008000u);
                        own[4u * l + 1u] = mag.y ^ (((s8 >> 2u) * 0x40008000u) & 0x80008000u);
                        own[4u * l + 2u] = mag.z ^ (((s8 >> 4u) * 0x40008000u) & 0x80008000u);
                        own[4u * l + 3u] = mag.w ^ (((s8 >> 6u) * 0x40008000u) & 0x80008000u);
                    }
                    /* The swap has to run on every lane: it stays outside
                     * the select, whose arms only half the lanes evaluate. */
#pragma unroll
                    for (uint32_t i = 0; i < 8u; i++) other[i] = wave_swap_halves(own[i]);
#pragma unroll
                    for (uint32_t h = 0; h < 2u; h++) {
                        union {
                            uint32_t u[8];
                            ds4_q8_half16_t v;
                        } b;
#pragma unroll
                        for (uint32_t i = 0; i < 8u; i++) b.u[i] = lane_hi == h ? own[i] : other[i];
#pragma unroll
                        for (uint32_t t = 0; t < T; t++) {
                            acc[2u * m + h][t] =
                                __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a[t], b.v, acc[2u * m + h][t]);
                        }
                    }
                }
            }
        }
    }

#pragma unroll
    for (uint32_t t = 0; t < T; t++) {
#pragma unroll
        for (uint32_t j = 0; j < 8u; j++) {
            const uint32_t tok = t * 16u + 2u * j + lane_hi;
            if (tok >= count) continue;
            const uint32_t pair = pairs[first + tok];
            const float weight = weights[pair];
#pragma unroll
            for (uint32_t h = 0; h < 2u; h++) {
                /* Same SwiGLU as moe_swiglu_weighted_f32_kernel. */
                float g = acc[h][t][j] * scale[h];
                float u = acc[2u + h][t][j] * scale[2u + h];
                g = isfinite(g) ? g : 0.0f;
                u = isfinite(u) ? u : 0.0f;
                if (clamp > 1.0e-6f) {
                    g = fminf(g, clamp);
                    u = fminf(fmaxf(u, -clamp), clamp);
                }
                mid_h[(uint64_t)pair * mid_dim + row0 + h * 16u] =
                    __float2half((g / (1.0f + expf(-g))) * u * weight);
            }
        }
    }
}

/* Half of a Q2_K block of one weight row, as moe_down_q2K_tile_wmma_kernel
 * fetches it: the eight sub-block scale bytes, the 32 code bytes they share
 * and the block scales.  Sub-block w of the half reads its 16 two-bit codes
 * from bit 2 * (w / 2) of the 16 bytes at qs + 16 * (w & 1). */
struct ds4_q2k_half {
    uint32_t scales[2];
    uint32_t qs[8];
    uint32_t d;         /* d | dmin << 16, both F16 */
};

/* Q2_K down projection for resident prefill, on the tiles and the layout of
 * moe_gate_up_iq2_tile_wmma_kernel: one block computes T*16 pairs of one
 * expert against ROW_TILES * THREADS / 2 output rows. F16 mid rows pass
 * through LDS once per 64-wide K chunk; every wave dequantizes its own
 * ROW_TILES * 16 rows into the B operand,
 * one lane per row with the other half of each tile pair taken from the
 * opposite half of the wave.  A sub-block is one 16-wide WMMA step, so its
 * scale and minimum are folded into the 16 weights the lane converts.
 *
 * Weights are fetched half a block of a row at a time, which reads every
 * cache line once. Two row tiles leave enough registers to prefetch weights
 * even at the widest token tile; four row tiles keep the original schedule. */
template <uint32_t T, uint32_t ROW_TILES = 4u, uint32_t THREADS = 256u>
__global__ static void __launch_bounds__(THREADS) moe_down_q2K_tile_wmma_kernel(
        half *down_h,
        const char *down_base,
        const half *mid_h,
        const uint32_t *pairs,
        uint32_t mid_dim,
        uint32_t out_dim,
        uint64_t expert_bytes,
        uint32_t row_bytes,
        const uint32_t *tiles) {
    constexpr uint32_t BM = T * 16u;
    constexpr uint32_t BN = ROW_TILES * (THREADS / 2u); /* rows per block */
    constexpr uint32_t BK = 64u;
    constexpr uint32_t A_STRIDE = BK + 8u;        /* odd multiple of 16 bytes: no LDS bank conflicts */
    constexpr bool W_AHEAD = ROW_TILES == 2u || T <= 2u;
    const uint32_t expert = tiles[2u * blockIdx.y];
    const uint32_t first = tiles[2u * blockIdx.y + 1u] & 0x00ffffffu;
    const uint32_t count = (tiles[2u * blockIdx.y + 1u] >> 24u) + 1u;
    const uint32_t tid = threadIdx.x;
    const uint32_t wave = tid >> 5u;
    const uint32_t lane = tid & 31u;
    const uint32_t lane16 = lane & 15u;
    const uint32_t lane_hi = lane >> 4u;
    const uint32_t row0 = (uint32_t)blockIdx.x * BN + wave * (ROW_TILES * 16u) + lane16;

    __shared__ __align__(16) half shA[BM * A_STRIDE];

    /* Two or four threads stage each pair, keeping adjacent 16-byte loads
     * coalesced even with the smaller block. */
    constexpr uint32_t STAGE_COLS = THREADS / 64u;
    constexpr uint32_t STAGE_VECTORS = 8u / STAGE_COLS;
    const uint32_t stage_row = tid / STAGE_COLS;
    const half *a_src = stage_row < count ?
        mid_h + (uint64_t)pairs[first + stage_row] * mid_dim + (tid % STAGE_COLS) * (BK / STAGE_COLS) : NULL;
    uint4 *a_dst = reinterpret_cast<uint4 *>(shA + (stage_row % BM) * A_STRIDE + (tid % STAGE_COLS) * (BK / STAGE_COLS));
    /* This lane's row of the wave's first tile pair; the second pair is 32
     * rows on. */
    const unsigned char *w_src = (const unsigned char *)down_base + (uint64_t)expert * expert_bytes +
                                 (uint64_t)(row0 + lane_hi * 16u) * row_bytes;

    uint4 a_pre[STAGE_VECTORS];
    ds4_q2k_half w_pre[ROW_TILES / 2u], w_cur[ROW_TILES / 2u];
    auto load_a = [&](uint32_t k0) {
#pragma unroll
        for (uint32_t i = 0; i < STAGE_VECTORS; i++) a_pre[i] = make_uint4(0u, 0u, 0u, 0u);
        if (a_src && stage_row < BM) {
            const uint4 *src = reinterpret_cast<const uint4 *>(a_src + k0);
#pragma unroll
            for (uint32_t i = 0; i < STAGE_VECTORS; i++) a_pre[i] = src[i];
        }
    };
    auto load_w = [&](ds4_q2k_half *w, uint32_t k0) {
#pragma unroll
        for (uint32_t p = 0; p < ROW_TILES / 2u; p++) {
            const uint32_t *blk = reinterpret_cast<const uint32_t *>(
                    w_src + p * 32u * row_bytes + (k0 >> 8u) * 84u);
            const uint32_t half = (k0 >> 7u) & 1u;
#pragma unroll
            for (uint32_t i = 0; i < 2u; i++) w[p].scales[i] = blk[half * 2u + i];
#pragma unroll
            for (uint32_t i = 0; i < 8u; i++) w[p].qs[i] = blk[4u + half * 8u + i];
            w[p].d = blk[20];
        }
    };
    load_a(0u);
    if (W_AHEAD) load_w(w_pre, 0u);

    ds4_q8_float8_t acc[ROW_TILES][T];
#pragma unroll
    for (uint32_t rt = 0; rt < ROW_TILES; rt++) {
#pragma unroll
        for (uint32_t t = 0; t < T; t++) {
#pragma unroll
            for (uint32_t j = 0; j < 8u; j++) acc[rt][t][j] = 0.0f;
        }
    }

    for (uint32_t kb = 0; kb < mid_dim; kb += 128u) {
        if (W_AHEAD) {
#pragma unroll
            for (uint32_t p = 0; p < ROW_TILES / 2u; p++) w_cur[p] = w_pre[p];
            if (kb + 128u < mid_dim) load_w(w_pre, kb + 128u);
        } else {
            load_w(w_cur, kb);
        }
#pragma unroll
        for (uint32_t w = 0; w < 8u; w++) {
            if ((w & 3u) == 0u) {
                const uint32_t k0 = kb + w * 16u;
                __syncthreads();
                if (stage_row < BM) {
#pragma unroll
                    for (uint32_t i = 0; i < STAGE_VECTORS; i++) a_dst[i] = a_pre[i];
                }
                __syncthreads();
                if (k0 + BK < mid_dim) load_a(k0 + BK);
            }
            ds4_q8_half16_t a[T];
#pragma unroll
            for (uint32_t t = 0; t < T; t++) {
                a[t] = lds_load_half16(shA + (t * 16u + lane16) * A_STRIDE + (w & 3u) * 16u);
            }
#pragma unroll
            for (uint32_t p = 0; p < ROW_TILES / 2u; p++) {
                const uint32_t sc = (w_cur[p].scales[w >> 2u] >> (8u * (w & 3u))) & 0xffu;
                const float ds = dev_f16_to_f32((uint16_t)w_cur[p].d) * (float)(sc & 0x0fu);
                const float dmm = dev_f16_to_f32((uint16_t)(w_cur[p].d >> 16u)) * (float)(sc >> 4u);
                union {
                    uint32_t u[8];
                    ds4_q8_half16_t v;
                } own;
                uint32_t other[8];
#pragma unroll
                for (uint32_t i = 0; i < 16u; i++) {
                    const uint32_t q = (w_cur[p].qs[(w & 1u) * 4u + (i >> 2u)] >> (8u * (i & 3u) + (w >> 1u) * 2u)) & 3u;
                    own.v[i] = (_Float16)(ds * (float)q - dmm);
                }
                /* The swap has to run on every lane: it stays outside the
                 * select, whose arms only half the lanes evaluate. */
#pragma unroll
                for (uint32_t i = 0; i < 8u; i++) other[i] = wave_swap_halves(own.u[i]);
#pragma unroll
                for (uint32_t h = 0; h < 2u; h++) {
                    union {
                        uint32_t u[8];
                        ds4_q8_half16_t v;
                    } b;
#pragma unroll
                    for (uint32_t i = 0; i < 8u; i++) b.u[i] = lane_hi == h ? own.u[i] : other[i];
#pragma unroll
                    for (uint32_t t = 0; t < T; t++) {
                        acc[2u * p + h][t] =
                            __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a[t], b.v, acc[2u * p + h][t]);
                    }
                }
            }
        }
    }

#pragma unroll
    for (uint32_t t = 0; t < T; t++) {
#pragma unroll
        for (uint32_t j = 0; j < 8u; j++) {
            const uint32_t tok = t * 16u + 2u * j + lane_hi;
            if (tok >= count) continue;
            const uint32_t pair = pairs[first + tok];
#pragma unroll
            for (uint32_t rt = 0; rt < ROW_TILES; rt++) {
                down_h[(uint64_t)pair * out_dim + row0 + rt * 16u] = __float2half(acc[rt][t][j]);
            }
        }
    }
}

template <int MTILES=8, int BM=16, int BN=16, int BK=16, bool MID_F16=false, bool OUT_F16=false, bool SLOT_MAJOR=false>
__global__ static void moe_down_q2K_hotlist_wmma_n2_kernel(
        float *down_out,
        half *down_out_h,
        const char *down_base,
        const float *mid,
        const half *mid_h,
        const uint32_t *counts,
        const uint32_t *offsets,
        const uint32_t *pairs,
        const uint32_t *hot_experts,
        uint32_t hot_count,
        uint32_t expert_mid_dim,
        uint32_t out_dim,
        uint64_t down_expert_bytes,
        uint64_t down_row_bytes,
        uint32_t n_expert,
        uint32_t n_tokens = 0u) {
    extern __shared__ unsigned char raw_sh[];
    half *shA = reinterpret_cast<half *>(raw_sh);
    half *shB0 = shA + MTILES * BM * BK;
    half *shB1 = shB0 + BK * BN;
    constexpr uint32_t RAW_DWORDS = 84u / sizeof(uint32_t);
    constexpr uint32_t RAW_ROWS = 2u * BN;
    uint32_t *shW = reinterpret_cast<uint32_t *>(shB1 + BK * BN);
    float *shC = reinterpret_cast<float *>(shW + RAW_ROWS * RAW_DWORDS);
    const uint32_t hot_idx = (uint32_t)blockIdx.z;
    if (hot_idx >= hot_count) return;
    const uint32_t expert = hot_experts[hot_idx];
    const uint32_t count = counts[expert];
    const uint32_t m_group0 = (uint32_t)blockIdx.y * MTILES * BM;
    if (m_group0 >= count) return;
    const uint32_t n0 = (uint32_t)blockIdx.x * (2u * BN);
    const uint32_t tid = threadIdx.x;
    const uint32_t wave = tid >> 5u;
    const uint32_t first = offsets[expert];
    __shared__ uint32_t shPair[MTILES * BM];
    for (uint32_t j = tid; j < MTILES * BM; j += blockDim.x) {
        const uint32_t bucket_row = m_group0 + j;
        shPair[j] = (bucket_row < count) ? pairs[first + bucket_row] : UINT32_MAX;
    }
    __syncthreads();

    using frag_a = rocwmma::fragment<rocwmma::matrix_a, BM, BN, BK, half, rocwmma::row_major>;
    using frag_b = rocwmma::fragment<rocwmma::matrix_b, BM, BN, BK, half, rocwmma::col_major>;
    using frag_c = rocwmma::fragment<rocwmma::accumulator, BM, BN, BK, float>;
    frag_a a;
    frag_b b0;
    frag_b b1;
    frag_c acc0;
    frag_c acc1;
    if (wave < MTILES) {
        rocwmma::fill_fragment(acc0, 0.0f);
        rocwmma::fill_fragment(acc1, 0.0f);
    }

    const unsigned char *dew = (const unsigned char *)down_base + (uint64_t)expert * down_expert_bytes;
    for (uint32_t kb = 0; kb < expert_mid_dim; kb += 256u) {
        for (uint32_t j = tid; j < RAW_ROWS * RAW_DWORDS; j += blockDim.x) {
            const uint32_t row_local = j / RAW_DWORDS;
            const uint32_t word = j - row_local * RAW_DWORDS;
            const uint32_t row = n0 + row_local;
            uint32_t v = 0u;
            if (row < out_dim) {
                const unsigned char *blk = dew + (uint64_t)row * down_row_bytes +
                                           (uint64_t)(kb >> 8u) * 84u;
                v = *reinterpret_cast<const uint32_t *>(blk + word * sizeof(uint32_t));
            }
            shW[j] = v;
        }
        __syncthreads();

        for (uint32_t krel = 0; krel < 256u && kb + krel < expert_mid_dim; krel += BK) {
            const uint32_t k0 = kb + krel;
            if (MID_F16) {
                for (uint32_t j = tid; j < MTILES * BM * (BK / 2); j += blockDim.x) {
                    const uint32_t pair_row = j / (BK / 2);
                    const uint32_t kk2 = j - pair_row * (BK / 2);
                    const uint32_t pair = shPair[pair_row];
                    uint32_t v = 0u;
                    if (pair != UINT32_MAX) {
                        const uint64_t moff = (uint64_t)pair * expert_mid_dim + k0 + kk2 * 2u;
                        v = *reinterpret_cast<const uint32_t *>(mid_h + moff);
                    }
                    *reinterpret_cast<uint32_t *>(shA + pair_row * BK + kk2 * 2u) = v;
                }
            } else {
                for (uint32_t j = tid; j < MTILES * BM * BK; j += blockDim.x) {
                    const uint32_t mt = j / (BM * BK);
                    const uint32_t rem = j - mt * BM * BK;
                    const uint32_t mm = rem / BK;
                    const uint32_t kk = rem - mm * BK;
                    const uint32_t pair = shPair[mt * BM + mm];
                    if (pair != UINT32_MAX) {
                        shA[j] = __float2half(mid[(uint64_t)pair * expert_mid_dim + k0 + kk]);
                    } else {
                        shA[j] = __float2half(0.0f);
                    }
                }
            }
            q2_K_dequant_pair_tile_half_rowwise_staged<BN, BK>(
                    shB0, shB1, shW, krel, tid);
            __syncthreads();
            if (wave < MTILES) {
                rocwmma::load_matrix_sync(a, shA + wave * BM * BK, BK);
                rocwmma::load_matrix_sync(b0, shB0, BN);
                rocwmma::load_matrix_sync(b1, shB1, BN);
                rocwmma::mma_sync(acc0, a, b0, acc0);
                rocwmma::mma_sync(acc1, a, b1, acc1);
            }
            __syncthreads();
        }
    }

    if (wave < MTILES) {
        rocwmma::store_matrix_sync(shC + wave * BM * BN, acc0, BN, rocwmma::mem_row_major);
    }
    __syncthreads();
    for (uint32_t j = tid; j < MTILES * BM * BN; j += blockDim.x) {
        const uint32_t mt = j / (BM * BN);
        const uint32_t rem = j - mt * BM * BN;
        const uint32_t mm = rem / BN;
        const uint32_t nn = rem - mm * BN;
        const uint32_t pair = shPair[mt * BM + mm];
        if (pair != UINT32_MAX) {
            const uint32_t row0 = n0 + nn;
            const uint32_t tok = pair / n_expert;
            const uint32_t slot = pair - tok * n_expert;
            if (row0 < out_dim) {
                if (OUT_F16) {
                    uint64_t dst = (uint64_t)pair * out_dim + row0;
                    if (SLOT_MAJOR) dst = ((uint64_t)slot * n_tokens + tok) * out_dim + row0;
                    down_out_h[dst] = __float2half(shC[j]);
                } else {
                    uint64_t dst = (uint64_t)pair * out_dim + row0;
                    if (SLOT_MAJOR) dst = ((uint64_t)slot * n_tokens + tok) * out_dim + row0;
                    down_out[dst] = shC[j];
                }
            }
        }
    }
    __syncthreads();

    if (wave < MTILES) {
        rocwmma::store_matrix_sync(shC + wave * BM * BN, acc1, BN, rocwmma::mem_row_major);
    }
    __syncthreads();
    for (uint32_t j = tid; j < MTILES * BM * BN; j += blockDim.x) {
        const uint32_t mt = j / (BM * BN);
        const uint32_t rem = j - mt * BM * BN;
        const uint32_t mm = rem / BN;
        const uint32_t nn = rem - mm * BN;
        const uint32_t pair = shPair[mt * BM + mm];
        if (pair != UINT32_MAX) {
            const uint32_t row1 = n0 + BN + nn;
            const uint32_t tok = pair / n_expert;
            const uint32_t slot = pair - tok * n_expert;
            if (row1 < out_dim) {
                if (OUT_F16) {
                    uint64_t dst = (uint64_t)pair * out_dim + row1;
                    if (SLOT_MAJOR) dst = ((uint64_t)slot * n_tokens + tok) * out_dim + row1;
                    down_out_h[dst] = __float2half(shC[j]);
                } else {
                    uint64_t dst = (uint64_t)pair * out_dim + row1;
                    if (SLOT_MAJOR) dst = ((uint64_t)slot * n_tokens + tok) * out_dim + row1;
                    down_out[dst] = shC[j];
                }
            }
        }
    }
}

#endif

__device__ static float dev_iq2_xxs_dot_f32(const cuda_block_iq2_xxs *row, const float *x, uint32_t nb) {
    float acc = 0.0f;
    for (uint32_t b = 0; b < nb; b++) {
        const cuda_block_iq2_xxs *xb = row + b;
        const float d = dev_f16_to_f32(xb->d);
        const uint16_t *q2 = xb->qs;
        const float *xf = x + (uint64_t)b * CUDA_QK_K;
        for (uint32_t ib32 = 0; ib32 < CUDA_QK_K / 32; ib32++) {
            const uint32_t aux_g = (uint32_t)q2[0] | ((uint32_t)q2[1] << 16);
            const uint32_t aux_s = (uint32_t)q2[2] | ((uint32_t)q2[3] << 16);
            q2 += 4;
            const float dl = d * (0.5f + (float)(aux_s >> 28)) * 0.25f;
            const uint8_t grids[4] = {
                (uint8_t)(aux_g & 0xffu),
                (uint8_t)((aux_g >> 8) & 0xffu),
                (uint8_t)((aux_g >> 16) & 0xffu),
                (uint8_t)((aux_g >> 24) & 0xffu),
            };
            for (uint32_t half = 0; half < 2; half++) {
                for (uint32_t g = 0; g < 2; g++) {
                    const uint32_t gi = half * 2 + g;
                    const uint64_t grid = cuda_iq2xxs_grid[grids[gi]];
                    const uint8_t signs = cuda_ksigns_iq2xs[(aux_s >> (14u * half + 7u * g)) & 127u];
                    for (uint32_t i = 0; i < 8; i++) {
                        float w = (float)((grid >> (8u * i)) & 0xffu);
                        if (signs & (1u << i)) w = -w;
                        acc += dl * w * xf[ib32 * 32u + half * 16u + g * 8u + i];
                    }
                }
            }
        }
    }
    return acc;
}

__device__ static float dev_q2_K_dot_f32(const cuda_block_q2_K *row, const float *x, uint32_t nb) {
    float acc = 0.0f;
    for (uint32_t b = 0; b < nb; b++) {
        const cuda_block_q2_K *xb = row + b;
        const float d = dev_f16_to_f32(xb->d);
        const float dmin = dev_f16_to_f32(xb->dmin);
        for (uint32_t il = 0; il < 16; il++) {
            const uint32_t chunk = il / 8u;
            const uint32_t pair = il & 1u;
            const uint32_t shift = ((il / 2u) & 3u) * 2u;
            const uint8_t sc = xb->scales[il];
            const float dl = d * (float)(sc & 0x0fu);
            const float ml = dmin * (float)(sc >> 4);
            const uint8_t *q = xb->qs + 32u * chunk + 16u * pair;
            const float *xf = x + (uint64_t)b * CUDA_QK_K + chunk * 128u + ((il % 8u) / 2u) * 32u + pair * 16u;
            for (uint32_t i = 0; i < 16; i++) {
                const float w = dl * (float)((q[i] >> shift) & 3u) - ml;
                acc += w * xf[i];
            }
        }
    }
    return acc;
}

__global__ static void moe_gate_up_mid_f32_kernel(
        float *gate_out,
        float *up_out,
        float *mid_out,
        const char *gate_base,
        const char *up_base,
        const float *x,
        const int32_t *selected,
        const float *weights,
        uint64_t gate_expert_bytes,
        uint64_t gate_row_bytes,
        uint32_t expert_in_dim,
        uint32_t expert_mid_dim,
        uint32_t n_expert,
        float clamp) {
    uint32_t row = blockIdx.x;
    uint32_t pair = blockIdx.y;
    if (row >= expert_mid_dim) return;
    uint32_t tok = pair / n_expert;
    uint32_t slot = pair - tok * n_expert;
    int32_t expert_i = selected[(uint64_t)tok * n_expert + slot];
    if (expert_i < 0) expert_i = 0;
    uint32_t expert = (uint32_t)expert_i;
    const uint32_t nb = expert_in_dim / CUDA_QK_K;
    const cuda_block_iq2_xxs *gr = (const cuda_block_iq2_xxs *)(gate_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes);
    const cuda_block_iq2_xxs *ur = (const cuda_block_iq2_xxs *)(up_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes);
    const float *xr = x + (uint64_t)tok * expert_in_dim;
    float gate = 0.0f;
    float up = 0.0f;
    for (uint32_t b = threadIdx.x; b < nb; b += blockDim.x) {
        gate += dev_iq2_xxs_dot_f32(gr + b, xr + (uint64_t)b * CUDA_QK_K, 1);
        up += dev_iq2_xxs_dot_f32(ur + b, xr + (uint64_t)b * CUDA_QK_K, 1);
    }
    __shared__ float partial_gate[256];
    __shared__ float partial_up[256];
    partial_gate[threadIdx.x] = gate;
    partial_up[threadIdx.x] = up;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            partial_gate[threadIdx.x] += partial_gate[threadIdx.x + stride];
            partial_up[threadIdx.x] += partial_up[threadIdx.x + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        gate = partial_gate[0];
        up = partial_up[0];
        if (clamp > 1.0e-6f) {
            if (gate > clamp) gate = clamp;
            if (up > clamp) up = clamp;
            if (up < -clamp) up = -clamp;
        }
        const uint64_t off = (uint64_t)pair * expert_mid_dim + row;
        gate_out[off] = gate;
        up_out[off] = up;
        mid_out[off] = (gate / (1.0f + expf(-gate))) * up * weights[(uint64_t)tok * n_expert + slot];
    }
}

__global__ static void moe_gate_up_mid_q2K_f32_kernel(
        float *gate_out,
        float *up_out,
        float *mid_out,
        const char *gate_base,
        const char *up_base,
        const float *x,
        const int32_t *selected,
        const float *weights,
        uint64_t gate_expert_bytes,
        uint64_t gate_row_bytes,
        uint32_t expert_in_dim,
        uint32_t expert_mid_dim,
        uint32_t n_expert,
        float clamp) {
    uint32_t row = blockIdx.x;
    uint32_t pair = blockIdx.y;
    if (row >= expert_mid_dim) return;
    uint32_t tok = pair / n_expert;
    uint32_t slot = pair - tok * n_expert;
    int32_t expert_i = selected[(uint64_t)tok * n_expert + slot];
    if (expert_i < 0) expert_i = 0;
    uint32_t expert = (uint32_t)expert_i;
    const uint32_t nb = expert_in_dim / CUDA_QK_K;
    const cuda_block_q2_K *gr = (const cuda_block_q2_K *)(gate_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes);
    const cuda_block_q2_K *ur = (const cuda_block_q2_K *)(up_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes);
    const float *xr = x + (uint64_t)tok * expert_in_dim;
    float gate = 0.0f;
    float up = 0.0f;
    for (uint32_t b = threadIdx.x; b < nb; b += blockDim.x) {
        gate += dev_q2_K_dot_f32(gr + b, xr + (uint64_t)b * CUDA_QK_K, 1);
        up += dev_q2_K_dot_f32(ur + b, xr + (uint64_t)b * CUDA_QK_K, 1);
    }
    __shared__ float partial_gate[256];
    __shared__ float partial_up[256];
    partial_gate[threadIdx.x] = gate;
    partial_up[threadIdx.x] = up;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            partial_gate[threadIdx.x] += partial_gate[threadIdx.x + stride];
            partial_up[threadIdx.x] += partial_up[threadIdx.x + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        gate = partial_gate[0];
        up = partial_up[0];
        if (clamp > 1.0e-6f) {
            if (gate > clamp) gate = clamp;
            if (up > clamp) up = clamp;
            if (up < -clamp) up = -clamp;
        }
        const uint64_t off = (uint64_t)pair * expert_mid_dim + row;
        gate_out[off] = gate;
        up_out[off] = up;
        mid_out[off] = (gate / (1.0f + expf(-gate))) * up * weights[(uint64_t)tok * n_expert + slot];
    }
}

__global__ static void moe_down_f32_kernel(
        float *down_out,
        const char *down_base,
        const float *mid,
        const int32_t *selected,
        uint64_t down_expert_bytes,
        uint64_t down_row_bytes,
        uint32_t expert_mid_dim,
        uint32_t out_dim,
        uint32_t n_expert) {
    uint32_t row = blockIdx.x;
    uint32_t pair = blockIdx.y;
    if (row >= out_dim) return;
    uint32_t tok = pair / n_expert;
    uint32_t slot = pair - tok * n_expert;
    int32_t expert_i = selected[(uint64_t)tok * n_expert + slot];
    if (expert_i < 0) expert_i = 0;
    const uint32_t nb = expert_mid_dim / CUDA_QK_K;
    const cuda_block_q2_K *wr = (const cuda_block_q2_K *)(down_base + (uint64_t)(uint32_t)expert_i * down_expert_bytes + (uint64_t)row * down_row_bytes);
    const float *xr = mid + (uint64_t)pair * expert_mid_dim;
    float acc = 0.0f;
    for (uint32_t b = threadIdx.x; b < nb; b += blockDim.x) acc += dev_q2_K_dot_f32(wr + b, xr + (uint64_t)b * CUDA_QK_K, 1);
    __shared__ float partial[256];
    partial[threadIdx.x] = acc;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) down_out[(uint64_t)pair * out_dim + row] = partial[0];
}
