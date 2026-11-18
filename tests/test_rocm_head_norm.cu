/* Check the production fused Q projection/norm/RoPE against the original
 * 256-thread norm on the exact same GEMM output, including partial blocks. */
#include "ds4_gpu.h"
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <algorithm>
#define CHECK(x) do { if (!(x)) { fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #x); exit(1); } } while (0)
bool ds4_log_is_tty(FILE *) { return false; }
__device__ static float rope_yarn_ramp_dev(float low, float high, int i0);

__global__ static void reference_norm_rope(
        float *out,
        const __half *x,
        uint32_t n_tok,
        uint32_t n_head,
        uint32_t head_dim,
        uint32_t n_rot,
        uint32_t pos0,
        uint32_t n_ctx_orig,
        int inverse,
        float freq_base,
        float freq_scale,
        float ext_factor,
        float attn_factor,
        float beta_fast,
        float beta_slow,
        float eps) {
    uint32_t row = blockIdx.x;
    if (row >= n_tok * n_head) return;
    uint32_t t = row / n_head;
    const __half *xr = x + (uint64_t)row * head_dim;
    float *orow = out + (uint64_t)row * head_dim;
    float sum = 0.0f;
    for (uint32_t i = threadIdx.x; i < head_dim; i += blockDim.x) {
        float v = __half2float(xr[i]);
        sum += v * v;
    }
    __shared__ float partial[256];
    partial[threadIdx.x] = sum;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
        __syncthreads();
    }
    const float scale = rsqrtf(partial[0] / (float)head_dim + eps);
    const uint32_t n_nope = head_dim - n_rot;
    for (uint32_t i = threadIdx.x; i < n_nope; i += blockDim.x) {
        orow[i] = __half2float(xr[i]) * scale;
    }

    float corr0 = 0.0f, corr1 = 0.0f;
    if (ext_factor != 0.0f) {
        float denom = 2.0f * logf(freq_base);
        corr0 = floorf((float)n_rot * logf((float)n_ctx_orig / (beta_fast * 2.0f * (float)M_PI)) / denom);
        corr1 = ceilf((float)n_rot * logf((float)n_ctx_orig / (beta_slow * 2.0f * (float)M_PI)) / denom);
        corr0 = fmaxf(0.0f, corr0);
        corr1 = fminf((float)(n_rot - 1), corr1);
    }
    const float theta_scale = powf(freq_base, -2.0f / (float)n_rot);
    const __half *tail = xr + n_nope;
    float *otail = orow + n_nope;
    for (uint32_t pair = threadIdx.x; pair < n_rot / 2; pair += blockDim.x) {
        uint32_t i = pair * 2u;
        float theta_extrap = (float)(pos0 + t) * powf(theta_scale, (float)pair);
        float theta_interp = freq_scale * theta_extrap;
        float theta = theta_interp;
        float mscale = attn_factor;
        if (ext_factor != 0.0f) {
            float ramp_mix = rope_yarn_ramp_dev(corr0, corr1, (int)i) * ext_factor;
            theta = theta_interp * (1.0f - ramp_mix) + theta_extrap * ramp_mix;
            mscale *= 1.0f + 0.1f * logf(1.0f / freq_scale);
        }
        float c = cosf(theta) * mscale;
        float s = sinf(theta) * mscale;
        if (inverse) s = -s;
        float x0 = __half2float(tail[i]) * scale;
        float x1 = __half2float(tail[i + 1]) * scale;
        otail[i] = x0 * c - x1 * s;
        otail[i + 1] = x0 * s + x1 * c;
    }
}

__device__ static float rope_yarn_ramp_dev(float low, float high, int i0) {
    float y = ((float)(i0 / 2) - low) / fmaxf(0.001f, high - low);
    return 1.0f - fminf(1.0f, fmaxf(0.0f, y));
}

int main() {
    CHECK(ds4_gpu_init());
    hipDeviceProp_t prop;
    CHECK(hipGetDeviceProperties(&prop, 0) == hipSuccess);
    if (strncmp(prop.gcnArchName, "gfx1151", 7) != 0) {
        puts("SKIP: wave specialization requires gfx1151");
        return 0;
    }
    const uint32_t max_out = 64u * 512u;
    std::vector<unsigned char> weights((size_t)max_out * 34u);
    for (uint32_t r = 0; r < max_out; r++) {
        __half d = __float2half(0.03125f);
        memcpy(weights.data() + r * 34u, &d, 2);
        for (uint32_t j = 0; j < 32u; j++)
            weights[r * 34u + 2u + j] = (unsigned char)(int8_t)((int)((r * 17u + j * 13u) % 255u) - 127);
    }
    CHECK(ds4_gpu_set_model_map(weights.data(), weights.size()));
    unsigned cases = 0;
    for (uint32_t tokens : {32u, 33u, 127u, 2048u}) {
        for (uint32_t heads : {1u, 3u, 64u}) {
            for (uint32_t dim : {256u, 512u}) {
                const size_t count = (size_t)tokens * heads * dim;
                std::vector<float> input((size_t)tokens * 32u), actual(count + 8u), expected(count);
                ds4_gpu_tensor *x = ds4_gpu_tensor_alloc(input.size() * sizeof(float));
                ds4_gpu_tensor *half = ds4_gpu_tensor_alloc(count * 2u);
                ds4_gpu_tensor *out = ds4_gpu_tensor_alloc((count + 8u) * sizeof(float));
                float *ref;
                CHECK(x && half && out && hipMalloc(&ref, count * sizeof(float)) == hipSuccess);
                for (unsigned mode = 0; mode < 4u; mode++) {
                    for (size_t i = 0; i < input.size(); i++) {
                        input[i] = mode == 0 ? 0.0f : sinf((float)i * 0.123f) *
                            (mode == 3 ? ldexpf(1.0f, (int)(i % 15u) - 7) : 3.0f);
                    }
                    CHECK(ds4_gpu_tensor_write(x, 0, input.data(), input.size() * sizeof(float)));
                    std::fill(actual.begin(), actual.end(), -12345.0f);
                    CHECK(ds4_gpu_tensor_write(out, 0, actual.data(), actual.size() * sizeof(float)));
                    const uint32_t rot = mode == 3 ? 128u : 64u;
                    const uint32_t pos = mode & 1u ? 233u : 0u;
                    const float ext = mode >= 2u ? 1.0f : 0.0f;
                    const bool inverse = mode == 3u;
                    CHECK(ds4_gpu_attn_q_b_f16_head_rms_rope_tail_tensor(out, half,
                        weights.data(), weights.size(), 0, 32, heads * dim, x,
                        tokens, heads, dim, rot, pos, 65536, inverse,
                        10000.0f, 0.25f, ext, 1.0f, 32.0f, 1.0f, 1e-6f));
                    reference_norm_rope<<<tokens * heads, 256>>>(ref,
                        (const __half *)ds4_gpu_tensor_contents(half), tokens, heads, dim,
                        rot, pos, 65536, inverse, 10000.0f, 0.25f, ext, 1.0f, 32.0f, 1.0f, 1e-6f);
                    CHECK(hipGetLastError() == hipSuccess);
                    CHECK(hipMemcpy(expected.data(), ref, count * sizeof(float), hipMemcpyDeviceToHost) == hipSuccess);
                    CHECK(ds4_gpu_tensor_read(out, 0, actual.data(), actual.size() * sizeof(float)));
                    for (size_t i = 0; i < count; i++) {
                        if (memcmp(&expected[i], &actual[i], sizeof(float)) != 0) {
                            fprintf(stderr, "tokens=%u heads=%u dim=%u mode=%u index=%zu: %.9g != %.9g\n",
                                    tokens, heads, dim, mode, i, actual[i], expected[i]);
                            return 1;
                        }
                    }
                    for (size_t i = count; i < count + 8u; i++) CHECK(actual[i] == -12345.0f);
                    cases++;
                }
                CHECK(hipFree(ref) == hipSuccess);
                ds4_gpu_tensor_free(out); ds4_gpu_tensor_free(half); ds4_gpu_tensor_free(x);
            }
        }
    }
    printf("PASS: %u head norm/RoPE cases, bit-identical with output guards intact\n", cases);
    ds4_gpu_cleanup();
}
