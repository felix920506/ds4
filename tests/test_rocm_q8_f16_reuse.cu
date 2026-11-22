/* A Q8_0 batch whose F16 weights are not cached, with no room to cache them,
 * expands them over the least recently used cache entry.  Cycle more weights
 * than the cache holds and check every product against a host evaluation: a
 * batch that read an entry's previous weight would be wrong everywhere. */
#include "ds4_gpu.h"
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#define CHECK(x) do { if (!(x)) { fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #x); exit(1); } } while (0)
bool ds4_log_is_tty(FILE *) { return false; }
extern "C" uint64_t ds4_gpu_q8_f16_reuse_count(void);

constexpr uint32_t K = 4096, N = 2048, MATS = 7, CACHED = 4, NT = 256;
constexpr size_t ROW = K / 32 * 34, MAT = (size_t)N * ROW;

int main() {
    /* Room for CACHED of the MATS expansions. */
    char mb[32];
    snprintf(mb, sizeof(mb), "%u", (unsigned)((size_t)CACHED * K * N * 2 >> 20));
    CHECK(setenv("DS4_ROCM_Q8_F16_CACHE_MB", mb, 1) == 0);
    CHECK(ds4_gpu_init());
    std::vector<unsigned char> model(MATS * MAT);
    uint32_t rng = 7;
    for (size_t i = 0; i < model.size(); i += 34) {
        rng = 1664525u * rng + 1013904223u;
        const __half scale = __float2half(0.001f + (float)(rng >> 24) * 1.0e-5f);
        memcpy(&model[i], &scale, 2);
        for (size_t j = 2; j < 34; j++) {
            rng = 1664525u * rng + 1013904223u;
            model[i + j] = rng >> 24;
        }
    }
    CHECK(ds4_gpu_set_model_map(model.data(), model.size()));
    CHECK(ds4_gpu_cache_model_range(model.data(), model.size(), 0, model.size(), "q8_0"));

    std::vector<float> x((size_t)NT * K), y((size_t)NT * N);
    for (size_t i = 0; i < x.size(); i++) x[i] = sinf((float)i * 0.37f);
    ds4_gpu_tensor *xt = ds4_gpu_tensor_alloc(x.size() * 4), *yt = ds4_gpu_tensor_alloc(y.size() * 4);
    CHECK(xt && yt && ds4_gpu_tensor_write(xt, 0, x.data(), x.size() * 4));

    const uint32_t tokens[] = {0u, 1u, NT / 2u, NT - 1u};
    double worst = 0.0;
    unsigned products = 0;
    /* Forward twice, then backward: both orders find their weight replaced. */
    for (unsigned step = 0; step < 3 * MATS; step++) {
        const unsigned m = step < 2 * MATS ? step % MATS : 3 * MATS - 1 - step;
        CHECK(ds4_gpu_matmul_q8_0_tensor(yt, model.data(), model.size(), m * MAT, K, N, xt, NT));
        CHECK(ds4_gpu_tensor_read(yt, 0, y.data(), y.size() * 4));
        double err = 0.0, norm = 0.0;
        for (uint32_t t : tokens) {
            for (uint32_t row = 0; row < N; row += 7) {
                const unsigned char *w = &model[m * MAT + row * ROW];
                double ref = 0.0;
                for (uint32_t b = 0; b < K / 32; b++) {
                    __half scale;
                    memcpy(&scale, w + b * 34, 2);
                    double part = 0.0;
                    for (uint32_t j = 0; j < 32; j++) {
                        part += (double)(int8_t)w[b * 34 + 2 + j] *
                                __half2float(__float2half(x[(size_t)t * K + b * 32 + j]));
                    }
                    ref += part * __half2float(scale);
                }
                const double d = y[(size_t)t * N + row] - ref;
                err += d * d;
                norm += ref * ref;
            }
        }
        CHECK(norm > 0.0);
        const double rel = sqrt(err / norm);
        if (rel > worst) worst = rel;
        /* F16 weights and F16 GEMM accumulation; another weight gives ~1.4. */
        CHECK(rel < 2.0e-3);
        products++;
    }
    const uint64_t reuses = ds4_gpu_q8_f16_reuse_count();
    CHECK(reuses >= products - CACHED - MATS);
    ds4_gpu_tensor_free(xt);
    ds4_gpu_tensor_free(yt);
    ds4_gpu_cleanup();
    printf("PASS: %u Q8 products over a %u-entry F16 cache, %llu expanded over another entry, "
           "relative RMS error %.2e\n", products, CACHED, (unsigned long long)reuses, worst);
}
