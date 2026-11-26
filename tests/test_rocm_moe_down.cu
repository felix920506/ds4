/* Exercise production IQ2/Q2 prefill routing with both down tile schedules,
 * against a double-precision host evaluation of the same routed MoE: F16
 * activations, exact IQ2_XXS gate/up and Q2_K down weights, F16 mid and down
 * rows.  The two schedules must also agree bit for bit.
 * Skewed and sparse routes cover partial 16/32/48/64-token tiles, empty
 * experts, nontrivial pair permutations, and multiple tiles per expert. */
#include "ds4_gpu.h"
#include "cuda/mmq/test/iq2_host_tables.h"
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#define CHECK(x) do { if (!(x)) { fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #x); exit(1); } } while (0)
bool ds4_log_is_tty(FILE *) { return false; }

constexpr uint32_t E = 32, K = 4096, M = 2048, N = 4096, ROUTES = 6;
constexpr float CLAMP = 7.0f;

static ds4_gpu_tensor *alloc(size_t bytes) {
    ds4_gpu_tensor *p = ds4_gpu_tensor_alloc(bytes);
    CHECK(p);
    return p;
}

static float round_f16(float v) {
    return __half2float(__float2half(v));
}

static double load_f16(const unsigned char *p) {
    __half h;
    memcpy(&h, p, 2);
    return __half2float(h);
}

/* One IQ2_XXS row (66-byte blocks: F16 d, then per 32 weights four grid
 * indices and a word of 4x7 sign bits and a 4-bit scale) against y. */
static double dot_iq2_xxs(const unsigned char *row, const float *y, uint32_t n) {
    double sum = 0.0;
    for (uint32_t b = 0; b < n / 256u; b++) {
        const unsigned char *blk = row + (size_t)b * 66u;
        const double d = load_f16(blk);
        for (uint32_t g = 0; g < 8u; g++) {
            uint32_t aux[2];
            memcpy(aux, blk + 2u + g * 8u, 8);
            double part = 0.0;
            for (uint32_t l = 0; l < 4u; l++) {
                const uint8_t *grid = (const uint8_t *)&iq2_host::iq2xxs_grid[(aux[0] >> (8u * l)) & 255u];
                const uint8_t signs = iq2_host::ksigns_iq2xs[(aux[1] >> (7u * l)) & 127u];
                for (uint32_t j = 0; j < 8u; j++) {
                    const double v = (signs & iq2_host::kmask_iq2xs[j]) ? -(double)grid[j] : (double)grid[j];
                    part += v * y[b * 256u + g * 32u + l * 8u + j];
                }
            }
            sum += 0.125 * d * (2.0 * (aux[1] >> 28) + 1.0) * part;
        }
    }
    return sum;
}

/* One Q2_K row (84-byte blocks: 16 scale/minimum bytes, 64 code bytes, F16 d
 * and dmin) against y. */
static double dot_q2_K(const unsigned char *row, const float *y, uint32_t n) {
    double sum = 0.0;
    for (uint32_t b = 0; b < n / 256u; b++) {
        const unsigned char *blk = row + (size_t)b * 84u;
        const double d = load_f16(blk + 80), dmin = load_f16(blk + 82);
        for (uint32_t i = 0; i < 256u; i++) {
            const uint32_t group = i / 16u, l = i % 16u;
            const uint32_t q_base = 32u * (group / 8u) + 16u * (group & 1u);
            const uint32_t q = (blk[16u + q_base + l] >> (((group / 2u) & 3u) * 2u)) & 3u;
            const uint32_t sc = blk[group];
            sum += (d * (sc & 15u) * q - dmin * (sc >> 4u)) * y[b * 256u + i];
        }
    }
    return sum;
}

int main() {
    CHECK(ds4_gpu_init());
    hipDeviceProp_t prop;
    CHECK(hipGetDeviceProperties(&prop, 0) == hipSuccess);
    if (strncmp(prop.gcnArchName, "gfx1151", 7)) {
        puts("SKIP: gfx1151 required");
        ds4_gpu_cleanup();
        return 0;
    }
    const size_t gr = K / 256 * 66, dr = M / 256 * 84;
    const size_t ge = M * gr, de = N * dr, go = 0, uo = E * ge, dwo = 2 * E * ge;
    std::vector<unsigned char> model(dwo + E * de);
    uint32_t rng = 42;
    for (auto &b : model) { rng = 1664525u * rng + 1013904223u; b = rng >> 24; }
    for (size_t i = 0; i < dwo; i += 66) {
        __half scale = __float2half((i % 19 == 0) ? 0.0f : 0.002f);
        memcpy(model.data() + i, &scale, 2);
    }
    for (size_t i = dwo; i < model.size(); i += 84) {
        __half scale = __float2half((i % 17 == 0) ? 0.0f : 0.003f);
        __half minimum = __float2half(0.002f);
        memcpy(model.data() + i + 80, &scale, 2);
        memcpy(model.data() + i + 82, &minimum, 2);
    }
    CHECK(ds4_gpu_set_model_map(model.data(), model.size()));
    CHECK(ds4_gpu_cache_model_range(model.data(), model.size(), go, E * ge, "moe_gate"));
    CHECK(ds4_gpu_cache_model_range(model.data(), model.size(), uo, E * ge, "moe_up"));
    CHECK(ds4_gpu_cache_model_range(model.data(), model.size(), dwo, E * de, "moe_down"));
    unsigned cases = 0;
    double worst = 0.0;
    std::vector<float> x16(K), mid16(M);
    for (uint32_t nt : {128u, 137u, 300u, 2048u}) {
        const size_t pairs = nt * ROUTES, mids = pairs * M, downs = pairs * N, output = nt * N;
        std::vector<float> x(nt * K), w(pairs), expected(output + 8), actual(output + 8);
        std::vector<int32_t> ids(pairs);
        for (size_t i = 0; i < x.size(); i++) x[i] = sinf((float)i * 0.123f);
        for (size_t i = 0; i < w.size(); i++) w[i] = 0.1f + (i % 7) * 0.01f;
        ds4_gpu_tensor *xt = alloc(x.size() * 4), *wt = alloc(w.size() * 4), *it = alloc(ids.size() * 4);
        ds4_gpu_tensor *gate = alloc(mids * 4), *up = alloc(mids * 4), *mid = alloc(mids * 4);
        ds4_gpu_tensor *down = alloc(downs * 4), *out = alloc((output + 8) * 4);
        CHECK(ds4_gpu_tensor_write(xt, 0, x.data(), x.size() * 4));
        CHECK(ds4_gpu_tensor_write(wt, 0, w.data(), w.size() * 4));
        const uint32_t sampled[] = {0u, 15u, 16u, 63u, 64u, nt / 2u + 1u, nt - 2u, nt - 1u};
        for (unsigned pattern = 0; pattern < 3; pattern++) {
            for (uint32_t t = 0; t < nt; t++) for (uint32_t r = 0; r < ROUTES; r++) {
                ids[t * ROUTES + r] = pattern == 0 ? r :
                    pattern == 1 ? (t * 7 + r * 3) % E :
                    (r < 3 ? r : 3 + (t * 13 + r * 7) % (E - 3));
            }
            CHECK(ds4_gpu_tensor_write(it, 0, ids.data(), ids.size() * 4));
            std::vector<double> ref(8 * (size_t)N, 0.0);
            double norm = 0.0;
            for (uint32_t s = 0; s < 8u; s++) {
                const uint32_t t = sampled[s];
                for (uint32_t i = 0; i < K; i++) x16[i] = round_f16(x[(size_t)t * K + i]);
                for (uint32_t r = 0; r < ROUTES; r++) {
                    const size_t e = ids[t * ROUTES + r];
                    for (uint32_t row = 0; row < M; row++) {
                        float g = (float)dot_iq2_xxs(model.data() + go + e * ge + row * gr, x16.data(), K);
                        float u = (float)dot_iq2_xxs(model.data() + uo + e * ge + row * gr, x16.data(), K);
                        g = fminf(g, CLAMP);
                        u = fminf(fmaxf(u, -CLAMP), CLAMP);
                        mid16[row] = round_f16((g / (1.0f + expf(-g))) * u * w[t * ROUTES + r]);
                    }
                    for (uint32_t row = 0; row < N; row++) {
                        ref[(size_t)s * N + row] += round_f16((float)dot_q2_K(
                                model.data() + dwo + e * de + row * dr, mid16.data(), M));
                    }
                }
                for (uint32_t row = 0; row < N; row++) norm += ref[(size_t)s * N + row] * ref[(size_t)s * N + row];
            }
            CHECK(norm > 0.0);
            for (int variant = 0; variant < 2; variant++) {
                CHECK(setenv("DS4_ROCM_MOE_DOWN_NARROW", variant ? "1" : "0", 1) == 0);
                std::fill(actual.begin(), actual.end(), -12345.0f);
                CHECK(ds4_gpu_tensor_write(out, 0, actual.data(), actual.size() * 4));
                bool half = false;
                CHECK(ds4_gpu_routed_moe_batch_tensor(out, gate, up, mid, down,
                    model.data(), model.size(), go, uo, dwo, 16u, 10u, ge, gr, de, dr,
                    K, M, N, it, wt, E, ROUTES, CLAMP, xt, 0u, nt, &half, true));
                CHECK(ds4_gpu_tensor_read(out, 0, actual.data(), actual.size() * 4));
                for (size_t i = 0; i < output; i++) CHECK(std::isfinite(actual[i]));
                for (size_t i = output; i < output + 8; i++) CHECK(actual[i] == -12345.0f);
                if (!variant) {
                    double err = 0.0;
                    for (uint32_t s = 0; s < 8u; s++) {
                        for (uint32_t row = 0; row < N; row++) {
                            const double d = actual[(size_t)sampled[s] * N + row] - ref[(size_t)s * N + row];
                            err += d * d;
                        }
                    }
                    const double rel = sqrt(err / norm);
                    if (rel > worst) worst = rel;
                    /* The kernels differ from the reference by float32
                     * accumulation and the F16 rounding of the dequantised
                     * Q2_K weights (about 3e-4); a wrong scale or expert is
                     * off by tens of percent. */
                    if (!(rel < 2.0e-3)) {
                        fprintf(stderr, "FAIL: %u tokens, routes %u: relative RMS error %.3e against the host reference\n",
                                nt, pattern, rel);
                        exit(1);
                    }
                    expected = actual;
                } else {
                    CHECK(memcmp(expected.data(), actual.data(), actual.size() * 4) == 0);
                }
            }
            cases++;
        }
        for (auto *p : {xt, wt, it, gate, up, mid, down, out}) ds4_gpu_tensor_free(p);
    }
    unsetenv("DS4_ROCM_MOE_DOWN_NARROW");
    ds4_gpu_cleanup();
    printf("PASS: %u production MoE routing cases, relative RMS error %.2e against the host reference, "
           "bit-identical schedules with intact output guards\n", cases, worst);
}
