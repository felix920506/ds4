/* Check the tiled MXFP4 prefill kernels against a double-precision host
 * evaluation of the same routed MoE: F16 activations, exact MXFP4 weights,
 * F16 mid and down rows.  The Q8_K integer kernels they replace are run on
 * the same inputs as a second opinion on the reference itself.
 *
 * Skewed and sparse routes cover partial 16/32/48/64-token tiles, empty
 * experts and several tiles per expert; the weights include zero-scale
 * blocks and blocks below the exponent range the kernels rescale in. */
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

constexpr uint32_t E = 32, K = 4096, M = 2048, N = 4096, ROUTES = 6;
constexpr float CLAMP = 7.0f;
static const double mxfp4_values[16] = {
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
    -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
};

static ds4_gpu_tensor *alloc(size_t bytes) {
    ds4_gpu_tensor *p = ds4_gpu_tensor_alloc(bytes);
    CHECK(p);
    return p;
}

static float round_f16(float v) {
    return __half2float(__float2half(v));
}

/* One MXFP4 row of n weights against y. */
static double dot_mxfp4(const unsigned char *row, const float *y, uint32_t n) {
    double sum = 0.0;
    for (uint32_t b = 0; b < n / 32u; b++) {
        const unsigned char *blk = row + (size_t)b * 17u;
        double part = 0.0;
        for (uint32_t i = 0; i < 16u; i++) {
            part += mxfp4_values[blk[1u + i] & 15u] * y[b * 32u + i];
            part += mxfp4_values[blk[1u + i] >> 4u] * y[b * 32u + 16u + i];
        }
        sum += ldexp(part, blk[0] == 0 ? -127 : (int)blk[0] - 127);
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
    const size_t gr = K / 32 * 17, dr = M / 32 * 17;
    const size_t ge = M * gr, de = N * dr, go = 0, uo = E * ge, dwo = 2 * E * ge;
    std::vector<unsigned char> model(dwo + E * de);
    uint32_t rng = 42;
    for (auto &b : model) { rng = 1664525u * rng + 1013904223u; b = rng >> 24; }
    for (size_t i = 0, blk = 0; i < model.size(); i += 17, blk++) {
        rng = 1664525u * rng + 1013904223u;
        model[i] = blk % 23 == 0 ? 0 : blk % 29 == 0 ? 40 : 117 + (rng >> 29);
    }
    CHECK(ds4_gpu_set_model_map(model.data(), model.size()));
    CHECK(ds4_gpu_cache_model_range(model.data(), model.size(), go, E * ge, "moe_gate"));
    CHECK(ds4_gpu_cache_model_range(model.data(), model.size(), uo, E * ge, "moe_up"));
    CHECK(ds4_gpu_cache_model_range(model.data(), model.size(), dwo, E * de, "moe_down"));

    unsigned cases = 0;
    double worst_tiles = 0.0, worst_q8 = 0.0;
    std::vector<float> x16(K), mid16(M);
    for (uint32_t nt : {128u, 137u, 300u, 4096u}) {
        const size_t pairs = nt * ROUTES, mids = pairs * M, downs = pairs * N, output = nt * N;
        std::vector<float> x(nt * K), w(pairs), actual(output + 8);
        std::vector<int32_t> ids(pairs);
        for (size_t i = 0; i < x.size(); i++) x[i] = 2.0f * sinf((float)i * 0.123f);
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
                        float g = (float)dot_mxfp4(model.data() + go + e * ge + row * gr, x16.data(), K);
                        float u = (float)dot_mxfp4(model.data() + uo + e * ge + row * gr, x16.data(), K);
                        g = fminf(g, CLAMP);
                        u = fminf(fmaxf(u, -CLAMP), CLAMP);
                        mid16[row] = round_f16((g / (1.0f + expf(-g))) * u * w[t * ROUTES + r]);
                    }
                    for (uint32_t row = 0; row < N; row++) {
                        ref[(size_t)s * N + row] += round_f16((float)dot_mxfp4(
                                model.data() + dwo + e * de + row * dr, mid16.data(), M));
                    }
                }
                for (uint32_t row = 0; row < N; row++) norm += ref[(size_t)s * N + row] * ref[(size_t)s * N + row];
            }
            double err[2] = {0.0, 0.0};
            for (int tiles = 1; tiles >= 0; tiles--) {
                CHECK(setenv("DS4_ROCM_MXFP4_TILES", tiles ? "1" : "0", 1) == 0);
                std::fill(actual.begin(), actual.end(), -12345.0f);
                CHECK(ds4_gpu_tensor_write(out, 0, actual.data(), actual.size() * 4));
                bool half = !tiles;
                CHECK(ds4_gpu_routed_moe_batch_tensor(out, gate, up, mid, down,
                    model.data(), model.size(), go, uo, dwo, 39u, 39u, ge, gr, de, dr,
                    K, M, N, it, wt, E, ROUTES, CLAMP, xt, 0u, nt, &half, true));
                CHECK(half == (tiles != 0));
                CHECK(ds4_gpu_tensor_read(out, 0, actual.data(), actual.size() * 4));
                for (size_t i = 0; i < output; i++) CHECK(std::isfinite(actual[i]));
                for (size_t i = output; i < output + 8; i++) CHECK(actual[i] == -12345.0f);
                for (uint32_t s = 0; s < 8u; s++) {
                    for (uint32_t row = 0; row < N; row++) {
                        const double d = actual[(size_t)sampled[s] * N + row] - ref[(size_t)s * N + row];
                        err[tiles] += d * d;
                    }
                }
            }
            CHECK(norm > 0.0);
            const double rel_tiles = sqrt(err[1] / norm), rel_q8 = sqrt(err[0] / norm);
            if (rel_tiles > worst_tiles) worst_tiles = rel_tiles;
            if (rel_q8 > worst_q8) worst_q8 = rel_q8;
            /* The tiles differ from the reference by float32 accumulation
             * and the odd F16 rounding it flips; the integer kernels by
             * their 8-bit activations. */
            CHECK(rel_tiles < 5.0e-4);
            CHECK(rel_q8 < 5.0e-2);
            cases++;
        }
        for (auto *p : {xt, wt, it, gate, up, mid, down, out}) ds4_gpu_tensor_free(p);
    }
    unsetenv("DS4_ROCM_MXFP4_TILES");
    ds4_gpu_cleanup();
    printf("PASS: %u MXFP4 routing cases, relative RMS error %.2e tiled, %.2e Q8_K\n",
           cases, worst_tiles, worst_q8);
}
