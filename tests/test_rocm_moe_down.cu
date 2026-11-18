/* Exercise production IQ2/Q2 prefill routing with both down tile schedules.
 * Skewed and sparse routes cover partial 16/32/48/64-token tiles, empty
 * experts, nontrivial pair permutations, and multiple tiles per expert. */
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
static ds4_gpu_tensor *alloc(size_t bytes) {
    ds4_gpu_tensor *p = ds4_gpu_tensor_alloc(bytes);
    CHECK(p);
    return p;
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
    constexpr uint32_t E = 32, K = 4096, M = 2048, N = 4096, ROUTES = 6;
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
        for (unsigned pattern = 0; pattern < 3; pattern++) {
            for (uint32_t t = 0; t < nt; t++) for (uint32_t r = 0; r < ROUTES; r++) {
                ids[t * ROUTES + r] = pattern == 0 ? r :
                    pattern == 1 ? (t * 7 + r * 3) % E :
                    (r < 3 ? r : 3 + (t * 13 + r * 7) % (E - 3));
            }
            CHECK(ds4_gpu_tensor_write(it, 0, ids.data(), ids.size() * 4));
            for (int variant = 0; variant < 2; variant++) {
                CHECK(setenv("DS4_ROCM_MOE_DOWN_NARROW", variant ? "1" : "0", 1) == 0);
                std::fill(actual.begin(), actual.end(), -12345.0f);
                CHECK(ds4_gpu_tensor_write(out, 0, actual.data(), actual.size() * 4));
                bool half = false;
                CHECK(ds4_gpu_routed_moe_batch_tensor(out, gate, up, mid, down,
                    model.data(), model.size(), go, uo, dwo, 16u, 10u, ge, gr, de, dr,
                    K, M, N, it, wt, E, ROUTES, 7.0f, xt, 0u, nt, &half, true));
                CHECK(ds4_gpu_tensor_read(out, 0, actual.data(), actual.size() * 4));
                for (size_t i = 0; i < output; i++) CHECK(std::isfinite(actual[i]));
                for (size_t i = output; i < output + 8; i++) CHECK(actual[i] == -12345.0f);
                if (!variant) expected = actual;
                else CHECK(memcmp(expected.data(), actual.data(), actual.size() * 4) == 0);
            }
            cases++;
        }
        for (auto *p : {xt, wt, it, gate, up, mid, down, out}) ds4_gpu_tensor_free(p);
    }
    unsetenv("DS4_ROCM_MOE_DOWN_NARROW");
    ds4_gpu_cleanup();
    printf("PASS: %u production MoE routing cases, bit-identical with intact output guards\n", cases);
}
