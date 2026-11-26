/* IQ2_XXS/Q2_K prefill batches of 128 or more rows on gfx1151 skip the Q8_K
 * quantisation of x: the token tiles and the MMQ pair read F32/F16 x
 * directly.  When neither produces gate/up, the scalar fallback gate/up
 * kernels read the Q8_K x from xq (the down buffer), so the launcher must
 * quantise it first, or fail.
 *
 * Each shape runs the identical call twice with the down buffer pre-filled
 * with two different valid Q8_K patterns.  A launch that quantises x before
 * reading it gives bit-identical outputs; one that returns 0 is accepted.
 *   mmq-refusal: out_dim 256 (the token tiles need a multiple of 512) and 4
 *     experts with 6 routes, which the MMQ pair refuses, so the scalar
 *     fallback runs.
 *   token-tiles: the production shape the token tiles take; covers their
 *     ordinary path and, when the tile setup fails, the fallback behind it. */
#include "ds4_gpu.h"
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <sys/mman.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#define CHECK(x) do { if (!(x)) { fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #x); exit(1); } } while (0)
bool ds4_log_is_tty(FILE *) { return false; }

struct shape { const char *name; uint32_t E, K, M, N; };
constexpr uint32_t ROUTES = 6, NT = 128;

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
        puts("SKIP: gfx1151 required (the MMQ IQ2 gate/up and token tiles are gfx1151-only)");
        ds4_gpu_cleanup();
        return 0;
    }
    unsetenv("DS4_ROCM_MMQ_IQ2");
    const shape shapes[] = {{"mmq-refusal", 4, 512, 512, 256}, {"token-tiles", 32, 4096, 2048, 4096}};
    /* The runtime registers the page-rounded host range of each cached
     * model range, so the fake model gets a mapping of its own. */
    size_t offset[2], total = 0;
    for (int s = 0; s < 2; s++) {
        const shape &sh = shapes[s];
        offset[s] = total;
        total += (size_t)2 * sh.E * sh.M * (sh.K / 256 * 66) + (size_t)sh.E * sh.N * (sh.M / 256 * 84);
        total = (total + 4095) & ~(size_t)4095;
    }
    unsigned char *model = (unsigned char *)mmap(NULL, total, PROT_READ | PROT_WRITE,
                                                 MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    CHECK(model != MAP_FAILED);
    uint32_t rng = 5;
    for (size_t i = 0; i < total; i++) { rng = 1664525u * rng + 1013904223u; model[i] = rng >> 24; }
    for (int s = 0; s < 2; s++) {
        const shape &sh = shapes[s];
        const size_t gr = sh.K / 256 * 66, dr = sh.M / 256 * 84, ge = sh.M * gr;
        const size_t go = offset[s], dwo = go + 2 * sh.E * ge, end = dwo + sh.E * sh.N * dr;
        for (size_t i = go; i < dwo; i += 66) { __half d = __float2half(0.002f); memcpy(model + i, &d, 2); }
        for (size_t i = dwo; i < end; i += 84) {
            __half d = __float2half(0.003f), m = __float2half(0.002f);
            memcpy(model + i + 80, &d, 2);
            memcpy(model + i + 82, &m, 2);
        }
    }
    CHECK(ds4_gpu_set_model_map(model, total));

    int failed = 0;
    for (int s = 0; s < 2; s++) {
        const shape &sh = shapes[s];
        const uint32_t E = sh.E, K = sh.K, M = sh.M, N = sh.N;
        const size_t gr = K / 256 * 66, dr = M / 256 * 84, ge = M * gr, de = N * dr;
        const size_t go = offset[s], uo = go + E * ge, dwo = go + 2 * E * ge;
        CHECK(ds4_gpu_cache_model_range(model, total, go, E * ge, "moe_gate"));
        CHECK(ds4_gpu_cache_model_range(model, total, uo, E * ge, "moe_up"));
        CHECK(ds4_gpu_cache_model_range(model, total, dwo, E * de, "moe_down"));

        const size_t pairs = NT * ROUTES, mids = pairs * M, downs = pairs * N, output = NT * N;
        std::vector<float> x(NT * K), w(pairs);
        std::vector<int32_t> ids(pairs);
        for (size_t i = 0; i < x.size(); i++) x[i] = sinf((float)i * 0.05f);
        for (size_t i = 0; i < w.size(); i++) w[i] = 0.1f + (i % 7) * 0.01f;
        for (uint32_t t = 0; t < NT; t++) for (uint32_t r = 0; r < ROUTES; r++) ids[t * ROUTES + r] = (int32_t)((t + r) % E);
        ds4_gpu_tensor *xt = alloc(x.size() * 4), *wt = alloc(w.size() * 4), *it = alloc(ids.size() * 4);
        ds4_gpu_tensor *gate = alloc(mids * 4), *up = alloc(mids * 4), *mid = alloc(mids * 4);
        ds4_gpu_tensor *down = alloc(downs * 4), *out = alloc(output * 4);
        CHECK(ds4_gpu_tensor_write(xt, 0, x.data(), x.size() * 4));
        CHECK(ds4_gpu_tensor_write(wt, 0, w.data(), w.size() * 4));
        CHECK(ds4_gpu_tensor_write(it, 0, ids.data(), ids.size() * 4));

        /* Pattern 0: all-zero bytes (Q8_K d = 0).  Pattern 1: valid Q8_K
         * blocks (float d, 256 int8, 16 int16 bsums) of some other input. */
        std::vector<unsigned char> pat(downs * 4);
        std::vector<float> res[2];
        int rcs[2];
        for (int p = 0; p < 2; p++) {
            if (p == 0) std::fill(pat.begin(), pat.end(), 0);
            else {
                for (size_t i = 0; i + 292 <= pat.size(); i += 292) {
                    const float d = 0.01f;
                    memcpy(pat.data() + i, &d, 4);
                    int sums[16] = {0};
                    for (int j = 0; j < 256; j++) {
                        rng = 1664525u * rng + 1013904223u;
                        const int8_t q = (int8_t)((int)(rng >> 25) - 64);
                        pat[i + 4 + j] = (unsigned char)q;
                        sums[j / 16] += q;
                    }
                    for (int j = 0; j < 16; j++) { int16_t v = (int16_t)sums[j]; memcpy(pat.data() + i + 260 + 2 * j, &v, 2); }
                }
            }
            CHECK(ds4_gpu_tensor_write(down, 0, pat.data(), pat.size()));
            bool half = false;
            rcs[p] = ds4_gpu_routed_moe_batch_tensor(out, gate, up, mid, down,
                model, total, go, uo, dwo, 16u, 10u, ge, gr, de, dr,
                K, M, N, it, wt, E, ROUTES, 0.0f, xt, 0u, NT, &half, true);
            res[p].resize(output);
            if (rcs[p]) CHECK(ds4_gpu_tensor_read(out, 0, res[p].data(), output * 4));
        }
        for (auto *q : {xt, wt, it, gate, up, mid, down, out}) ds4_gpu_tensor_free(q);
        if (!rcs[0] && !rcs[1]) {
            printf("%s: launch returned 0\n", sh.name);
            continue;
        }
        CHECK(rcs[0] && rcs[1]);
        size_t diff = 0, nonzero = 0;
        for (size_t i = 0; i < output; i++) {
            diff += memcmp(&res[0][i], &res[1][i], 4) != 0;
            nonzero += res[0][i] != 0.0f;
        }
        printf("%s: %zu/%zu outputs depend on the previous contents of down, %zu nonzero\n",
               sh.name, diff, output, nonzero);
        if (diff || !nonzero) failed = 1;
    }
    ds4_gpu_cleanup();
    munmap(model, total);
    if (failed) {
        puts("FAIL: routed MoE gate/up read a Q8_K x that this call never wrote");
        return 1;
    }
    puts("PASS: routed MoE gate/up output is independent of the previous contents of down");
    return 0;
}
