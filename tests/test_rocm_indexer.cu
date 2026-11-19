/* Long-context indexer decode.
 *
 * Scores: the single-token scores must match the original block-per-row
 * kernel bit for bit, on both sides of the row count where the production
 * path changes kernel.  The top-k that follows compares these scores, so a
 * different rounding could select different compressed rows.
 *
 * Top-k: the selected rows must be the best ones under (score descending, row
 * ascending), best first, including among ties, signed zeros and masked rows. */
#include "ds4_gpu.h"
#include <hip/hip_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <algorithm>
#define CHECK(x) do { if (!(x)) { fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #x); exit(1); } } while (0)
bool ds4_log_is_tty(FILE *) { return false; }

__global__ static void reference_score_one(
        float *scores,
        const float *q,
        const float *weights,
        const float *index_comp,
        uint32_t n_comp,
        float scale) {
    const uint32_t c = blockIdx.x;
    const uint32_t tid = threadIdx.x;
    const uint32_t lane = tid & 31u;
    const uint32_t warp = tid >> 5u;
    if (c >= n_comp) return;
    __shared__ float krow[128];
    __shared__ float partial[4];
    krow[tid] = index_comp[(uint64_t)c * 128u + tid];
    __syncthreads();
    float total = 0.0f;
    for (uint32_t h0 = 0; h0 < 64u; h0 += 4u) {
        const uint32_t h = h0 + warp;
        const float4 qv = ((const float4 *)(q + (uint64_t)h * 128u))[lane];
        const float4 kv = ((const float4 *)krow)[lane];
        float dot = qv.x * kv.x + qv.y * kv.y + qv.z * kv.z + qv.w * kv.w;
        for (int offset = 16; offset > 0; offset >>= 1) dot += __shfl_down(dot, offset, 32);
        if (lane == 0) partial[warp] = fmaxf(dot, 0.0f) * weights[h] * scale;
        __syncthreads();
        if (tid == 0) total += partial[0] + partial[1] + partial[2] + partial[3];
        __syncthreads();
    }
    if (tid == 0) scores[c] = total;
}

static unsigned check_topk(void) {
    unsigned cases = 0;
    for (uint32_t rows : {8193u, 40000u, 250000u}) {
        for (uint32_t tokens : {1u, 3u}) {
            for (uint32_t top_k : {512u, 1024u, 2048u}) {
                for (unsigned mode = 0; mode < 5u; mode++) {
                    std::vector<float> scores((size_t)rows * tokens);
                    uint32_t seed = 99u + mode + rows + top_k;
                    for (size_t i = 0; i < scores.size(); i++) {
                        seed = seed * 1664525u + 1013904223u;
                        float v = (float)(seed >> 8) / 8388608.0f - 1.0f;
                        if (mode == 1) v = floorf(v * 8.0f) / 8.0f;                 /* few distinct values */
                        if (mode == 2) v = i % 3u == 0 ? 0.0f : (i % 3u == 1 ? -0.0f : v);
                        if (mode == 3) v = 0.25f;                                   /* one value */
                        if (mode == 4 && i % rows > rows / 64u) v = -INFINITY;      /* causal mask */
                        scores[i] = v;
                    }
                    std::vector<uint32_t> actual((size_t)tokens * top_k + 8u, 0xdeadbeefu), order(rows);
                    ds4_gpu_tensor *ts = ds4_gpu_tensor_alloc(scores.size() * sizeof(float));
                    ds4_gpu_tensor *tsel = ds4_gpu_tensor_alloc(actual.size() * sizeof(uint32_t));
                    CHECK(ts && tsel);
                    CHECK(ds4_gpu_tensor_write(ts, 0, scores.data(), scores.size() * sizeof(float)));
                    CHECK(ds4_gpu_tensor_write(tsel, 0, actual.data(), actual.size() * sizeof(uint32_t)));
                    CHECK(ds4_gpu_indexer_topk_tensor(tsel, ts, rows, tokens, top_k));
                    CHECK(ds4_gpu_tensor_read(tsel, 0, actual.data(), actual.size() * sizeof(uint32_t)));
                    for (uint32_t t = 0; t < tokens; t++) {
                        const float *row = scores.data() + (size_t)t * rows;
                        for (uint32_t i = 0; i < rows; i++) order[i] = i;
                        std::partial_sort(order.begin(), order.begin() + top_k, order.end(),
                            [row](uint32_t a, uint32_t b) { return row[a] > row[b] || (row[a] == row[b] && a < b); });
                        for (uint32_t i = 0; i < top_k; i++) {
                            if (actual[(size_t)t * top_k + i] != order[i]) {
                                fprintf(stderr, "topk rows=%u tokens=%u k=%u mode=%u token=%u rank=%u: %u != %u\n",
                                        rows, tokens, top_k, mode, t, i, actual[(size_t)t * top_k + i], order[i]);
                                exit(1);
                            }
                        }
                    }
                    for (size_t i = (size_t)tokens * top_k; i < actual.size(); i++) CHECK(actual[i] == 0xdeadbeefu);
                    ds4_gpu_tensor_free(tsel); ds4_gpu_tensor_free(ts);
                    cases++;
                }
            }
        }
    }
    return cases;
}

int main() {
    CHECK(ds4_gpu_init());
    const uint32_t max_rows = 70001u;
    std::vector<float> q(64u * 128u), w(64u), k((size_t)max_rows * 128u);
    ds4_gpu_tensor *tq = ds4_gpu_tensor_alloc(q.size() * sizeof(float));
    ds4_gpu_tensor *tw = ds4_gpu_tensor_alloc(w.size() * sizeof(float));
    ds4_gpu_tensor *tk = ds4_gpu_tensor_alloc(k.size() * sizeof(float));
    ds4_gpu_tensor *ts = ds4_gpu_tensor_alloc((size_t)(max_rows + 8u) * sizeof(float));
    float *ref;
    CHECK(tq && tw && tk && ts && hipMalloc(&ref, (size_t)max_rows * sizeof(float)) == hipSuccess);
    const float scale = 1.0f / sqrtf(128.0f * 64.0f);
    unsigned cases = 0;
    for (unsigned mode = 0; mode < 3u; mode++) {
        /* Smooth values, values spread over many binades, and rows that tie. */
        uint32_t seed = 12345u + mode;
        auto rnd = [&seed]() {
            seed = seed * 1664525u + 1013904223u;
            return (float)(seed >> 8) / 8388608.0f - 1.0f;
        };
        for (auto &v : q) v = rnd() * (mode == 1 ? ldexpf(1.0f, (int)(seed % 13u) - 6) : 2.0f);
        for (auto &v : w) v = rnd();
        for (auto &v : k) v = rnd() * (mode == 1 ? ldexpf(1.0f, (int)(seed % 9u) - 4) : 1.0f);
        if (mode == 2) for (size_t i = 128u * 7u; i < k.size(); i++) k[i] = k[i - 128u * 7u];
        CHECK(ds4_gpu_tensor_write(tq, 0, q.data(), q.size() * sizeof(float)));
        CHECK(ds4_gpu_tensor_write(tw, 0, w.data(), w.size() * sizeof(float)));
        CHECK(ds4_gpu_tensor_write(tk, 0, k.data(), k.size() * sizeof(float)));
        for (uint32_t rows : {1u, 31u, 2048u, 2049u, 2305u, 4096u, 32769u, max_rows}) {
            std::vector<float> actual(rows + 8u, -12345.0f), expected(rows);
            CHECK(ds4_gpu_tensor_write(ts, 0, actual.data(), actual.size() * sizeof(float)));
            CHECK(ds4_gpu_indexer_score_one_tensor(ts, tq, tw, tk, rows, 64, 128, scale));
            reference_score_one<<<rows, 128>>>(ref,
                (const float *)ds4_gpu_tensor_contents(tq),
                (const float *)ds4_gpu_tensor_contents(tw),
                (const float *)ds4_gpu_tensor_contents(tk), rows, scale);
            CHECK(hipGetLastError() == hipSuccess);
            CHECK(hipMemcpy(expected.data(), ref, (size_t)rows * sizeof(float), hipMemcpyDeviceToHost) == hipSuccess);
            CHECK(ds4_gpu_tensor_read(ts, 0, actual.data(), actual.size() * sizeof(float)));
            for (uint32_t i = 0; i < rows; i++) {
                if (memcmp(&expected[i], &actual[i], sizeof(float)) != 0) {
                    fprintf(stderr, "mode=%u rows=%u row=%u: %.9g != %.9g\n",
                            mode, rows, i, actual[i], expected[i]);
                    return 1;
                }
            }
            for (uint32_t i = rows; i < rows + 8u; i++) CHECK(actual[i] == -12345.0f);
            cases++;
        }
    }
    CHECK(hipFree(ref) == hipSuccess);
    ds4_gpu_tensor_free(ts); ds4_gpu_tensor_free(tk); ds4_gpu_tensor_free(tw); ds4_gpu_tensor_free(tq);
    printf("PASS: %u indexer score cases, bit-identical with output guards intact\n", cases);
    printf("PASS: %u indexer top-k cases, exact with output guards intact\n", check_topk());
    ds4_gpu_cleanup();
}
