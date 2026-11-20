/* Vision-Exp prefill attention on ROCm.
 *
 * Every case is checked against a host reference in double precision: the keys
 * of a query are the raw rows inside its bounds (the causal window for text,
 * the bidirectional span for image tokens) plus its compressed rows, which are
 * all the rows whose source tokens have arrived, or the part of the indexer's
 * selection among them.  The selection is passed both ways the API takes it,
 * as row numbers and as a dense additive mask.
 *
 * The first case is a chunk at the end of a one million token context.  The
 * launch must not take device memory there: a score matrix over every
 * compressed row is gigabytes of scratch that prefill cannot reclaim for. */
#include "ds4_gpu.h"
#include "ds4_image.h"
#include <hip/hip_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <algorithm>
#include <unistd.h>
#define CHECK(x) do { if (!(x)) { fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #x); exit(1); } } while (0)
bool ds4_log_is_tty(FILE *) { return false; }

enum { DIM = 512, VOCAB = 129280, WINDOW = 128, TOP_K = 512 };
enum selection { ALL, ROWS, MASK };

static uint32_t g_seed = 20261120u;
static float rnd(void) {
    g_seed = g_seed * 1664525u + 1013904223u;
    return (float)(g_seed >> 8) / 8388608.0f - 1.0f;
}

/* The host checks every step-th token and head, and always the last one. */
static uint32_t sample_next(uint32_t i, uint32_t step, uint32_t n) {
    return i + step < n || i + 1 == n ? i + step : n - 1;
}

struct image_span { uint32_t start, tokens; };

struct attention_case {
    const char *name;
    uint32_t pos0, n_tokens, n_head, ratio;
    uint32_t raw_cap, raw_start;       /* ring geometry; n_raw is derived */
    selection select;
    std::vector<image_span> images;
    uint32_t token_step, head_step;    /* part of the output the host checks */
};

static double run_case(const attention_case &c, const void *model, uint64_t model_size,
                       const float *sinks, bool expect_no_scratch) {
    const uint32_t n_tokens = c.n_tokens, n_head = c.n_head;
    const uint32_t end = c.pos0 + n_tokens;
    const uint32_t n_raw = end < c.raw_cap ? end : c.raw_cap;
    const uint32_t first_raw_pos = end - n_raw;
    const uint32_t n_comp = c.ratio ? end / c.ratio : 0;
    CHECK(n_raw >= n_tokens && c.raw_start < c.raw_cap);
    CHECK(c.select == ALL || n_comp > TOP_K);

    std::vector<int32_t> tokens(n_tokens, 1);
    for (const image_span &im : c.images) {
        CHECK(im.tokens >= 2 && im.start + im.tokens <= n_tokens);
        tokens[im.start] = VOCAB + DS4_DEEPSEEK4_IMAGE_START;
        for (uint32_t i = 1; i + 1 < im.tokens; i++)
            tokens[im.start + i] = VOCAB + DS4_DEEPSEEK4_IMAGE;
        tokens[im.start + im.tokens - 1] = VOCAB + DS4_DEEPSEEK4_IMAGE_END;
    }
    std::vector<uint32_t> bounds((size_t)n_tokens * 2);
    CHECK(ds4_deepseek4_attention_bounds(tokens.data(), n_tokens, VOCAB, c.pos0,
                                         n_raw, WINDOW, bounds.data()));

    const size_t q_count = (size_t)n_tokens * n_head * DIM;
    std::vector<float> q(q_count), raw((size_t)c.raw_cap * DIM);
    std::vector<float> comp((size_t)(n_comp ? n_comp : 1) * DIM);
    for (float &v : q) v = 4.0f * rnd();
    for (float &v : raw) v = rnd();
    for (float &v : comp) v = rnd();

    /* The selection of each query: distinct rows, some of them still in the
     * future of the query and some slots empty, as a short context leaves
     * them.  Neither may be attended. */
    std::vector<uint32_t> rows;
    std::vector<float> mask;
    if (c.select != ALL) {
        rows.resize((size_t)n_tokens * TOP_K);
        std::vector<uint8_t> used(n_comp);
        for (uint32_t t = 0; t < n_tokens; t++) {
            std::fill(used.begin(), used.end(), 0);
            for (uint32_t i = 0; i < TOP_K; i++) {
                uint32_t row;
                do {
                    g_seed = g_seed * 1664525u + 1013904223u;
                    row = (g_seed >> 4) % n_comp;
                } while (used[row]);
                used[row] = 1;
                rows[(size_t)t * TOP_K + i] = i % 97u == 96u ? UINT32_MAX : row;
            }
        }
        if (c.select == MASK) {
            mask.assign((size_t)n_tokens * n_comp, -INFINITY);
            for (uint32_t t = 0; t < n_tokens; t++) {
                for (uint32_t i = 0; i < TOP_K; i++) {
                    const uint32_t row = rows[(size_t)t * TOP_K + i];
                    if (row != UINT32_MAX) mask[(size_t)t * n_comp + row] = 0.0f;
                }
            }
        }
    }

    ds4_gpu_tensor *tq = ds4_gpu_tensor_alloc(q_count * sizeof(float));
    ds4_gpu_tensor *tout = ds4_gpu_tensor_alloc(q_count * sizeof(float));
    ds4_gpu_tensor *traw = ds4_gpu_tensor_alloc(raw.size() * sizeof(float));
    ds4_gpu_tensor *tcomp = ds4_gpu_tensor_alloc(comp.size() * sizeof(float));
    ds4_gpu_tensor *trows = ds4_gpu_tensor_alloc((rows.size() + 1) * sizeof(uint32_t));
    ds4_gpu_tensor *tmask = ds4_gpu_tensor_alloc((mask.size() + 1) * sizeof(float));
    CHECK(tq && tout && traw && tcomp && trows && tmask);
    CHECK(ds4_gpu_tensor_write(tq, 0, q.data(), q_count * sizeof(float)));
    CHECK(ds4_gpu_tensor_write(traw, 0, raw.data(), raw.size() * sizeof(float)));
    CHECK(ds4_gpu_tensor_write(tcomp, 0, comp.data(), comp.size() * sizeof(float)));
    if (!rows.empty())
        CHECK(ds4_gpu_tensor_write(trows, 0, rows.data(), rows.size() * sizeof(uint32_t)));
    if (!mask.empty())
        CHECK(ds4_gpu_tensor_write(tmask, 0, mask.data(), mask.size() * sizeof(float)));
    CHECK(ds4_gpu_tensor_fill_f32(tout, 12345.0f, q_count));
    CHECK(ds4_gpu_synchronize());

    size_t free_before = 0, free_after = 0, total = 0;
    CHECK(hipMemGetInfo(&free_before, &total) == hipSuccess);
    int ok;
    if (c.select == ROWS) {
        ok = ds4_gpu_attention_visual_indexed_batch_heads_tensor(
                tout, model, model_size, 0, tq, traw, tcomp, 0, trows, TOP_K,
                tokens.data(), VOCAB, n_tokens, c.pos0, n_raw, c.raw_cap,
                c.raw_start, n_comp, WINDOW, c.ratio, n_head, DIM);
    } else {
        ok = ds4_gpu_attention_visual_mixed_batch_heads_tensor(
                tout, model, model_size, 0, tq, traw, n_comp ? tcomp : NULL, 0,
                c.select == MASK ? tmask : NULL, c.select == MASK,
                tokens.data(), VOCAB, n_tokens, c.pos0, n_raw, c.raw_cap,
                c.raw_start, n_comp, WINDOW, c.ratio, n_head, DIM);
    }
    CHECK(ok);
    CHECK(ds4_gpu_synchronize());
    CHECK(hipMemGetInfo(&free_after, &total) == hipSuccess);
    if (expect_no_scratch) {
        const size_t taken = free_before > free_after ? free_before - free_after : 0;
        if (taken > ((size_t)256 << 20)) {
            fprintf(stderr, "%s: the launch took %.1f MiB of device memory\n",
                    c.name, (double)taken / 1048576.0);
            exit(1);
        }
    }
    std::vector<float> actual(q_count);
    CHECK(ds4_gpu_tensor_read(tout, 0, actual.data(), q_count * sizeof(float)));

    double max_error = 0.0;
    std::vector<double> weight;
    std::vector<const float *> key;
    const double scale = 1.0 / sqrt((double)DIM);
    for (uint32_t t = 0; t < n_tokens; t = sample_next(t, c.token_step, n_tokens)) {
        const uint32_t qpos = c.pos0 + t;
        key.clear();
        for (uint32_t kpos = bounds[2 * t]; kpos <= bounds[2 * t + 1]; kpos++) {
            const uint32_t physical = (c.raw_start + kpos - first_raw_pos) % c.raw_cap;
            key.push_back(raw.data() + (size_t)physical * DIM);
        }
        const uint32_t visible = c.ratio ? std::min((qpos + 1) / c.ratio, n_comp) : 0;
        if (c.select == ALL) {
            for (uint32_t row = 0; row < visible; row++)
                key.push_back(comp.data() + (size_t)row * DIM);
        } else {
            for (uint32_t i = 0; i < TOP_K; i++) {
                const uint32_t row = rows[(size_t)t * TOP_K + i];
                if (row < visible) key.push_back(comp.data() + (size_t)row * DIM);
            }
        }
        for (uint32_t h = 0; h < n_head; h = sample_next(h, c.head_step, n_head)) {
            const float *qh = q.data() + ((size_t)t * n_head + h) * DIM;
            weight.resize(key.size());
            double top = sinks[h];
            for (size_t k = 0; k < key.size(); k++) {
                double dot = 0.0;
                for (uint32_t d = 0; d < DIM; d++) dot += (double)qh[d] * key[k][d];
                weight[k] = dot * scale;
                top = std::max(top, weight[k]);
            }
            double sum = exp(sinks[h] - top);
            for (double &w : weight) { w = exp(w - top); sum += w; }
            const float *out = actual.data() + ((size_t)t * n_head + h) * DIM;
            for (uint32_t d = 0; d < DIM; d++) {
                double expected = 0.0;
                for (size_t k = 0; k < key.size(); k++) expected += weight[k] * key[k][d];
                expected /= sum;
                const double error = fabs(expected - (double)out[d]);
                if (!(error <= 2e-5)) {
                    fprintf(stderr, "%s: token %u head %u dim %u: %.9g, expected %.9g "
                            "(%zu keys)\n", c.name, t, h, d, out[d], expected, key.size());
                    exit(1);
                }
                max_error = std::max(max_error, error);
            }
        }
    }

    ds4_gpu_tensor_free(tmask); ds4_gpu_tensor_free(trows); ds4_gpu_tensor_free(tcomp);
    ds4_gpu_tensor_free(traw); ds4_gpu_tensor_free(tout); ds4_gpu_tensor_free(tq);
    fprintf(stderr, "test_rocm_visual_attention: %-34s max_abs=%.3g\n", c.name, max_error);
    return max_error;
}

int main() {
    CHECK(ds4_gpu_init());
    const uint64_t page = (uint64_t)getpagesize();
    void *model = NULL;
    CHECK(posix_memalign(&model, (size_t)page, (size_t)page) == 0);
    memset(model, 0, (size_t)page);
    float *sinks = (float *)model;
    for (uint32_t h = 0; h < 64; h++) sinks[h] = 2.0f * rnd();
    CHECK(ds4_gpu_set_model_map(model, page));

    /* An image of 900 tokens is longer than the 383 rows to the left and 384
     * to the right an image token may see, so both limits are exercised; the
     * rings are wrapped. */
    const attention_case cases[] = {
        {"1M context, selected rows", 999936, 64, 64, 4, 192, 150, ROWS, {{10, 40}}, 9, 21},
        {"raw only, first chunk", 0, 1100, 64, 0, 1100, 0, ALL, {{60, 900}, {1000, 30}}, 53, 9},
        {"raw only, ring", 5000, 1100, 64, 0, 1228, 700, ALL, {{0, 900}, {1040, 60}}, 53, 9},
        {"ratio 128, 120k context", 120000, 300, 64, 128, 428, 400, ALL, {{20, 200}}, 29, 13},
        {"ratio 4, every row", 1000, 300, 33, 4, 428, 17, ALL, {{100, 150}}, 29, 8},
        {"ratio 4, first chunk, selected", 0, 4096, 2, 4, 4096, 0, ROWS, {{5, 900}, {3000, 600}}, 211, 1},
        {"ratio 4, first chunk, mask", 0, 2100, 2, 4, 2100, 0, MASK, {{5, 900}}, 211, 1},
        {"ratio 4, 100k context, selected", 100000, 300, 64, 4, 428, 301, ROWS, {{0, 120}, {130, 170}}, 29, 13},
        {"ratio 4, 100k context, mask", 100000, 300, 33, 4, 428, 301, MASK, {{0, 120}, {130, 170}}, 29, 8},
    };
    for (size_t i = 0; i < sizeof(cases) / sizeof(cases[0]); i++)
        run_case(cases[i], model, page, sinks, i == 0);
    ds4_gpu_cleanup();
    free(model);
    printf("test_rocm_visual_attention: ok\n");
    return 0;
}
