/* Vision encoder benchmark.
 *
 * Times the images given on the command line two ways, each over several
 * passes in one process:
 *
 *   single   one image per call, the latency of a one-image request;
 *   request  all images in one ds4_engine_vision_encode_memory_batch() call,
 *            the way a request with several images reaches the encoder.
 *
 * The first image of the process is reported apart: it pays for mapping the
 * encoder weights and loading the kernels.  With --dump DIR the embeddings
 * are written as raw F32 so two builds can be compared; --passes 0 encodes
 * every image once and times nothing.  The language model is opened
 * inspect-only, so a run takes seconds. */

#include "ds4.h"

#include <errno.h>
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1.0e6;
}

static const char *base_name(const char *path) {
    const char *slash = strrchr(path, '/');
    return slash ? slash + 1 : path;
}

static uint8_t *read_file(const char *path, size_t *len) {
    FILE *fp = fopen(path, "rb");
    uint8_t *data = NULL;
    long size = 0;
    if (fp && !fseek(fp, 0, SEEK_END) && (size = ftell(fp)) > 0 &&
        !fseek(fp, 0, SEEK_SET) && (data = malloc((size_t)size)) &&
        fread(data, 1, (size_t)size, fp) != (size_t)size) {
        free(data);
        data = NULL;
    }
    if (fp) fclose(fp);
    if (!data) fprintf(stderr, "%s: %s\n", path, errno ? strerror(errno) : "empty");
    *len = (size_t)size;
    return data;
}

/* Every pass must reproduce the first embedding of an image bit for bit. */
static bool same_embedding(const ds4_vision_embedding *a,
                           const ds4_vision_embedding *b, size_t width) {
    const size_t values = (size_t)a->token_count * width;
    if (a->token_count != b->token_count) return false;
    for (size_t i = 0; i < values; i++) {
        if (!isfinite(b->data[i])) return false;
    }
    return !memcmp(a->data, b->data, values * sizeof(float));
}

int main(int argc, char **argv) {
    const char *dump = NULL;
    int passes = 5;
    int arg = 1;
    for (; arg + 1 < argc && argv[arg][0] == '-'; arg += 2) {
        if (!strcmp(argv[arg], "--dump")) dump = argv[arg + 1];
        else if (!strcmp(argv[arg], "--passes")) passes = atoi(argv[arg + 1]);
        else break;
    }
    if (argc - arg < 3 || passes < 0) {
        fprintf(stderr, "usage: %s [--passes N] [--dump DIR] "
                        "MAIN.gguf VISION.gguf IMAGE...\n", argv[0]);
        return 2;
    }
    char **paths = argv + arg + 2;
    const size_t count = (size_t)(argc - arg - 2);

    ds4_engine_options options = {0};
    options.model_path = argv[arg];
    options.vision_path = argv[arg + 1];
#ifdef __APPLE__
    options.backend = DS4_BACKEND_METAL;
#else
    options.backend = DS4_BACKEND_CUDA;
#endif
    options.inspect_only = true;
    ds4_engine *engine = NULL;
    if (ds4_engine_open(&engine, &options) != 0) return 1;
    const size_t width = (size_t)ds4_engine_embd_dim(engine);

    const uint8_t **encoded = calloc(count, sizeof(*encoded));
    size_t *encoded_len = calloc(count, sizeof(*encoded_len));
    ds4_vision_embedding *first = calloc(count, sizeof(*first));
    ds4_vision_embedding *out = calloc(count, sizeof(*out));
    double *single_ms = calloc(count, sizeof(*single_ms));
    if (!encoded || !encoded_len || !first || !out || !single_ms) return 1;
    for (size_t i = 0; i < count; i++) {
        if (!(encoded[i] = read_file(paths[i], &encoded_len[i]))) return 1;
    }

    char error[256] = {0};
    double cold_ms = 0.0, single_set_ms = 0.0, request_ms = 0.0;
    double request_best_ms = 0.0;
    int status = 0;
    for (int pass = 0; pass <= passes && !status; pass++) {
        for (size_t i = 0; i < count && !status; i++) {
            const double t0 = now_ms();
            if (!ds4_engine_vision_encode_memory(engine, encoded[i], encoded_len[i],
                                                 &out[i], error, sizeof(error))) {
                fprintf(stderr, "%s: %s\n", paths[i], error);
                status = 1;
                break;
            }
            const double ms = now_ms() - t0;
            if (pass == 0 && i == 0) cold_ms = ms;
            if (pass == 0) {
                first[i] = out[i];   /* pass 0 warms up and is the reference */
                continue;
            }
            single_ms[i] += ms;
            single_set_ms += ms;
            if (!same_embedding(&first[i], &out[i], width)) {
                fprintf(stderr, "%s: embedding is not repeatable\n", paths[i]);
                status = 1;
            }
            ds4_vision_embedding_free(&out[i]);
        }
        if (pass == 0 || status) continue;
        const double t0 = now_ms();
        if (!ds4_engine_vision_encode_memory_batch(engine, encoded, encoded_len,
                                                   count, out, error, sizeof(error))) {
            fprintf(stderr, "request: %s\n", error);
            status = 1;
            break;
        }
        const double ms = now_ms() - t0;
        request_ms += ms;
        if (pass == 1 || ms < request_best_ms) request_best_ms = ms;
        for (size_t i = 0; i < count; i++) {
            if (!same_embedding(&first[i], &out[i], width)) {
                fprintf(stderr, "%s: request embedding differs from single\n",
                        paths[i]);
                status = 1;
            }
            ds4_vision_embedding_free(&out[i]);
        }
    }
    if (!status) {
        for (size_t i = 0; i < count; i++) {
            printf("%-28s %5ux%-5u grid %2ux%-2u %4u tokens %8.1f ms\n",
                   base_name(paths[i]), first[i].width, first[i].height,
                   first[i].grid_width, first[i].grid_height,
                   first[i].token_count, passes ? single_ms[i] / passes : 0.0);
        }
        printf("cold first image: %.1f ms\n", cold_ms);
        if (passes) {
            printf("%zu images, one per call: %.1f ms\n",
                   count, single_set_ms / passes);
            printf("%zu images, one request:  %.1f ms (best %.1f ms, "
                   "%.1f ms/image)\n", count, request_ms / passes,
                   request_best_ms, request_ms / passes / (double)count);
        }
        for (size_t i = 0; dump && i < count && !status; i++) {
            char path[4096];
            snprintf(path, sizeof(path), "%s/%s.f32", dump, base_name(paths[i]));
            FILE *fp = fopen(path, "wb");
            const size_t values = (size_t)first[i].token_count * width;
            if (!fp || fwrite(first[i].data, sizeof(float), values, fp) != values) {
                fprintf(stderr, "cannot write %s\n", path);
                status = 1;
            }
            if (fp) fclose(fp);
        }
    }
    for (size_t i = 0; i < count; i++) {
        ds4_vision_embedding_free(&first[i]);
        free((void *)encoded[i]);
    }
    free(single_ms);
    free(out);
    free(first);
    free(encoded_len);
    free(encoded);
    ds4_engine_close(engine);
    return status;
}
