/* A growing context evicts F16 cache entries down to a floor of 512 MiB.  With
 * an oldest entry larger than the room above the floor, eviction must still
 * free the smaller entries behind it, and once it has, nothing may be left
 * reported as evictable. */
#include "ds4_gpu.h"
#include <cstdio>
#include <cstdlib>
#include <vector>
#define CHECK(x) do { if (!(x)) { fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #x); exit(1); } } while (0)
bool ds4_log_is_tty(FILE *) { return false; }

constexpr uint64_t MiB = 1ull << 20;

int main() {
    CHECK(ds4_gpu_init());
    /* One 1024 x 32768 weight (64 MiB as F16), then 31 of 4096 x 2048
     * (16 MiB): 560 MiB cached, the 64 MiB one least recently used. */
    struct { uint64_t in, out; } shapes[32];
    shapes[0] = {1024, 32768};
    for (int i = 1; i < 32; i++) shapes[i] = {4096, 2048};
    uint64_t offsets[32], size = 0;
    for (int i = 0; i < 32; i++) {
        offsets[i] = size;
        size += shapes[i].out * (shapes[i].in / 32 * 34);
    }
    std::vector<unsigned char> model(size, 0);
    CHECK(ds4_gpu_set_model_map(model.data(), model.size()));
    CHECK(ds4_gpu_cache_model_range(model.data(), model.size(), 0, model.size(), "q8_0"));
    for (int i = 0; i < 32; i++) {
        const uint64_t bytes = shapes[i].out * (shapes[i].in / 32 * 34);
        CHECK(ds4_gpu_cache_q8_f16_range(model.data(), model.size(), offsets[i], bytes,
                                         shapes[i].in, shapes[i].out, "q8_0"));
    }

    const uint64_t first = ds4_gpu_reclaim_memory(16 * MiB);
    const uint64_t rest = ds4_gpu_reclaim_memory(1024 * MiB);
    const uint64_t after = ds4_gpu_reclaim_memory(1024 * MiB);
    printf("reclaim 16 MiB: freed %llu MiB; 1 GiB: freed %llu MiB; again: freed %llu MiB\n",
           (unsigned long long)(first / MiB), (unsigned long long)(rest / MiB),
           (unsigned long long)(after / MiB));
    CHECK(first == 16 * MiB);
    CHECK(rest == 32 * MiB);
    CHECK(after == 0);
    ds4_gpu_cleanup();
    puts("PASS: F16 cache eviction skips an oversized LRU entry and stops at the floor");
}
