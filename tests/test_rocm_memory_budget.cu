/* A session admitted on ds4_gpu_memory_ceiling() must be able to grow to its
 * limit through ds4_gpu_memory_shortfall().  With no optional cache and no
 * elastic experts both measure the same free memory, so what the ceiling
 * admits has to fit, and not much more. */
#include "ds4_gpu.h"
#include <cstdio>
#include <cstdlib>
#define CHECK(x) do { if (!(x)) { fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #x); exit(1); } } while (0)
bool ds4_log_is_tty(FILE *) { return false; }

int main() {
    CHECK(ds4_gpu_init());
    /* Other processes may move free memory between the calls. */
    const uint64_t slack = 256ull << 20;
    const uint64_t ceiling = ds4_gpu_memory_ceiling(0);
    CHECK(ceiling > slack);
    const uint64_t below = ds4_gpu_memory_shortfall(ceiling - slack, 0);
    const uint64_t above = ds4_gpu_memory_shortfall(ceiling + slack, 0);
    printf("ceiling %.2f GiB: shortfall %.0f MiB at ceiling - 256 MiB, %.0f MiB at ceiling + 256 MiB\n",
           (double)ceiling / (1ull << 30), (double)below / (1 << 20), (double)above / (1 << 20));
    CHECK(below == 0);
    CHECK(above > 0);
    ds4_gpu_cleanup();
    puts("PASS: admission ceiling and growth shortfall agree");
}
