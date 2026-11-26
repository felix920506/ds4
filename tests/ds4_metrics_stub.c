/* Phase counters for the standalone backend tests.  ds4_cuda.o and ds4_metal.o call
 * ds4_metric_add() and ds4_metric_now_ns(), which ds4.c defines.  Tests that link a
 * backend without ds4.o use these no-op versions; never link this with ds4.o. */
#include "../ds4_metrics.h"

void ds4_metric_add(ds4_metric m, uint64_t v) {
    (void)m;
    (void)v;
}

uint64_t ds4_metric_now_ns(void) {
    return 0;
}
