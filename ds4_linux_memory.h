#ifndef DS4_LINUX_MEMORY_H
#define DS4_LINUX_MEMORY_H

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>

/* CMA pages can back movable allocations, but not ordinary pinned GPU pages. */
static inline bool ds4_linux_memory_parse(FILE *fp, uint64_t *available, uint64_t *cma_free) {
    char line[256];
    bool found = false;
    *available = 0;
    *cma_free = 0;
    while (fgets(line, sizeof(line), fp)) {
        unsigned long long kib;
        if (sscanf(line, "MemAvailable: %llu kB", &kib) == 1) {
            if (kib > UINT64_MAX / 1024u) return false;
            *available = (uint64_t)kib * 1024u;
            found = true;
        } else if (sscanf(line, "CmaFree: %llu kB", &kib) == 1) {
            if (kib > UINT64_MAX / 1024u) return false;
            *cma_free = (uint64_t)kib * 1024u;
        }
    }
    return found && !ferror(fp);
}

static inline bool ds4_linux_nonmovable_memory_parse(FILE *fp, uint64_t *bytes) {
    uint64_t available, cma;
    if (!ds4_linux_memory_parse(fp, &available, &cma)) return false;
    *bytes = available > cma ? available - cma : 0;
    return true;
}

static inline bool ds4_linux_nonmovable_memory(uint64_t *bytes) {
    FILE *fp = fopen("/proc/meminfo", "r");
    if (!fp) return false;
    const bool ok = ds4_linux_nonmovable_memory_parse(fp, bytes);
    fclose(fp);
    return ok;
}

/* Free memory only movable allocations (managed GPU memory) can take. */
static inline uint64_t ds4_linux_cma_free_memory(void) {
    FILE *fp = fopen("/proc/meminfo", "r");
    if (!fp) return 0;
    uint64_t available, cma;
    const bool ok = ds4_linux_memory_parse(fp, &available, &cma);
    fclose(fp);
    return ok && cma < available ? cma : 0;
}

#endif
