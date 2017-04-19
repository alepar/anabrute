#include "os.h"

uint32_t num_cpu_cores() {
    // posix-way
    return (uint32_t) sysconf(_SC_NPROCESSORS_ONLN);
}

uint64_t current_micros() {
    struct timeval t;
    gettimeofday(&t, 0);
    return (uint64_t) t.tv_sec*1000000L + t.tv_usec;
}