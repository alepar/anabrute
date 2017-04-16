#include "os.h"

uint32_t num_cpu_cores() {
    // posix-way
    return (uint32_t) sysconf(_SC_NPROCESSORS_ONLN);
}
