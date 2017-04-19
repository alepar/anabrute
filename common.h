#ifndef ANABRUTE_CONSTANTS_H
#define ANABRUTE_CONSTANTS_H

#include <errno.h>
#include <pthread.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#ifdef __APPLE__
    #include "OpenCL/opencl.h"
    #include <event.h>
    #include <unitypes.h>
#else
    #include <sys/time.h>
    #include "CL/cl.h"
#endif

#define MAX_WORD_LENGTH 5

// cpu<->gpu tasks buffers length
#define TASKS_BUFFERS_SIZE 4

// defines task size for gpu cruncher
// peak at ~256-512K, try lowering if kernel times out
#define PERMUT_TASKS_IN_KERNEL_TASK 16*1024
// peak at ~512, try lowering if kernel times out
#define MAX_ITERS_IN_KERNEL_TASK 256

// has to be in sync with constants in permut.cl
#define MAX_STR_LENGTH 40
#define MAX_OFFSETS_LENGTH 16 // should always have 1 extra for 0-terminated

#define TIMES_WINDOW_LENGTH 32
#define REFRESH_INTERVAL_HASHES_REVERSED_MILLIS 1000

#define ret_iferr(val, msg) \
if (val) {\
    fprintf(stderr, "FATAL: %d - %s\n", val, msg);\
    return val;\
}

#endif //ANABRUTE_CONSTANTS_H
