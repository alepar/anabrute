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

// defines task size for gpu cruncher
// peak at ~256-512K
#define PERMUT_TASKS_IN_BATCH 256*1024
// peak at ~512
#define MAX_ITERS_PER_TASK 512
// 512*512*1024 == 256M (11! < 256M < 12!)

// has to be in sync with constants in permut.cl
#define MAX_STR_LENGTH 40
#define MAX_OFFSETS_LENGTH 20 // should always have 1 extra for 0-terminated



#define ret_iferr(val, msg) \
if (val) {\
    fprintf(stderr, "FATAL: %d - %s\n", val, msg);\
    return val;\
}

#endif //ANABRUTE_CONSTANTS_H
