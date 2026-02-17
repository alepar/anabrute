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
    #include <event.h>
    #include <unitypes.h>
    #ifdef HAVE_OPENCL
        #include "OpenCL/opencl.h"
    #endif
#else
    #include <sys/time.h>
    #ifdef HAVE_OPENCL
        #include "CL/cl.h"
    #endif
#endif

/* Provide stub OpenCL types when OpenCL is not available */
#ifndef HAVE_OPENCL
    typedef int cl_int;
    typedef unsigned int cl_uint;
    typedef void* cl_mem;
    typedef void* cl_platform_id;
    typedef void* cl_device_id;
    typedef void* cl_context;
    typedef void* cl_program;
    typedef void* cl_command_queue;
    typedef void* cl_kernel;
    typedef void* cl_event;
    typedef unsigned long cl_device_type;
    typedef intptr_t cl_context_properties;
    #define CL_SUCCESS 0
    #define CL_DEVICE_TYPE_ALL 0xFFFFFFFF
    #define CL_DEVICE_TYPE_CPU (1 << 1)
#endif

#define MAX_WORD_LENGTH 7

// cpu<->gpu tasks buffers length, main RAM consumer
#define TASKS_BUFFERS_SIZE 64

// defines task size for gpu cruncher
// peak at ~256-512K, try lowering if kernel times out
#define PERMUT_TASKS_IN_KERNEL_TASK 256*1024
// peak at ~512, try lowering if kernel times out
#define MAX_ITERS_IN_KERNEL_TASK 512

// has to be in sync with constants in permut.cl
#define MAX_STR_LENGTH 40
#define MAX_OFFSETS_LENGTH 16 // should always have 1 extra for 0-terminated

#define TIMES_WINDOW_LENGTH 32
#define REFRESH_INTERVAL_HASHES_REVERSED_MILLIS 10000

#define ret_iferr(val, msg) \
if (val) {\
    fprintf(stderr, "FATAL: %d - %s\n", val, msg);\
    return val;\
}

#endif //ANABRUTE_CONSTANTS_H
