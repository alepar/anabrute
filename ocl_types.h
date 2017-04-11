#ifndef OPENCL_TEST_OCL_TYPES_H
#define OPENCL_TEST_OCL_TYPES_H

#ifdef __APPLE__
    #include "OpenCL/opencl.h"
#else
    #include "CL/cl.h"
#endif

typedef struct {
    cl_platform_id platform_id;
    cl_device_id device_id;

    cl_context cl_ctx;
    cl_program program;

    cl_command_queue queue;

    // TODO hashes in-out buffer
    // TODO queues for kernel execution

    // TODO stats
} ana_threadctx;

cl_int ana_threadctx_create(ana_threadctx *anactx, cl_platform_id platform_id, cl_device_id device_id);
cl_int ana_threadctx_free(ana_threadctx *anactx);

// TODO add to queue
// TODO make queue available
// TODO process events routine

#endif //OPENCL_TEST_OCL_TYPES_H
