#include <string.h>
#include <stdio.h>

#include "ocl_types.h"

#define ret_ifnz(errcode) if (errcode) return errcode;

// private stuff

char* read_file(const char* filename) {
    FILE *fd = fopen(filename, "r");
    if (fd == NULL) {
        return NULL;
    }

    fseek(fd, 0L, SEEK_END);
    const size_t filesize = (size_t) ftell(fd);
    rewind(fd);

    char *buf = (char *)malloc(filesize+1);
    const size_t read = fread(buf, 1, filesize, fd);

    if (filesize != read) {
        free(buf);
        return NULL;
    }

    return buf;
}

// public stuff

cl_int ana_threadctx_create(ana_threadctx *anactx, cl_platform_id platform_id, cl_device_id device_id) {
    anactx->platform_id = platform_id;
    anactx->device_id = device_id;

    cl_int errcode;
    const cl_context_properties ctx_props [] = { CL_CONTEXT_PLATFORM, platform_id, 0, 0 };
    anactx->cl_ctx = clCreateContext(ctx_props, 1, &device_id, NULL, NULL, &errcode);
    ret_ifnz(errcode);

    char *const kernel_source = read_file("kernels/permut.cl");
    ret_ifnz(!kernel_source);
    size_t lengths[] = {strlen(kernel_source)};
    const char *sources[] = {kernel_source};
    anactx->program = clCreateProgramWithSource(anactx->cl_ctx, 1, sources, lengths, &errcode);
    ret_ifnz(errcode);
    errcode = clBuildProgram(anactx->program, 0, NULL, NULL, NULL, NULL);
    if (errcode == CL_BUILD_PROGRAM_FAILURE) {
        // Determine the size of the log
        size_t log_size;
        clGetProgramBuildInfo(anactx->program, anactx->device_id, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        // Allocate memory for the log
        char *log = (char *) malloc(log_size);
        // Get the log
        clGetProgramBuildInfo(anactx->program, anactx->device_id, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
        // Print the log
        fprintf(stderr, "kernel compilation failed, see compiler output below\n------\n%s\n------\n", log);
    }
    ret_ifnz(errcode);

    anactx->queue = clCreateCommandQueue(anactx->cl_ctx, anactx->device_id, NULL, &errcode);
    ret_ifnz(errcode);

    return CL_SUCCESS;
}

cl_int ana_threadctx_free(ana_threadctx *anactx) {
    cl_int errcode = CL_SUCCESS;
    errcode |= clReleaseCommandQueue (anactx->queue);
    errcode |= clReleaseProgram(anactx->program);
    errcode |= clReleaseContext(anactx->cl_ctx);
    return errcode;
}