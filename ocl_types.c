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

cl_int anactx_create(anactx *anactx, cl_platform_id platform_id, cl_device_id device_id) {
    anactx->platform_id = platform_id;
    anactx->device_id = device_id;

    anactx->tasks_in_buffer_count = 0;
    anactx->hashes_num = 0;
    anactx->hashes_reversed = NULL;
    anactx->cur_exec_kernel = NULL;
    anactx->tasks_buffer = calloc(PERMUT_TEMPLATES_SIZE, sizeof(permut_task));

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

cl_int anactx_set_input_hashes(anactx *anactx, uint32_t *hashes, uint32_t hashes_num) {
    anactx->hashes_num = hashes_num;

    anactx->hashes_reversed = malloc(hashes_num * MAX_STR_LENGTH);
    ret_ifnz(!anactx->hashes_reversed);

    cl_int errcode;
    anactx->mem_hashes = clCreateBuffer(anactx->cl_ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, hashes_num*16, hashes, &errcode);
    ret_ifnz(errcode);
    anactx->mem_hashes_reversed = clCreateBuffer(anactx->cl_ctx, CL_MEM_WRITE_ONLY, hashes_num * MAX_STR_LENGTH, NULL, &errcode);
    ret_ifnz(errcode);

    return CL_SUCCESS;
}

const uint32_t* anactx_read_hashes_reversed(anactx *anactx, cl_int *errcode) {
    *errcode = clEnqueueReadBuffer (anactx->queue, anactx->mem_hashes_reversed, CL_TRUE, 0, anactx->hashes_num * MAX_STR_LENGTH, anactx->hashes_reversed, 0, NULL, NULL);
    if (*errcode) {
        return NULL;
    }
    return anactx->hashes_reversed;
}

cl_int anactx_free(anactx *anactx) {
    if (anactx->hashes_reversed) {
        free(anactx->hashes_reversed);
    }

    free(anactx->tasks_buffer);

    cl_int errcode = CL_SUCCESS;
    errcode |= clReleaseMemObject(anactx->mem_hashes);
    errcode |= clReleaseMemObject(anactx->mem_hashes_reversed);
    errcode |= clReleaseCommandQueue (anactx->queue);
    errcode |= clReleaseProgram(anactx->program);
    errcode |= clReleaseContext(anactx->cl_ctx);

    return errcode;
}

cl_int anakrnl_permut_create(anakrnl_permut *anakrnl, anactx *anactx, uint32_t iters_per_item, permut_task *templates, uint32_t num_templates) {
    cl_int errcode;

    anakrnl->ctx = anactx;
    anakrnl->iters_per_item = iters_per_item;
    anakrnl->num_templates = num_templates;
    anakrnl->templates = templates;

    anakrnl->kernel = clCreateKernel(anactx->program, "permut", &errcode);
    ret_ifnz(errcode);

    anakrnl->mem_permut_templates = clCreateBuffer(anactx->cl_ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, num_templates*sizeof(permut_task), templates, &errcode);
    ret_ifnz(errcode);

/*
        __kernel void permut(
            __global const permut_task *permut_templates,
                     const uint iters_per_item,
            __global uint *hashes,
                     uint hashes_num,
            __global uint *hashes_reversed)
*/
    errcode |= clSetKernelArg(anakrnl->kernel, 0, sizeof (cl_mem), &anakrnl->mem_permut_templates);
    errcode |= clSetKernelArg(anakrnl->kernel, 1, sizeof (iters_per_item), &iters_per_item);
    errcode |= clSetKernelArg(anakrnl->kernel, 2, sizeof (cl_mem), &anactx->mem_hashes);
    errcode |= clSetKernelArg(anakrnl->kernel, 3, sizeof (anactx->hashes_num), &anactx->hashes_num);
    errcode |= clSetKernelArg(anakrnl->kernel, 4, sizeof (cl_mem), &anactx->mem_hashes_reversed);

    return errcode;
}

cl_int anakrnl_permut_enqueue(anakrnl_permut *anakrnl) {
    size_t globalWorkSize[] = {anakrnl->num_templates, 0, 0};
    return clEnqueueNDRangeKernel(anakrnl->ctx->queue, anakrnl->kernel, 1, NULL, globalWorkSize, NULL, 0, NULL, &anakrnl->event);
}

cl_int anakrnl_permut_wait(anakrnl_permut *anakrnl) {
    return clWaitForEvents(1, &anakrnl->event);
}

cl_int anakrnl_permut_free(anakrnl_permut *anakrnl) {
    clReleaseMemObject(anakrnl->mem_permut_templates);
    clReleaseKernel(anakrnl->kernel);
}