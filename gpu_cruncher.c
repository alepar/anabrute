#include <string.h>
#include <stdio.h>

#include "gpu_cruncher.h"
#include "hashes.h"

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

cl_int gpu_cruncher_ctx_create(gpu_cruncher_ctx *ctx, cl_platform_id platform_id, cl_device_id device_id, tasks_buffers* tasks_buffs) {
    ctx->platform_id = platform_id;
    ctx->device_id = device_id;
    ctx->tasks_buffs = tasks_buffs;

    ctx->tasks_in_buffer_count = 0;
    ctx->hashes_num = 0;
    ctx->hashes_seen = 0;
    ctx->hashes_reversed = NULL;
    ctx->cur_exec_kernel = NULL;
//    ctx->tasks_buffer = calloc(PERMUT_TASKS_IN_BATCH, sizeof(permut_task));

    cl_int errcode;
    const cl_context_properties ctx_props [] = { CL_CONTEXT_PLATFORM, platform_id, 0, 0 };
    ctx->cl_ctx = clCreateContext(ctx_props, 1, &device_id, NULL, NULL, &errcode);
    ret_iferr(errcode, "failed to create context");

    char *const kernel_source = read_file("kernels/permut.cl");
    ret_iferr(!kernel_source, "failed to read kernel source");
    size_t lengths[] = {strlen(kernel_source)};
    const char *sources[] = {kernel_source};
    ctx->program = clCreateProgramWithSource(ctx->cl_ctx, 1, sources, lengths, &errcode);
    ret_iferr(errcode, "failed to create program");
    errcode = clBuildProgram(ctx->program, 0, NULL, NULL, NULL, NULL);
    if (errcode == CL_BUILD_PROGRAM_FAILURE) {
        // Determine the size of the log
        size_t log_size;
        clGetProgramBuildInfo(ctx->program, ctx->device_id, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        // Allocate memory for the log
        char *log = (char *) malloc(log_size);
        // Get the log
        clGetProgramBuildInfo(ctx->program, ctx->device_id, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
        // Print the log
        fprintf(stderr, "kernel compilation failed, see compiler output below\n------\n%s\n------\n", log);
    }
    ret_iferr(errcode, "failed to build program");

    ctx->queue = clCreateCommandQueue(ctx->cl_ctx, ctx->device_id, NULL, &errcode);
    ret_iferr(errcode, "failed to create queue");

    return CL_SUCCESS;
}

cl_int gpu_cruncher_ctx_set_input_hashes(gpu_cruncher_ctx *ctx, uint32_t *hashes, uint32_t hashes_num) {
    ctx->hashes = hashes;
    ctx->hashes_num = hashes_num;

    ctx->hashes_reversed = malloc(hashes_num * MAX_STR_LENGTH);
    ret_iferr(!ctx->hashes_reversed, "failed to malloc hashes_reversed");

    cl_int errcode;
    ctx->mem_hashes = clCreateBuffer(ctx->cl_ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, hashes_num*16, hashes, &errcode);
    ret_iferr(errcode, "failed to create mem_hashes");
    ctx->mem_hashes_reversed = clCreateBuffer(ctx->cl_ctx, CL_MEM_WRITE_ONLY, hashes_num * MAX_STR_LENGTH, NULL, &errcode);
    ret_iferr(errcode, "failed to create mem_hashes_reversed");

    return CL_SUCCESS;
}

const uint32_t* gpu_cruncher_ctx_read_hashes_reversed(gpu_cruncher_ctx *ctx, cl_int *errcode) {
    *errcode = clEnqueueReadBuffer (ctx->queue, ctx->mem_hashes_reversed, CL_TRUE, 0, ctx->hashes_num * MAX_STR_LENGTH, ctx->hashes_reversed, 0, NULL, NULL);
    if (*errcode) {
        return NULL;
    }
    return ctx->hashes_reversed;
}

cl_int gpu_cruncher_ctx_free(gpu_cruncher_ctx *ctx) {
    if (ctx->hashes_reversed) {
        free(ctx->hashes_reversed);
    }

//    free(ctx->tasks_buffer);

    cl_int errcode = CL_SUCCESS;
    errcode |= clReleaseMemObject(ctx->mem_hashes);
    errcode |= clReleaseMemObject(ctx->mem_hashes_reversed);
    errcode |= clReleaseCommandQueue (ctx->queue);
    errcode |= clReleaseProgram(ctx->program);
    errcode |= clReleaseContext(ctx->cl_ctx);

    return errcode;
}

cl_int gpu_cruncher_ctx_submit_permut_task(gpu_cruncher_ctx *ctx, permut_task *task) {
    if (ctx->tasks_in_buffer_count >= PERMUT_TASKS_IN_BATCH) {
        cl_int errcode;
        errcode = gpu_cruncher_ctx_flush_tasks_buffer(ctx);
        ret_iferr(errcode, "failed to flush tasks buffer");
    }

//    memcpy(&ctx->tasks_buffer[ctx->tasks_in_buffer_count++], task, sizeof(permut_task));
    return CL_SUCCESS;
}

cl_int gpu_cruncher_ctx_flush_tasks_buffer(gpu_cruncher_ctx *ctx) {
/*
    FILE *file = fopen("buffer.log", "w");
    for (int i=0; i<gpu_cruncher_ctx->tasks_in_buffer_count; i++) {
        permut_task *task = &gpu_cruncher_ctx->tasks_buffer[i];
        
        fprintf(file, "task %d, start from %d\n", i, task->start_from);

        for (int j=0; j<MAX_OFFSETS_LENGTH && task->offsets[j]; j++) {
            fprintf(file, "%d ", task->offsets[j]);
        }
        fprintf(file, "\n");

        for (int j=0; j<MAX_OFFSETS_LENGTH && task->offsets[j]; j++) {
            char offset = task->offsets[j];
            if (offset < 0) {
                offset = -offset;
            } else {
                fprintf(file, "*");
            }
            offset--;
            fprintf(file, "%s ", &task->all_strs[offset]);
        }
        fprintf(file, "\n\n");
    }
    fclose(file);
*/

    cl_int errcode;
    errcode = gpu_cruncher_ctx_wait_for_cur_kernel(ctx);
    ret_iferr(errcode, "failed to wait for current kernel");

    anakrnl_permut *krnl = malloc(sizeof(anakrnl_permut));
    ret_iferr(!krnl, "failed to malloc kernel");
//    errcode = anakrnl_permut_create(krnl, ctx, MAX_ITERS_PER_TASK, ctx->tasks_buffer, ctx->tasks_in_buffer_count);
//    ret_iferr(errcode, "failed to create kernel");

    errcode = anakrnl_permut_enqueue(krnl);
    if (errcode) {
        fprintf(stderr, "%d: failed to enqueue kernel\n", errcode);
    }

    ctx->cur_exec_kernel = krnl;
    ctx->tasks_in_buffer_count = 0;

    return CL_SUCCESS;
}

cl_int gpu_cruncher_ctx_wait_for_cur_kernel(gpu_cruncher_ctx *ctx) {
    anakrnl_permut *krnl = ctx->cur_exec_kernel;
    if (krnl == NULL) {
        return CL_SUCCESS;
    }

    cl_int errcode;
    errcode = anakrnl_permut_wait(krnl);
    ret_iferr(errcode, "failed to wait for current kernel");

    const uint32_t *hashes_reversed = gpu_cruncher_ctx_read_hashes_reversed(ctx, &errcode);
    char hash_ascii[33];
    uint32_t hashes_found = 0;
    for(int i=0; i<ctx->hashes_num; i++) {
        char* hash_reversed = (char*)hashes_reversed + i*MAX_STR_LENGTH;
        if (strlen(hash_reversed)) {
            hashes_found++;
        }
    }

    if (hashes_found > ctx->hashes_seen) {
        ctx->hashes_seen = hashes_found;
        for(int i=0; i<ctx->hashes_num; i++) {
            char* hash_reversed = (char*)hashes_reversed + i*MAX_STR_LENGTH;
            if (strlen(hash_reversed)) {
                hash_to_ascii(ctx->hashes+i*4, hash_ascii);
                printf("%s:  %s\n", hash_ascii, hash_reversed);
            }
        }
        printf("\n");
    }

    errcode = anakrnl_permut_free(krnl);
    ret_iferr(errcode, "failed to free kernel");

    free(krnl);
    ctx->cur_exec_kernel = NULL;

    return CL_SUCCESS;
}

cl_int anakrnl_permut_create(anakrnl_permut *anakrnl, gpu_cruncher_ctx *anactx, uint32_t iters_per_item, permut_task *tasks, uint32_t num_tasks) {
    cl_int errcode;

    anakrnl->ctx = anactx;
    anakrnl->iters_per_task = iters_per_item;
    anakrnl->num_tasks = num_tasks;
    anakrnl->tasks = tasks;

    anakrnl->kernel = clCreateKernel(anactx->program, "permut", &errcode);
    ret_iferr(errcode, "failed to create permut kernel");

    anakrnl->mem_permut_tasks = clCreateBuffer(anactx->cl_ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, num_tasks*sizeof(permut_task), tasks, &errcode);
    ret_iferr(errcode, "failed to create mem_permut_tasks");

/*
        __kernel void permut(
            __global const permut_task *permut_templates,
                     const uint iters_per_item,
            __global uint *hashes,
                     uint hashes_num,
            __global uint *hashes_reversed)
*/
    errcode |= clSetKernelArg(anakrnl->kernel, 0, sizeof (cl_mem), &anakrnl->mem_permut_tasks);
    errcode |= clSetKernelArg(anakrnl->kernel, 1, sizeof (iters_per_item), &iters_per_item);
    errcode |= clSetKernelArg(anakrnl->kernel, 2, sizeof (cl_mem), &anactx->mem_hashes);
    errcode |= clSetKernelArg(anakrnl->kernel, 3, sizeof (anactx->hashes_num), &anactx->hashes_num);
    errcode |= clSetKernelArg(anakrnl->kernel, 4, sizeof (cl_mem), &anactx->mem_hashes_reversed);

    return errcode;
}

cl_int anakrnl_permut_enqueue(anakrnl_permut *anakrnl) {
    size_t globalWorkSize[] = {anakrnl->num_tasks, 0, 0};
    return clEnqueueNDRangeKernel(anakrnl->ctx->queue, anakrnl->kernel, 1, NULL, globalWorkSize, NULL, 0, NULL, &anakrnl->event);
}

cl_int anakrnl_permut_wait(anakrnl_permut *anakrnl) {
    return clWaitForEvents(1, &anakrnl->event);
}

cl_int anakrnl_permut_free(anakrnl_permut *anakrnl) {
    cl_int errcode = CL_SUCCESS;
    errcode |= clReleaseMemObject(anakrnl->mem_permut_tasks);
    errcode |= clReleaseKernel(anakrnl->kernel);
    return errcode;
}