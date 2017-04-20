#include <string.h>
#include <stdio.h>

#include "gpu_cruncher.h"
#include "hashes.h"
#include "fact.h"
#include "os.h"

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
    buf[filesize] = 0;

    return buf;
}

// public stuff

cl_int gpu_cruncher_ctx_create(gpu_cruncher_ctx *ctx, cl_platform_id platform_id, cl_device_id device_id,
                               tasks_buffers* tasks_buffs, uint32_t *hashes, uint32_t hashes_num)
{
    ctx->platform_id = platform_id;
    ctx->device_id = device_id;

    ctx->tasks_buffs = tasks_buffs;

    ctx->is_running = true;
    ctx->consumed_bufs = 0;
    ctx->consumed_anas = 0L;

    ctx->hashes = hashes;
    ctx->hashes_num = hashes_num;
    ctx->last_refresh_hashes_reversed_millis = current_micros()/1000;

    memset(ctx->task_times_starts, 0, TIMES_WINDOW_LENGTH*sizeof(uint64_t));
    memset(ctx->task_times_ends, 0, TIMES_WINDOW_LENGTH*sizeof(uint64_t));
    memset(ctx->task_calculated_anas, 0, TIMES_WINDOW_LENGTH*sizeof(uint64_t));
    ctx->times_idx = 0;

    cl_int errcode;
    const cl_context_properties ctx_props [] = { CL_CONTEXT_PLATFORM, platform_id, 0, 0 };
    ctx->cl_ctx = clCreateContext(ctx_props, 1, &device_id, NULL, NULL, &errcode);
    ret_iferr(errcode, "failed to create context");

    char *const kernel_source = read_file("kernels/permut.cl");
    ret_iferr(!kernel_source, "failed to read kernel source");
    size_t lengths[] = {strlen(kernel_source)};
    const char *sources[] = {kernel_source};
    ctx->program = clCreateProgramWithSource(ctx->cl_ctx, 1, sources, lengths, &errcode);
    free(kernel_source);
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

    ctx->hashes_reversed = malloc(hashes_num * MAX_STR_LENGTH);
    ret_iferr(!ctx->hashes_reversed, "failed to malloc hashes_reversed");
    memset(ctx->hashes_reversed, 0, hashes_num * MAX_STR_LENGTH);

    ctx->mem_hashes = clCreateBuffer(ctx->cl_ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, hashes_num*16, hashes, &errcode);
    ret_iferr(errcode, "failed to create mem_hashes");
    ctx->mem_hashes_reversed = clCreateBuffer(ctx->cl_ctx, CL_MEM_WRITE_ONLY | CL_MEM_COPY_HOST_PTR, hashes_num * MAX_STR_LENGTH, ctx->hashes_reversed, &errcode);
    ret_iferr(errcode, "failed to create mem_hashes_reversed");

    return CL_SUCCESS;
}

cl_int gpu_cruncher_ctx_read_hashes_reversed(gpu_cruncher_ctx *ctx) {
    return clEnqueueReadBuffer (ctx->queue, ctx->mem_hashes_reversed, CL_TRUE, 0, ctx->hashes_num * MAX_STR_LENGTH, ctx->hashes_reversed, 0, NULL, NULL);
}

cl_int gpu_cruncher_ctx_refresh_hashes_reversed(gpu_cruncher_ctx *ctx) {
    uint64_t cur_millis = current_micros()/1000;
    if (cur_millis - ctx->last_refresh_hashes_reversed_millis > REFRESH_INTERVAL_HASHES_REVERSED_MILLIS) {
        ctx->last_refresh_hashes_reversed_millis = cur_millis;
        return gpu_cruncher_ctx_read_hashes_reversed(ctx);
    }

    return 0;
}

cl_int gpu_cruncher_ctx_free(gpu_cruncher_ctx *ctx) {
    if (ctx->hashes_reversed) {
        free(ctx->hashes_reversed);
    }

    cl_int errcode = CL_SUCCESS;
    errcode |= clReleaseMemObject(ctx->mem_hashes);
    errcode |= clReleaseMemObject(ctx->mem_hashes_reversed);
    errcode |= clReleaseCommandQueue (ctx->queue);
    errcode |= clReleaseProgram(ctx->program);
    errcode |= clReleaseContext(ctx->cl_ctx);

    return errcode;
}

void krnl_permut_record_stats(krnl_permut *krnl) {
    gpu_cruncher_ctx *ctx = krnl->ctx;

    ctx->consumed_anas += krnl->buf->num_anas;

    ctx->task_times_starts[ctx->times_idx] = krnl->time_start_micros;
    ctx->task_times_ends[ctx->times_idx] = krnl->time_end_micros;
    ctx->task_calculated_anas[ctx->times_idx] = krnl->buf->num_anas;
    ctx->times_idx = (ctx->times_idx+1) % TIMES_WINDOW_LENGTH;
}

void gpu_cruncher_get_stats(gpu_cruncher_ctx *ctx, float* busy_percentage, float* anas_per_sec) {
    uint64_t calculated_anas=0;
    uint64_t min_time_start = (uint64_t) -1L, max_time_ends=0;
    uint64_t micros_in_kernel=0;

    for (int i=0; i<TIMES_WINDOW_LENGTH; i++) {
        if (ctx->task_times_starts[i] > 0) {
            calculated_anas += ctx->task_calculated_anas[i];
            micros_in_kernel += ctx->task_times_ends[i] - ctx->task_times_starts[i];

            if (ctx->task_times_starts[i] < min_time_start) {
                min_time_start = ctx->task_times_starts[i];
            }
            if (ctx->task_times_ends[i] > max_time_ends) {
                max_time_ends = ctx->task_times_ends[i];
            }
        }
    }

    *busy_percentage = (float) micros_in_kernel / (max_time_ends-min_time_start) * 100.0f;
    *anas_per_sec = (float) (calculated_anas) / ((max_time_ends-min_time_start)/1000000.0f); // this is imprecise
}

void* run_gpu_cruncher_thread(void *ptr) {
    gpu_cruncher_ctx *ctx = ptr;

    int errcode;

    tasks_buffer* src_buf;
    errcode = tasks_buffers_get_buffer(ctx->tasks_buffs, &src_buf);
    ret_iferr(errcode, "failed to get first buffer"); // TODO try-catch
    uint32_t src_idx = 0;

    tasks_buffer* dst_buf = tasks_buffer_allocate();
    memset(dst_buf->permut_tasks, 0, PERMUT_TASKS_IN_KERNEL_TASK*sizeof(permut_task));

    main: while(1) {
        if (src_buf != NULL && src_idx >= src_buf->num_tasks && tasks_buffers_num_ready(ctx->tasks_buffs)) {
            // fetch new buffer from queue
            ctx->consumed_bufs++;
            tasks_buffer_free(src_buf);
            errcode = tasks_buffers_get_buffer(ctx->tasks_buffs, &src_buf);;
            ret_iferr(errcode, "failed to get first buffer");
            src_idx = 0;
        } else {
            // prepare dst_buf
            // TODO compaction?
            for (uint32_t i=dst_buf->num_tasks; i<PERMUT_TASKS_IN_KERNEL_TASK; i++) {
                permut_task *dst_task = dst_buf->permut_tasks + i;
                if (dst_task->i >= dst_task->n) {
                    // task is finished
                    if (dst_task->n) {
                        dst_task->n = 0;
                    }

                    if (src_buf != NULL) {  // do we still have src buffers?
                        if (src_idx >= src_buf->num_tasks) {  // need new src_buf
                            if (dst_buf->num_tasks == 0 || tasks_buffers_num_ready(ctx->tasks_buffs)>0) {
                                goto main;
                            } else {
                                break; // seems like we're temporarily out of buffers, run what we have for now
                            }
                        }

                        memcpy(dst_task, src_buf->permut_tasks + src_idx++, sizeof(permut_task));
                    } else {
                        break; // permanently out of buffers
                    }
                }
                if (dst_task->i < dst_task->n) {
                    dst_buf->num_tasks = i+1;
                    uint64_t iters_left = fact(dst_task->n) - dst_task->iters_done;
                    dst_buf->num_anas += iters_left > MAX_ITERS_IN_KERNEL_TASK ? MAX_ITERS_IN_KERNEL_TASK : iters_left;
                }
            }

            if (dst_buf->num_tasks == 0) {
                // we ran out of buffers, exit
                break;
            }

            // schedule kernel with dst_buf
            krnl_permut krnl;
            errcode = krnl_permut_create(&krnl, ctx, MAX_ITERS_IN_KERNEL_TASK, dst_buf);
            ret_iferr(errcode, "failed to create kernel");

            errcode = krnl_permut_enqueue(&krnl);
            ret_iferr(errcode, "failed to enqueue kernel");

            errcode = krnl_permut_wait(&krnl);
            ret_iferr(errcode, "failed to wait for kernel");

            errcode = krnl_permut_read_tasks(&krnl, dst_buf);
            ret_iferr(errcode, "failed to read tasks out from gpu");
            dst_buf->num_tasks = 0;
            dst_buf->num_anas = 0;

            errcode = gpu_cruncher_ctx_refresh_hashes_reversed(ctx);
            ret_iferr(errcode, "failed to refresh hashes_reversed");

            // TODO drop code below
/*
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
*/

            errcode = krnl_permut_free(&krnl);
            ret_iferr(errcode, "failed to free kernel");
        }
    }

    ctx->is_running = false;
    return NULL;
}


cl_int krnl_permut_create(krnl_permut *krnl, gpu_cruncher_ctx *ctx, uint32_t iters_per_krnl_task, tasks_buffer* buf) {
    cl_int errcode;

    krnl->ctx = ctx;
    krnl->iters_per_task = iters_per_krnl_task;

    krnl->kernel = clCreateKernel(ctx->program, "permut", &errcode);
    ret_iferr(errcode, "failed to create permut kernel");

    krnl->mem_permut_tasks = clCreateBuffer(ctx->cl_ctx, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, buf->num_tasks*sizeof(permut_task), buf->permut_tasks, &errcode);
    ret_iferr(errcode, "failed to create mem_permut_tasks");

    krnl->buf = buf;

/*
    __kernel void permut(
            __global const permut_task *tasks,
                     const uint iters_per_task,
            __global const uint *hashes,
                     const uint hashes_num,
            __global const uint *hashes_reversed)
*/
    errcode |= clSetKernelArg(krnl->kernel, 0, sizeof (cl_mem), &krnl->mem_permut_tasks);
    errcode |= clSetKernelArg(krnl->kernel, 1, sizeof (iters_per_krnl_task), &iters_per_krnl_task);
    errcode |= clSetKernelArg(krnl->kernel, 2, sizeof (cl_mem), &ctx->mem_hashes);
    errcode |= clSetKernelArg(krnl->kernel, 3, sizeof (ctx->hashes_num), &ctx->hashes_num);
    errcode |= clSetKernelArg(krnl->kernel, 4, sizeof (cl_mem), &ctx->mem_hashes_reversed);

    return errcode;
}

cl_int krnl_permut_enqueue(krnl_permut *krnl) {
    size_t globalWorkSize[] = {krnl->buf->num_tasks, 0, 0};
    krnl->time_start_micros = current_micros();
    return clEnqueueNDRangeKernel(krnl->ctx->queue, krnl->kernel, 1, NULL, globalWorkSize, NULL, 0, NULL, &krnl->event);
}

cl_int krnl_permut_wait(krnl_permut *krnl) {
    cl_int errcode = clWaitForEvents(1, &krnl->event);
    krnl->time_end_micros = current_micros();

    krnl_permut_record_stats(krnl);

    return errcode;
}

cl_int krnl_permut_free(krnl_permut *krnl) {
    cl_int errcode = CL_SUCCESS;
    errcode |= clReleaseMemObject(krnl->mem_permut_tasks);
    errcode |= clReleaseKernel(krnl->kernel);
    return errcode;
}

cl_int krnl_permut_read_tasks(krnl_permut *krnl, tasks_buffer* buf) {
    int errcode;

    errcode = clEnqueueReadBuffer (krnl->ctx->queue, krnl->mem_permut_tasks, CL_TRUE, 0, krnl->buf->num_tasks * sizeof(permut_task), buf->permut_tasks, 0, NULL, NULL);
    ret_iferr(errcode, "failed to read permut_tasks from gpu");

    buf->num_tasks = krnl->buf->num_tasks;

    return errcode;
}
