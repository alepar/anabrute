#include "common.h"
#ifdef HAVE_OPENCL

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

    ctx->cfg = NULL;

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

    ctx->local_hashes_reversed = malloc(hashes_num * MAX_STR_LENGTH);
    ret_iferr(!ctx->local_hashes_reversed, "failed to malloc local_hashes_reversed");
    memset(ctx->local_hashes_reversed, 0, hashes_num * MAX_STR_LENGTH);
    ctx->hashes_reversed = ctx->local_hashes_reversed;  // default for kernel_debug

    ctx->mem_hashes = clCreateBuffer(ctx->cl_ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, hashes_num*16, hashes, &errcode);
    ret_iferr(errcode, "failed to create mem_hashes");
    ctx->mem_hashes_reversed = clCreateBuffer(ctx->cl_ctx, CL_MEM_WRITE_ONLY | CL_MEM_COPY_HOST_PTR, hashes_num * MAX_STR_LENGTH, ctx->local_hashes_reversed, &errcode);
    ret_iferr(errcode, "failed to create mem_hashes_reversed");

    // Create persistent kernel
    ctx->kernel = clCreateKernel(ctx->program, "permut", &errcode);
    ret_iferr(errcode, "failed to create permut kernel");

    // Allocate double-buffered task memory
    size_t tasks_buf_size = PERMUT_TASKS_IN_KERNEL_TASK * sizeof(permut_task);
    for (int i = 0; i < 2; i++) {
        ctx->mem_tasks[i] = clCreateBuffer(ctx->cl_ctx, CL_MEM_READ_WRITE, tasks_buf_size, NULL, &errcode);
        ret_iferr(errcode, "failed to create mem_tasks buffer");
        ctx->host_tasks[i] = tasks_buffer_allocate();
        ret_iferr(!ctx->host_tasks[i], "failed to allocate host_tasks buffer");
        memset(ctx->host_tasks[i]->permut_tasks, 0, tasks_buf_size);
    }

    return CL_SUCCESS;
}

cl_int gpu_cruncher_ctx_read_hashes_reversed(gpu_cruncher_ctx *ctx) {
    cl_int err = clEnqueueReadBuffer(ctx->queue, ctx->mem_hashes_reversed, CL_TRUE, 0,
        ctx->hashes_num * MAX_STR_LENGTH, ctx->local_hashes_reversed, 0, NULL, NULL);
    if (err != CL_SUCCESS) return err;

    // Merge to shared buffer. No lock needed: each slot is written only with
    // the correct match data, so concurrent writes from multiple backends are
    // idempotent. The reader (main thread) may see a partial write briefly,
    // which is harmless for display purposes.
    if (ctx->cfg) {
        for (uint32_t i = 0; i < ctx->hashes_num; i++) {
            if (ctx->local_hashes_reversed[i * MAX_STR_LENGTH / 4]) {
                memcpy(ctx->cfg->hashes_reversed + i * MAX_STR_LENGTH / 4,
                       ctx->local_hashes_reversed + i * MAX_STR_LENGTH / 4,
                       MAX_STR_LENGTH);
            }
        }
    }
    return CL_SUCCESS;
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
    if (ctx->local_hashes_reversed) {
        free(ctx->local_hashes_reversed);
        ctx->local_hashes_reversed = NULL;
        ctx->hashes_reversed = NULL;
    }

    cl_int errcode = CL_SUCCESS;
    errcode |= clReleaseKernel(ctx->kernel);
    for (int i = 0; i < 2; i++) {
        errcode |= clReleaseMemObject(ctx->mem_tasks[i]);
        if (ctx->host_tasks[i]) {
            free(ctx->host_tasks[i]);
            ctx->host_tasks[i] = NULL;
        }
    }
    errcode |= clReleaseMemObject(ctx->mem_hashes);
    errcode |= clReleaseMemObject(ctx->mem_hashes_reversed);
    errcode |= clReleaseCommandQueue(ctx->queue);
    errcode |= clReleaseProgram(ctx->program);
    errcode |= clReleaseContext(ctx->cl_ctx);

    return errcode;
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

// Helper: prepare a task buffer with carry-over and new tasks from input queue
// Returns number of tasks prepared, 0 if no more work available
static uint32_t prepare_task_buffer(gpu_cruncher_ctx *ctx, tasks_buffer *buf,
                                     tasks_buffer **src_buf, uint32_t *src_idx) {
    int errcode;
    buf->num_tasks = 0;
    buf->num_anas = 0;

    for (uint32_t i = 0; i < PERMUT_TASKS_IN_KERNEL_TASK; i++) {
        permut_task *task = buf->permut_tasks + i;

        if (task->i >= task->n) {
            // Task finished or empty slot - try to fill with new task
            task->n = 0;

            if (*src_buf == NULL) break;  // permanently out of input

            // Need new source buffer?
            while (*src_idx >= (*src_buf)->num_tasks) {
                ctx->consumed_bufs++;
                tasks_buffers_recycle(ctx->tasks_buffs, *src_buf);
                errcode = tasks_buffers_get_buffer(ctx->tasks_buffs, src_buf);
                if (errcode || *src_buf == NULL) {
                    *src_buf = NULL;
                    break;
                }
                *src_idx = 0;
            }

            if (*src_buf == NULL) break;

            memcpy(task, (*src_buf)->permut_tasks + (*src_idx)++, sizeof(permut_task));
        }

        if (task->i < task->n) {
            buf->num_tasks = i + 1;
            uint64_t iters_left = fact(task->n) - task->iters_done;
            buf->num_anas += iters_left > MAX_ITERS_IN_KERNEL_TASK ? MAX_ITERS_IN_KERNEL_TASK : iters_left;
        }
    }

    return buf->num_tasks;
}

void* run_gpu_cruncher_thread(void *ptr) {
    gpu_cruncher_ctx *ctx = ptr;
    cl_int errcode;

    // Input source
    tasks_buffer *src_buf;
    errcode = tasks_buffers_get_buffer(ctx->tasks_buffs, &src_buf);
    ret_iferr(errcode, "failed to get first buffer");
    uint32_t src_idx = 0;

    // Double-buffer indices
    int gpu_buf = 0;   // buffer with kernel running (or just finished)
    int cpu_buf = 1;   // buffer being prepared/uploaded

    cl_event kernel_event = NULL;
    uint64_t kernel_start_time = 0;
    uint64_t kernel_num_anas = 0;
    bool kernel_running = false;
    bool first_iteration = true;

    // Bootstrap: prepare cpu_buf
    uint32_t cpu_tasks = prepare_task_buffer(ctx, ctx->host_tasks[cpu_buf], &src_buf, &src_idx);
    if (cpu_tasks == 0) goto done;

    while (1) {
        // Phase 1: Upload cpu_buf (while previous kernel may still run)
        errcode = clEnqueueWriteBuffer(ctx->queue, ctx->mem_tasks[cpu_buf], CL_FALSE, 0,
                                       cpu_tasks * sizeof(permut_task),
                                       ctx->host_tasks[cpu_buf]->permut_tasks, 0, NULL, NULL);
        ret_iferr(errcode, "failed to upload tasks");

        // Phase 2: Wait for previous kernel on gpu_buf (if any)
        if (kernel_running) {
            errcode = clFinish(ctx->queue);
            ret_iferr(errcode, "failed to wait for kernel");

            uint64_t end_time = current_micros();
            ctx->consumed_anas += kernel_num_anas;
            ctx->task_times_starts[ctx->times_idx] = kernel_start_time;
            ctx->task_times_ends[ctx->times_idx] = end_time;
            ctx->task_calculated_anas[ctx->times_idx] = kernel_num_anas;
            ctx->times_idx = (ctx->times_idx + 1) % TIMES_WINDOW_LENGTH;
            clReleaseEvent(kernel_event);
            kernel_running = false;

            errcode = gpu_cruncher_ctx_refresh_hashes_reversed(ctx);
            ret_iferr(errcode, "failed to refresh hashes_reversed");

            // Phase 3: Read gpu_buf (just finished) - SYNCHRONOUS to ensure data ready for prepare
            uint32_t gpu_tasks = ctx->host_tasks[gpu_buf]->num_tasks;
            if (gpu_tasks > 0) {
                errcode = clEnqueueReadBuffer(ctx->queue, ctx->mem_tasks[gpu_buf], CL_TRUE, 0,
                                              gpu_tasks * sizeof(permut_task),
                                              ctx->host_tasks[gpu_buf]->permut_tasks, 0, NULL, NULL);
                ret_iferr(errcode, "failed to read tasks");
            }
        }

        // Phase 4: Launch kernel on cpu_buf
        uint32_t iters = MAX_ITERS_IN_KERNEL_TASK;
        errcode = clSetKernelArg(ctx->kernel, 0, sizeof(cl_mem), &ctx->mem_tasks[cpu_buf]);
        errcode |= clSetKernelArg(ctx->kernel, 1, sizeof(iters), &iters);
        errcode |= clSetKernelArg(ctx->kernel, 2, sizeof(cl_mem), &ctx->mem_hashes);
        errcode |= clSetKernelArg(ctx->kernel, 3, sizeof(ctx->hashes_num), &ctx->hashes_num);
        errcode |= clSetKernelArg(ctx->kernel, 4, sizeof(cl_mem), &ctx->mem_hashes_reversed);
        ret_iferr(errcode, "failed to set kernel args");

        size_t global_size = cpu_tasks;
        kernel_start_time = current_micros();
        errcode = clEnqueueNDRangeKernel(ctx->queue, ctx->kernel, 1, NULL, &global_size, NULL, 0, NULL, &kernel_event);
        ret_iferr(errcode, "failed to enqueue kernel");
        kernel_num_anas = ctx->host_tasks[cpu_buf]->num_anas;
        kernel_running = true;

        // Phase 5: Swap buffers
        int tmp = gpu_buf;
        gpu_buf = cpu_buf;
        cpu_buf = tmp;

        // Phase 6: Prepare cpu_buf (old gpu_buf) for next iteration
        // On first iteration, cpu_buf (old gpu_buf) hasn't run yet, so skip
        // After first iteration, cpu_buf has fresh data from the read in Phase 3
        if (!first_iteration) {
            cpu_tasks = prepare_task_buffer(ctx, ctx->host_tasks[cpu_buf], &src_buf, &src_idx);
            if (cpu_tasks == 0) {
                // No more work - wait for current kernel and exit
                clFinish(ctx->queue);
                break;
            }
        } else {
            // First iteration: prepare cpu_buf from scratch
            cpu_tasks = prepare_task_buffer(ctx, ctx->host_tasks[cpu_buf], &src_buf, &src_idx);
            first_iteration = false;
            if (cpu_tasks == 0) {
                clFinish(ctx->queue);
                break;
            }
        }
    }

done:
    gpu_cruncher_ctx_read_hashes_reversed(ctx);
    ctx->is_running = false;
    return NULL;
}

#endif /* HAVE_OPENCL */
