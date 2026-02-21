#ifndef GPU_CRUNCHER_H
#define GPU_CRUNCHER_H

#include "common.h"
#include "cruncher.h"
#include "permut_types.h"
#include "task_buffers.h"

typedef struct gpu_cruncher_ctx_s {
    // job definition
    uint32_t *hashes;
    uint32_t hashes_num;
    cl_mem mem_hashes;

    // control stuff
    cl_platform_id platform_id;
    cl_device_id device_id;
    cl_context cl_ctx;
    cl_program program;
    cl_command_queue queue;

    // persistent kernel and double-buffered task memory
    cl_kernel kernel;
    cl_mem mem_tasks[2];           // ping-pong GPU buffers
    tasks_buffer *host_tasks[2];   // ping-pong host buffers

    // input queue
    tasks_buffers *tasks_buffs;

    // cruncher abstraction (NULL when used directly by kernel_debug)
    cruncher_config *cfg;

    // job output
    uint32_t *hashes_reversed;      // points to local_hashes_reversed (kernel_debug) or cfg->hashes_reversed (cruncher)
    uint32_t *local_hashes_reversed; // temp buffer for GPU readback
    cl_mem mem_hashes_reversed;

    // progress
    volatile bool is_running;
    volatile uint64_t consumed_bufs;
    volatile uint64_t consumed_anas;

    // misc internal state
    uint64_t last_refresh_hashes_reversed_millis;
    volatile uint64_t task_times_starts[TIMES_WINDOW_LENGTH];
    volatile uint64_t task_times_ends[TIMES_WINDOW_LENGTH];
    volatile uint64_t task_calculated_anas[TIMES_WINDOW_LENGTH];
    uint32_t times_idx;
} gpu_cruncher_ctx;

cl_int gpu_cruncher_ctx_create(gpu_cruncher_ctx *ctx, cl_platform_id platform_id, cl_device_id device_id,
                               tasks_buffers* tasks_buffs, uint32_t *hashes, uint32_t hashes_num);
cl_int gpu_cruncher_ctx_read_hashes_reversed(gpu_cruncher_ctx *ctx);
cl_int gpu_cruncher_ctx_refresh_hashes_reversed(gpu_cruncher_ctx *ctx);
void* run_gpu_cruncher_thread(void *ptr);
cl_int gpu_cruncher_ctx_free(gpu_cruncher_ctx *ctx);
void gpu_cruncher_get_stats(gpu_cruncher_ctx *ctx, float* busy_percentage, float* anas_per_sec);

#endif //GPU_CRUNCHER_H
