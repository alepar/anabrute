#ifndef GPU_CRUNCHER_H
#define GPU_CRUNCHER_H

#include "common.h"
#include "permut_types.h"
#include "task_buffers.h"

struct krnl_permut_s;
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

    // input queue
    tasks_buffers *tasks_buffs;

    // job output
    uint32_t *hashes_reversed;      // managed internally
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

typedef struct krnl_permut_s {
    gpu_cruncher_ctx *ctx;

    cl_kernel kernel;
    cl_event event;
    cl_mem mem_permut_tasks;

    tasks_buffer* buf;
    uint32_t iters_per_task;

    uint64_t time_start_micros;
    uint64_t time_end_micros;
} krnl_permut;

cl_int krnl_permut_create(krnl_permut *krnl, gpu_cruncher_ctx *ctx, uint32_t iters_per_krnl_task, tasks_buffer* buf);
cl_int krnl_permut_enqueue(krnl_permut *krnl);
cl_int krnl_permut_read_tasks(krnl_permut *krnl, tasks_buffer* buf);
cl_int krnl_permut_wait(krnl_permut *krnl);
cl_int krnl_permut_free(krnl_permut *krnl);
void krnl_permut_record_stats(krnl_permut *krnl);

#endif //GPU_CRUNCHER_H
