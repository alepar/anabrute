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
    volatile uint32_t consumed_bufs;
    volatile uint64_t consumed_anas;
} gpu_cruncher_ctx;

cl_int gpu_cruncher_ctx_create(gpu_cruncher_ctx *ctx, cl_platform_id platform_id, cl_device_id device_id,
                               tasks_buffers* tasks_buffs, uint32_t *hashes, uint32_t hashes_num);
const uint32_t* gpu_cruncher_ctx_read_hashes_reversed(gpu_cruncher_ctx *ctx, cl_int *errcode);
cl_int gpu_cruncher_ctx_free(gpu_cruncher_ctx *ctx);
void* run_gpu_cruncher_thread(void *ptr);

typedef struct krnl_permut_s {
    gpu_cruncher_ctx *ctx;

    cl_kernel kernel;
    cl_event event;
    cl_mem mem_permut_tasks;

    uint32_t iters_per_task;
    uint32_t tasks_in_last_buf;
} krnl_permut;

cl_int krnl_permut_create(krnl_permut *krnl, gpu_cruncher_ctx *ctx, uint32_t iters_per_krnl_task, tasks_buffer* buf);
cl_int krnl_permut_enqueue(krnl_permut *krnl);
cl_int krnl_permut_read_tasks(krnl_permut *krnl, tasks_buffer* buf);
cl_int krnl_permut_wait(krnl_permut *krnl);
cl_int krnl_permut_free(krnl_permut *krnl);

#endif //GPU_CRUNCHER_H
