#ifndef GPU_CRUNCHER_H
#define GPU_CRUNCHER_H

#include "common.h"
#include "permut_types.h"
#include "task_buffers.h"

struct anakrnl_permut_s;
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

    // task management
    struct anakrnl_permut_s *cur_exec_kernel;
    tasks_buffers *tasks_buffs;
    tasks_buffer *local_buffer;
    uint32_t tasks_in_buffer_count;

    // job output
    uint32_t *hashes_reversed;      // managed internally
    cl_mem mem_hashes_reversed;
    uint32_t hashes_seen;

    // progress
    volatile bool is_running;
    volatile uint32_t consumed;
} gpu_cruncher_ctx;

cl_int gpu_cruncher_ctx_create(gpu_cruncher_ctx *ctx, cl_platform_id platform_id, cl_device_id device_id, tasks_buffers* tasks_buffs);
cl_int gpu_cruncher_ctx_set_input_hashes(gpu_cruncher_ctx *ctx, uint32_t *hashes, uint32_t hashes_num);
const uint32_t* gpu_cruncher_ctx_read_hashes_reversed(gpu_cruncher_ctx *ctx, cl_int *errcode);
cl_int gpu_cruncher_ctx_free(gpu_cruncher_ctx *ctx);
cl_int gpu_cruncher_ctx_submit_permut_task(gpu_cruncher_ctx *ctx, permut_task *task);
cl_int gpu_cruncher_ctx_flush_tasks_buffer(gpu_cruncher_ctx *ctx);
cl_int gpu_cruncher_ctx_wait_for_cur_kernel(gpu_cruncher_ctx *ctx);

typedef struct anakrnl_permut_s {
    gpu_cruncher_ctx *ctx;

    cl_kernel kernel;
    cl_event event;
    cl_mem mem_permut_tasks;

    uint32_t iters_per_task;
    permut_task *tasks;
    uint32_t num_tasks;
} anakrnl_permut;

cl_int anakrnl_permut_create(anakrnl_permut *anakrnl, gpu_cruncher_ctx *anactx, uint32_t iters_per_item, permut_task *tasks, uint32_t num_tasks);
cl_int anakrnl_permut_enqueue(anakrnl_permut *anakrnl);
cl_int anakrnl_permut_wait(anakrnl_permut *anakrnl);
cl_int anakrnl_permut_free(anakrnl_permut *anakrnl);

#endif //GPU_CRUNCHER_H
