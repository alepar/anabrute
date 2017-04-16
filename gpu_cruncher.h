#ifndef OPENCL_TEST_OCL_TYPES_H
#define OPENCL_TEST_OCL_TYPES_H

#include <stdint.h>
#ifdef __APPLE__
    #include "OpenCL/opencl.h"
    #include <unitypes.h>
#else
    #include "CL/cl.h"
#endif

#include "constants.h"
#include "permut_types.h"

typedef struct permut_task_s {
    char all_strs[MAX_STR_LENGTH];
    char offsets[MAX_OFFSETS_LENGTH];  // positives - permutable, negatives - fixed, zeroes - empty; abs(offset)-1 to get offset in all_strs
    uint start_from;
} permut_task;

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
    permut_task *tasks_buffer;      // managed internally
    uint32_t tasks_in_buffer_count;

    // job output
    uint32_t *hashes_reversed;      // managed internally
    cl_mem mem_hashes_reversed;
    uint32_t hashes_seen;

    // progress
    // TODO
} gpu_cruncher_ctx;

cl_int gpu_cruncher_ctx_create(gpu_cruncher_ctx *anactx, cl_platform_id platform_id, cl_device_id device_id);
cl_int gpu_cruncher_ctx_set_input_hashes(gpu_cruncher_ctx *anactx, uint32_t *hashes, uint32_t hashes_num);
const uint32_t* gpu_cruncher_ctx_read_hashes_reversed(gpu_cruncher_ctx *anactx, cl_int *errcode);
cl_int gpu_cruncher_ctx_free(gpu_cruncher_ctx *anactx);
cl_int gpu_cruncher_ctx_submit_permut_task(gpu_cruncher_ctx *anactx, permut_task *task);
cl_int gpu_cruncher_ctx_flush_tasks_buffer(gpu_cruncher_ctx *anactx);
cl_int gpu_cruncher_ctx_wait_for_cur_kernel(gpu_cruncher_ctx *anactx);

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

#define ret_iferr(val, msg) \
if (val) {\
    fprintf(stderr, "FATAL: %d - %s\n", val, msg);\
    return val;\
}

#endif //OPENCL_TEST_OCL_TYPES_H
