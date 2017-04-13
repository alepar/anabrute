#ifndef OPENCL_TEST_OCL_TYPES_H
#define OPENCL_TEST_OCL_TYPES_H

#include <stdint.h>
#ifdef __APPLE__
    #include "OpenCL/opencl.h"
    #include <unitypes.h>
#else
    #include "CL/cl.h"
#endif

#include "anatypes.h"


// peak at ~256-512K
#define PERMUT_TASKS_IN_BATCH 256*1024
// peak at ~512
#define MAX_ITERS_PER_TASK 512
// 512*512*1024 == 256M (11! < 256M < 12!)

#define MAX_STR_LENGTH 40
#define MAX_OFFSETS_LENGTH 20

typedef struct permut_task_s {
    char all_strs[MAX_STR_LENGTH];
    char offsets[MAX_OFFSETS_LENGTH];  // positives - permutable, negatives - fixed, zeroes - empty; abs(offset)-1 to get offset in all_strs
    uint start_from;
} permut_task;

struct anakrnl_permut_s;
typedef struct anactx_s {
    // parallelization over devices
    uint32_t num_threads;
    uint32_t thread_id;

    // job definition
    char_counts *seed_phrase;
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

    // TODO stats
} anactx;

cl_int anactx_create(anactx *anactx, cl_platform_id platform_id, cl_device_id device_id);
cl_int anactx_set_input_hashes(anactx *anactx, uint32_t *hashes, uint32_t hashes_num);
const uint32_t* anactx_read_hashes_reversed(anactx *anactx, cl_int *errcode);
cl_int anactx_free(anactx *anactx);
cl_int anactx_submit_permut_task(anactx *anactx, permut_task *task);
cl_int anactx_flush_tasks_buffer(anactx *anactx);
cl_int anactx_wait_for_cur_kernel(anactx *anactx);

// TODO add to queue
// TODO make queue available
// TODO process events routine

typedef struct anakrnl_permut_s {
    anactx *ctx;

    cl_kernel kernel;
    cl_event event;
    cl_mem mem_permut_tasks;

    uint32_t iters_per_task;
    permut_task *tasks;
    uint32_t num_tasks;
} anakrnl_permut;

cl_int anakrnl_permut_create(anakrnl_permut *anakrnl, anactx *anactx, uint32_t iters_per_item, permut_task *tasks, uint32_t num_tasks);
cl_int anakrnl_permut_enqueue(anakrnl_permut *anakrnl);
cl_int anakrnl_permut_wait(anakrnl_permut *anakrnl);
cl_int anakrnl_permut_free(anakrnl_permut *anakrnl);

#define ret_iferr(val, msg) \
if (val) {\
    fprintf(stderr, "FATAL: %d - %s\n", val, msg);\
    return val;\
}

#endif //OPENCL_TEST_OCL_TYPES_H
