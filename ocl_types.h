#ifndef OPENCL_TEST_OCL_TYPES_H
#define OPENCL_TEST_OCL_TYPES_H

#include <stdint.h>

#ifdef __APPLE__
    #include "OpenCL/opencl.h"
    #include <unitypes.h>
#else
    #include "CL/cl.h"
#endif

// peak at ~256-512K
#define PERMUT_TEMPLATES_SIZE 512*1024
// peak at ~512
#define MAX_ITERS_PER_ITEM 512

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

    // control stuff
    cl_platform_id platform_id;
    cl_device_id device_id;
    cl_context cl_ctx;
    cl_program program;
    cl_command_queue queue;

    // shared input/output
    uint32_t hashes_num;
    uint32_t *hashes_reversed;      // managed internally
    cl_mem mem_hashes;
    cl_mem mem_hashes_reversed;

    // task management
    struct anakrnl_permut_s *cur_exec_kernel;
    permut_task *tasks_buffer;      // managed internally
    uint32_t tasks_in_buffer_count;

    // TODO stats
} anactx;

cl_int anactx_create(anactx *anactx, cl_platform_id platform_id, cl_device_id device_id);
cl_int anactx_set_input_hashes(anactx *anactx, uint32_t *hashes, uint32_t hashes_num);
const uint32_t* anactx_read_hashes_reversed(anactx *anactx, cl_int *errcode);
cl_int anactx_free(anactx *anactx);

// TODO add to queue
// TODO make queue available
// TODO process events routine

typedef struct anakrnl_permut_s {
    anactx *ctx;

    cl_kernel kernel;
    cl_event event;
    cl_mem mem_permut_templates;

    uint32_t iters_per_item;
    permut_task *templates;
    uint32_t num_templates;
} anakrnl_permut;

cl_int anakrnl_permut_create(anakrnl_permut *anakrnl, anactx *anactx, uint32_t iters_per_item, permut_task *templates, uint32_t num_templates);
cl_int anakrnl_permut_enqueue(anakrnl_permut *anakrnl);
cl_int anakrnl_permut_wait(anakrnl_permut *anakrnl);
cl_int anakrnl_permut_free(anakrnl_permut *anakrnl);

#endif //OPENCL_TEST_OCL_TYPES_H
