#ifndef OPENCL_TEST_OCL_TYPES_H
#define OPENCL_TEST_OCL_TYPES_H

#include <stdint.h>

#ifdef __APPLE__
    #include "OpenCL/opencl.h"
    #include <unitypes.h>
#else
    #include "CL/cl.h"
#endif

typedef struct {
    // control stuff
    cl_platform_id platform_id;
    cl_device_id device_id;
    cl_context cl_ctx;
    cl_program program;
    cl_command_queue queue;

    // shared input/output
    uint32_t hashes_num;
    uint32_t *hashes_reversed;  // managed internally, rest is external
    cl_mem mem_hashes;
    cl_mem mem_hashes_reversed;

    // TODO queues for kernel execution
    // TODO stats
} anactx;

cl_int anactx_create(anactx *anactx, cl_platform_id platform_id, cl_device_id device_id);
cl_int anactx_set_input_hashes(anactx *anactx, uint32_t *hashes, uint32_t hashes_num);
const uint32_t* anactx_read_hashes_reversed(anactx *anactx, cl_int *errcode);
cl_int anactx_free(anactx *anactx);

// TODO add to queue
// TODO make queue available
// TODO process events routine

#define MAX_STR_LENGTH 40
#define MAX_OFFSETS_LENGTH 20

typedef struct {
    char all_strs[MAX_STR_LENGTH];
    char offsets[MAX_OFFSETS_LENGTH];  // positives - permutable, negatives - fixed, zeroes - empty; abs(offset)-1 to get offset in all_strs
    uint start_from;
} permut_template;

typedef struct {
    anactx *ctx;

    cl_kernel kernel;
    cl_event event;
    cl_mem mem_permut_templates;

    uint32_t iters_per_item;
    permut_template *templates;
    uint32_t num_templates;
} anakrnl_permut;

cl_int anakrnl_permut_create(anakrnl_permut *anakrnl, anactx *anactx, uint32_t iters_per_item, permut_template *templates, uint32_t num_templates);
cl_int anakrnl_permut_enqueue(anakrnl_permut *anakrnl);
cl_int anakrnl_permut_wait(anakrnl_permut *anakrnl);
cl_int anakrnl_permut_free(anakrnl_permut *anakrnl);

#endif //OPENCL_TEST_OCL_TYPES_H
