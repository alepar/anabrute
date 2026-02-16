#include "common.h"
#include "gpu_cruncher.h"
#include "hashes.h"
#include "task_buffers.h"

static const char* size_suffixes[] = {"", "K", "M", "G", "T", "P"};
void format_bignum(uint64_t size, char *dst, uint16_t div) {
    int divs = 0;
    while (size/div > 1) {
        size = size/div;
        divs++;
    }
    sprintf(dst, "%lu%s", size, size_suffixes[divs]);
}

tasks_buffer *create_and_fill_task_buffer() {
/*
    df12bc796f2c4086ffa056b35b8cafde: x y t z a t b c d e
    3e1ebec048ca5f02e3d47ad33b0e2b08: a z t x y t e d c b
    ad2ae8b0c720abc7fefe5f7476d947c7: b c t d e t z a x y
    10eda7b7fb6bc8bd559a896167dc7486: d b t e c t y x a z
    9313c10eab276ddf2de6ff3d040b752e: e d t c b t a z y x
*/

    const char* all_strs = "t\0x\0y\0z\0a\0b\0c\0d\0e";
    const int8_t a[] = {3, 5, 7, 9, 11, 13, 15, 17, 0};
    const int8_t offsets[] = { 1, 2, -1, 3, 4, -1, 5, 6, 7, 8, 0 };

    tasks_buffer* buf = tasks_buffer_allocate();
    permut_task *task = buf->permut_tasks;
    buf->num_tasks = 1;

    memcpy(&task->all_strs, all_strs, 18);
    memcpy(&task->offsets, offsets, 11);
    memcpy(&task->a, a, 8);
    memset(&task->c, 0, 8);

    task->i = 0;
    task->n = 8;
    task->iters_done = 0;

    return buf;
}

tasks_buffer *create_and_fill_task_buffer2() {
/*
    81681bbf0446c002fc59c43f6ee4b390: x y z s a b f s g h
    2bdec9b6b72db90fd63a37252cedcf68: y x a s b z f s g h
    7bf742a10f73efc64c3de4f934a7fbe6: a b z s h g f s y x
    1f0038c914d8cd66cc94595ce5328792: f y z s a b x s g h
    36ee3b2c5ee4a8193e0311f1d3451844: h x z s a b f s g y
*/

    const char* all_strs = "s\0x\0y\0z\0a\0b\0f\0g\0h";
    const int8_t a[] = {3, 5, 7, 9, 11, 13, 15, 17, 0};
    const int8_t offsets[] = { 1, 2, 3, -1, 4, 5, 6, -1, 7, 8, 0 };

    tasks_buffer* buf = tasks_buffer_allocate();
    permut_task *task = buf->permut_tasks;
    buf->num_tasks = 1;

    memcpy(&task->all_strs, all_strs, 18);
    memcpy(&task->offsets, offsets, 11);
    memcpy(&task->a, a, 8);
    memset(&task->c, 0, 8);

    task->i = 0;
    task->n = 8;
    task->iters_done = 0;

    return buf;
}

int main(int argc, char *argv[]) {
    cl_platform_id platform_id;
    cl_uint num_platforms;
    clGetPlatformIDs (1, &platform_id, &num_platforms);
    ret_iferr(!num_platforms, "no platforms");

    cl_uint num_devices;
    cl_device_id device_id;
    clGetDeviceIDs (platform_id, CL_DEVICE_TYPE_ALL, 1, &device_id, &num_devices);
    ret_iferr(!num_devices, "no devices");

    char char_buf[1024];
    cl_ulong local_mem; char local_mem_str[32];
    cl_ulong global_mem; char global_mem_str[32];

    clGetDeviceInfo(device_id, CL_DEVICE_GLOBAL_MEM_SIZE, 8, &global_mem, NULL);
    clGetDeviceInfo(device_id, CL_DEVICE_LOCAL_MEM_SIZE, 8, &local_mem, NULL);
    clGetDeviceInfo (device_id, CL_DEVICE_NAME, 1024, char_buf, NULL);

    format_bignum(global_mem, global_mem_str, 1024);
    format_bignum(local_mem, local_mem_str, 1024);
    printf("OpenCL device: %s (g:%siB l:%siB)\n", char_buf, global_mem_str, local_mem_str);
    printf("\n");

    uint32_t *hashes;
    const uint32_t hashes_num = read_hashes("input.hashes.debug", &hashes);
    ret_iferr(!hashes_num, "failed to read hashes");
    ret_iferr(!hashes, "failed to allocate hashes");

    tasks_buffers tasks_bufs;
    tasks_buffers_create(&tasks_bufs);

    cl_int errcode;
    gpu_cruncher_ctx ctx;

    errcode = gpu_cruncher_ctx_create(&ctx, platform_id, device_id, &tasks_bufs, hashes, hashes_num);
    ret_iferr(errcode, "failed to create gpu_cruncher_ctx");

    pthread_t gpu_thread;
    int err = pthread_create(&gpu_thread, NULL, run_gpu_cruncher_thread, &ctx);
    ret_iferr(err, "failed to create gpu thread");

    for (int i=0; i<1; i++) {
        tasks_buffers_add_buffer(&tasks_bufs, create_and_fill_task_buffer());
        tasks_buffers_add_buffer(&tasks_bufs, create_and_fill_task_buffer2());
    }
    tasks_buffers_close(&tasks_bufs);

    err = pthread_join(gpu_thread, NULL);
    ret_iferr(err, "failed to join gpu thread");

    errcode = gpu_cruncher_ctx_read_hashes_reversed(&ctx);
    ret_iferr(errcode, "failed to read hashes_reversed");

    for(int i=0; i<hashes_num; i++) {
        char hash_ascii[33];
        hash_to_ascii(&hashes[i*4], hash_ascii);
        printf("%s:  %s\n", hash_ascii, (char*)(ctx.hashes_reversed + i*MAX_STR_LENGTH/4));
    }

    char strbuf[1024];
    format_bignum(ctx.consumed_anas, strbuf, 1000);
    printf("consumed %lu bufs, ~%sanas\n", ctx.consumed_bufs, strbuf);

    float busy_percentage;
    float anas_per_sec;
    gpu_cruncher_get_stats(&ctx, &busy_percentage, &anas_per_sec);
    format_bignum(anas_per_sec, strbuf, 1000);
    printf("%sanas/sec, load %.1f\n", strbuf, busy_percentage);

    gpu_cruncher_ctx_free(&ctx);

    return 0;
}