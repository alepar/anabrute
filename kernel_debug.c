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

const uint32_t read_hashes(char *file_name, uint32_t **hashes) {
    FILE *const fd = fopen(file_name, "r");
    if (!fd) {
        return 0;
    }

    fseek(fd, 0L, SEEK_END);
    const uint32_t file_size = (const uint32_t) ftell(fd);
    rewind(fd);

    const uint32_t hashes_num_est = (file_size + 1) / 33;
    uint32_t hashes_num = 0;

    *hashes = malloc(hashes_num_est*16);

    char buf[128];
    while(fgets(buf, sizeof(buf), fd) != NULL) {
        for (int i=0; i<sizeof(buf); i++) {
            if (buf[i] == '\n' || buf[i] == '\r') {
                buf[i] = 0;
            }
        }
        if (strlen(buf) != 32) {
            fprintf(stderr, "not a hash! (%s)\n", buf);
        }

        if (hashes_num>hashes_num_est) {
            fprintf(stderr, "too many hashes? skipping tail...\n");
            break;
        }

        ascii_to_hash(buf, &((*hashes)[hashes_num*4]));
        hashes_num++;
    }

    return hashes_num;
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
    for (int i=0; i<num_devices; i++) {
        cl_ulong local_mem; char local_mem_str[32];
        cl_ulong global_mem; char global_mem_str[32];

        clGetDeviceInfo(device_id, CL_DEVICE_GLOBAL_MEM_SIZE, 8, &global_mem, NULL);
        clGetDeviceInfo(device_id, CL_DEVICE_LOCAL_MEM_SIZE, 8, &local_mem, NULL);
        clGetDeviceInfo (device_id, CL_DEVICE_NAME, 1024, char_buf, NULL);

        format_bignum(global_mem, global_mem_str, 1024);
        format_bignum(local_mem, local_mem_str, 1024);
        printf("OpenCL device #%d: %s (g:%siB l:%siB)\n", i+1, char_buf, global_mem_str, local_mem_str);
    }
    printf("\n");

    uint32_t *hashes;
    const uint32_t hashes_num = read_hashes("input.hashes", &hashes);
    ret_iferr(!hashes_num, "failed to read hashes");
    ret_iferr(!hashes, "failed to allocate hashes");

    tasks_buffers tasks_bufs;

    cl_int errcode;
    gpu_cruncher_ctx ctx;

    errcode = gpu_cruncher_ctx_create(&ctx, platform_id, device_id, &tasks_bufs);
    ret_iferr(errcode, "failed to create gpu_cruncher_ctx");

    errcode = gpu_cruncher_ctx_set_input_hashes(&ctx, hashes, hashes_num);
    ret_iferr(errcode, "failed to set input hashes");

    pthread_t gpu_thread;
    int err = pthread_create(&gpu_thread, NULL, run_gpu_cruncher_thread, &ctx);
    ret_iferr(err, "failed to create gpu thread");

    for (int i=0; i<1; i++) {
        // TODO create and submit task buffer
//        printf("iter\n");
//        run_kernel(&ctx);
    }
    tasks_buffers_close(&tasks_bufs);

    err = pthread_join(gpu_thread, NULL);
    ret_iferr(err, "failed to join gpu thread");

    const uint32_t *hashes_reversed = gpu_cruncher_ctx_read_hashes_reversed(&ctx, &errcode);
    ret_iferr(errcode, "failed to read hashes_reversed");

    for(int i=0; i<hashes_num; i++) {
        char hash_ascii[33];
        hash_to_ascii(&hashes[i*4], hash_ascii);
        printf("%s:  %s\n", hash_ascii, (char*)&hashes_reversed[i*MAX_STR_LENGTH/4]);
    }

    for (int i=0; i<num_devices; i++) {
        gpu_cruncher_ctx_free(&ctx);
    }

    return 0;
}