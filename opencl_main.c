#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <time.h>
#include <pthread.h>

#ifdef __APPLE__
    #include <unitypes.h>
    #include <event.h>
#else
    #include <sys/time.h>
#endif

#include "ocl_layer.h"
#include "hashes.h"

static const char* size_suffixes[] = {"", "K", "M", "G", "T", "P"};
void format_bignum(uint64_t size, char *dst, uint16_t div) {
    int divs = 0;
    while (size/div > 1) {
        size = size/div;
        divs++;
    }
    sprintf(dst, "%lu%s", size, size_suffixes[divs]);
}

void* run_kernel(void *ptr) {
    anactx *anactx = ptr;

    FILE *file = fopen("submit.dump.crash1", "r");
/*
    fseek(file, 0L, SEEK_END);
    uint32_t buffer_len = (const uint32_t) ftell(file);
    rewind(file);
*/
    int fseekres = fseek(file, -152 * sizeof(permut_task), SEEK_END);
    if (fseekres) {
        printf("fseek caput\n");
        return NULL;
    }

    int num_tasks = 1;
    permut_task *tasks = calloc(num_tasks, sizeof(permut_task));
    uint32_t read = fread(tasks, sizeof(permut_task), num_tasks, file);
    fclose(file);

    if (read != num_tasks) {
        printf("oops\n");
        return NULL;
    }

    {
        printf("task %d, start from %d\n", 0, tasks->start_from);

        for (int j = 0; j < MAX_OFFSETS_LENGTH && tasks->offsets[j]; j++) {
            printf("%d ", tasks->offsets[j]);
        }
        printf("\n");

        for (int j = 0; j < MAX_OFFSETS_LENGTH && tasks->offsets[j]; j++) {
            char offset = tasks->offsets[j];
            if (offset < 0) {
                offset = -offset;
            } else {
                printf("*");
            }
            offset--;
            printf("%s ", &tasks->all_strs[offset]);
        }
        printf("\n\n");
    }

    struct timeval t0, t1;
    gettimeofday(&t0, 0);
    long elapsed_millis;

    cl_int errcode;
    for (int i=0; i<num_tasks; i++) {
        anactx_submit_permut_task(anactx, tasks+i);
    }
    errcode = anactx_flush_tasks_buffer(anactx);
    ret_iferr(errcode, "failed to flush tasks buffer");
    errcode = anactx_wait_for_cur_kernel(anactx);
    ret_iferr(errcode, "failed to wait for last kernel");

    gettimeofday(&t1, 0);
    elapsed_millis = (t1.tv_sec - t0.tv_sec) * 1000 + (t1.tv_usec - t0.tv_usec) / 1000;
    char char_buf[1024];
    format_bignum(1000L * MAX_ITERS_PER_TASK * num_tasks / elapsed_millis, char_buf, 1000);
    printf("[Thread %d] took %.2fsec, speed: ~%sHash/s\n", anactx->thread_id, elapsed_millis / 10 / 100.0, char_buf);

    return CL_SUCCESS;
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
    clGetDeviceIDs (platform_id, CL_DEVICE_TYPE_ALL, 0, NULL, &num_devices);
    cl_device_id device_ids[num_devices];
    clGetDeviceIDs (platform_id, CL_DEVICE_TYPE_ALL, num_devices, device_ids, &num_devices);
    ret_iferr(!num_devices, "no devices");

    uint32_t num_gpus = 0;
    for (int i=0; i<num_devices; i++) {
        cl_device_type dev_type;
        clGetDeviceInfo (device_ids[i], CL_DEVICE_TYPE, sizeof(dev_type), &dev_type, NULL);
        if (dev_type > CL_DEVICE_TYPE_CPU) {
            num_gpus++;
        }
    }

    if (num_gpus) {
        int d=0;
        for (int s=0; s<num_devices; s++) {
            cl_device_type dev_type;
            clGetDeviceInfo (device_ids[s], CL_DEVICE_TYPE, sizeof(dev_type), &dev_type, NULL);
            if (dev_type > CL_DEVICE_TYPE_CPU) {
                device_ids[d++] = device_ids[s];
            }
        }
        num_devices = num_gpus;
    }

    char char_buf[1024];
    for (int i=0; i<num_devices; i++) {
        cl_ulong local_mem; char local_mem_str[32];
        cl_ulong global_mem; char global_mem_str[32];

        clGetDeviceInfo(device_ids[i], CL_DEVICE_GLOBAL_MEM_SIZE, 8, &global_mem, NULL);
        clGetDeviceInfo(device_ids[i], CL_DEVICE_LOCAL_MEM_SIZE, 8, &local_mem, NULL);
        clGetDeviceInfo (device_ids[i], CL_DEVICE_NAME, 1024, char_buf, NULL);

        format_bignum(global_mem, global_mem_str, 1024);
        format_bignum(local_mem, local_mem_str, 1024);
        printf("OpenCL device #%d: %s (g:%siB l:%siB)\n", i+1, char_buf, global_mem_str, local_mem_str);
    }
    printf("\n");

    uint32_t *hashes;
    const uint32_t hashes_num = read_hashes("input.hashes", &hashes);
    ret_iferr(!hashes_num, "failed to read hashes");
    ret_iferr(!hashes, "failed to allocate hashes");

    cl_int errcode;
    anactx anactxs[num_devices];
    for (uint32_t i=0; i<num_devices; i++) {
        errcode = anactx_create(&anactxs[i], platform_id, device_ids[i]);
        ret_iferr(errcode, "failed to create anactx");

        anactxs[i].num_threads = num_devices;
        anactxs[i].thread_id = i;

        errcode = anactx_set_input_hashes(&anactxs[i], hashes, hashes_num);
        ret_iferr(errcode, "failed to set input hashes");
    }

    for (int i=0; i<1; i++) {
        printf("iter\n");
        run_kernel(&anactxs[0]);
    }

    const uint32_t *hashes_reversed = anactx_read_hashes_reversed(&anactxs[0], &errcode);
    ret_iferr(errcode, "failed to read hashes_reversed");

    for(int i=0; i<hashes_num; i++) {
        char hash_ascii[33];
        hash_to_ascii(&hashes[i*4], hash_ascii);
        printf("%s:  %s\n", hash_ascii, (char*)&hashes_reversed[i*MAX_STR_LENGTH/4]);
    }

    for (int i=0; i<num_devices; i++) {
        anactx_free(&anactxs[i]);
    }

    return 0;
}