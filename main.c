#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <time.h>

#ifdef __APPLE__
    #include <unitypes.h>
    #include <event.h>
#else
    #include <sys/time.h>
#endif

#include "ocl_types.h"

#define die_iferr(val, msg) \
if (val) {\
    fprintf(stderr, "FATAL: %d - %s\n", val, msg);\
    return val;\
}

static const char* size_suffixes[] = {"", "K", "M", "G", "T", "P"};
void format_bignum(uint64_t size, char *dst, uint16_t div) {
    int divs = 0;
    while (size/div > 1) {
        size = size/div;
        divs++;
    }
    sprintf(dst, "%lu%s", size, size_suffixes[divs]);
}

void hash_to_ascii(const uint32_t *hash, char *buf) {
    int di = 0;
    for(int si=0; si<4; si++) {
        buf[di++] = (hash[si] & 0x000000f0) >>  4;
        buf[di++] = (hash[si] & 0x0000000f)      ;

        buf[di++] = (hash[si] & 0x0000f000) >> 12;
        buf[di++] = (hash[si] & 0x00000f00) >>  8;

        buf[di++] = (hash[si] & 0x00f00000) >> 20;
        buf[di++] = (hash[si] & 0x000f0000) >> 16;

        buf[di++] = (hash[si] & 0xf0000000) >> 28;
        buf[di++] = (hash[si] & 0x0f000000) >> 24;
    }

    for(int i=0; i<32; i++) {
        if (buf[i] > 9) {
            buf[i] += 'a' - 10;
        } else {
            buf[i] += '0';
        }
    }

    buf[di] = 0;
}

void ascii_to_hash(const char *buf, uint32_t *hash) {
    char *hash_bytes = (char *)hash;
    for (int i=0; i<16; i++) {
        char l = buf[i*2], r = buf[i*2+1];
        if (l > '9') l-= 'a' - 10;
        else l-='0';
        if (r > '9') r-= 'a' - 10;
        else r-='0';
        hash_bytes[i] = l<<4 | r;
    }
}

permut_template* permut_templates_create(const uint32_t num_templates, const uint32_t iters_per_item) {
    permut_template *templates = malloc(num_templates*sizeof(permut_template));

    if (templates == NULL) {
        return NULL;
    }

    char all_strs[MAX_STR_LENGTH] = "x\0a\0b\0c\0d\0e\0f\0g\0h\0i\0j\0k\0l\0m\0";
    char offsets[MAX_OFFSETS_LENGTH] = {-1, 3, 5, 7, 9, -1, 11, 13, 15, 17, 19, 21, 23, -1, 25, 27};

    for (uint32_t i=0; i<num_templates; i++) {
        templates[i].start_from = (uint)(iters_per_item*i+1);
        memcpy(templates[i].all_strs, all_strs, MAX_STR_LENGTH);
        memcpy(templates[i].offsets, offsets, MAX_OFFSETS_LENGTH);
    }

    return templates;
}

const uint32_t read_hashes(char *file_name, uint32_t **hashes) {
    FILE *const fd = fopen(file_name, "r");
    if (!fd) {
        return 0;
    }

    fseek(fd, 0L, SEEK_END);
    const long file_size = ftell(fd);
    rewind(fd);

    const long hashes_num_est = (file_size + 1) / 33;
    long hashes_num = 0;

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
    die_iferr(!num_platforms, "no platforms");

    cl_uint num_devices;
    clGetDeviceIDs (platform_id, CL_DEVICE_TYPE_ALL, 0, NULL, &num_devices);
    cl_device_id device_ids[num_devices];
    clGetDeviceIDs (platform_id, CL_DEVICE_TYPE_ALL, num_devices, device_ids, &num_devices);
    die_iferr(!num_devices, "no devices");

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

    const uint32_t num_templates = 256*1024; // peak at ~256-512K
    const uint32_t iters_per_item = 512; // peak at ~512
    permut_template *permut_templates = permut_templates_create(num_templates, iters_per_item);
    die_iferr(!permut_templates, "failed to create permut_templates");

    uint32_t *hashes;
    const uint32_t hashes_num = read_hashes("input.hashes", &hashes);
    die_iferr(!hashes_num, "failed to read hashes");
    die_iferr(!hashes, "failed to allocate hashes");

    cl_int errcode;
    anactx anactxs[num_devices];
    anakrnl_permut permut_kernels[num_devices];
    for (int i=0; i<num_devices; i++) {
        errcode = anactx_create(&anactxs[i], platform_id, device_ids[i]);
        die_iferr(errcode, "failed to create anactx");

        errcode = anactx_set_input_hashes(&anactxs[i], hashes, hashes_num);
        die_iferr(errcode, "failed to set input hashes");

        errcode = anakrnl_permut_create(&permut_kernels[i], &anactxs[i], iters_per_item, permut_templates, num_templates);
        die_iferr(errcode, "failed to create kernel");
    }

    struct timeval t0;
    gettimeofday(&t0, 0);

    for (int i=0; i<num_devices; i++) {
        errcode = anakrnl_permut_enqueue(&permut_kernels[i]);
        die_iferr(errcode, "failed to enqueue kernel");
    }

    for (int i=0; i<num_devices; i++) {
        errcode = anakrnl_permut_wait(&permut_kernels[i]);
        die_iferr(errcode, "failed to wait for completion");
    }

    struct timeval t1;
    gettimeofday(&t1, 0);
    long elapsed_millis = (t1.tv_sec - t0.tv_sec) * 1000 + (t1.tv_usec - t0.tv_usec) / 1000;

    format_bignum(1000L * iters_per_item * num_templates * num_devices / elapsed_millis, char_buf, 1000);
    printf("kernel took %.2fsec, speed: ~%sHash/s\n", elapsed_millis / 10 / 100.0, char_buf);

    const uint32_t *hashes_reversed = anactx_read_hashes_reversed(&anactxs[0], &errcode);
    die_iferr(errcode, "failed to read hashes_reversed");

    for(int i=0; i<hashes_num; i++) {
        char hash_ascii[33];
        hash_to_ascii(&hashes[i*4], hash_ascii);
        printf("%s:  %s\n", hash_ascii, (char*)&hashes_reversed[i*MAX_STR_LENGTH/4]);
    }

    for (int i=0; i<num_devices; i++) {
        anakrnl_permut_free(&permut_kernels[i]);
        anactx_free(&anactxs[i]);
    }

    return 0;
}