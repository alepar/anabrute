#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <time.h>

#ifdef __APPLE__
    #include "OpenCL/opencl.h"
    #include <unitypes.h>
    #include <event.h>
#else
    #include "CL/cl.h"
    #include <sys/time.h>
#endif

#define MAX_STR_LENGTH 40
#define MAX_OFFSETS_LENGTH 20

typedef struct {
    char all_strs[MAX_STR_LENGTH];
    char offsets[MAX_OFFSETS_LENGTH];  // positives - permutable, negatives - fixed, zeroes - empty; abs(offset)-1 to get offset in all_strs
    uint start_from;
} permut_template;

#define die_iferr(val, msg) \
if (val) {\
    fprintf(stderr, "FATAL: %d - %s\n", val, msg);\
    return val;\
}

char* read_file(const char* filename) {
    FILE *fd = fopen(filename, "r");
    if (fd == NULL) {
        return NULL;
    }

    fseek(fd, 0L, SEEK_END);
    const size_t filesize = (size_t) ftell(fd);
    rewind(fd);

    char *buf = (char *)malloc(filesize+1);
    const size_t read = fread(buf, 1, filesize, fd);
    
    if (filesize != read) {
        free(buf);
        return NULL;
    }

    return buf;
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

    char *const kernel_source = read_file("kernels/permut.cl");
    die_iferr(!kernel_source, "failed to read kernel source");
    size_t lengths[] = {strlen(kernel_source)};
    const char *sources[] = {kernel_source};

    const cl_context_properties ctx_props [] = { CL_CONTEXT_PLATFORM, platform_id, 0, 0 };
    cl_context ctxs[num_devices];

    cl_program programs[num_devices];
    cl_kernel permut_kernels[num_devices];

    const uint32_t num_templates = 1024*1024; // peak at ~256-512K
    const uint32_t iters_per_item = 512; // peak at ~512
    permut_template *permut_templates = permut_templates_create(num_templates, iters_per_item);
    die_iferr(!permut_templates, "failed to create permut_templates");

    uint32_t *hashes;
    const uint32_t hashes_num = read_hashes("input.hashes", &hashes);
    die_iferr(!hashes_num, "failed to read hashes");
    die_iferr(!hashes, "failed to allocate hashes");

    const size_t hashes_reversed_len = hashes_num * MAX_STR_LENGTH;
    uint32_t *hashes_reversed = malloc(hashes_reversed_len);
    die_iferr(!hashes_reversed, "failed to allocate hashes_reversed");

    cl_mem permut_templates_bufs[num_devices];
    cl_mem hashes_bufs[num_devices];
    cl_mem hashes_reversed_bufs[num_devices];

    cl_command_queue queue[num_devices];

    cl_int errcode;
    for (int i=0; i<num_devices; i++) {
        ctxs[i] = clCreateContext(ctx_props, 1, &device_ids[i], NULL, NULL, &errcode);
        die_iferr(errcode, "failed to create context");

        // loading kernel
        programs[i] = clCreateProgramWithSource(ctxs[i], 1, sources, lengths, &errcode);
        die_iferr(errcode, "failed to create program");
        errcode = clBuildProgram(programs[i], 0, NULL, NULL, NULL, NULL);
        if (errcode == CL_BUILD_PROGRAM_FAILURE) {
            // Determine the size of the log
            size_t log_size;
            clGetProgramBuildInfo(programs[i], device_ids[i], CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
            // Allocate memory for the log
            char *log = (char *) malloc(log_size);
            // Get the log
            clGetProgramBuildInfo(programs[i], device_ids[i], CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
            // Print the log
            printf("%s\n", log);
        }
        die_iferr(errcode, "failed to build program");

        permut_kernels[i] = clCreateKernel(programs[i], "permut", &errcode);
        die_iferr(errcode, "failed to create kernel");

        permut_templates_bufs[i] = clCreateBuffer(ctxs[i], CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, num_templates*sizeof(permut_template), permut_templates, &errcode);
        die_iferr(errcode, "failed to create permut_templates_bufs");
        hashes_bufs[i] = clCreateBuffer(ctxs[i], CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, hashes_num*16, hashes, &errcode);
        die_iferr(errcode, "failed to create hashes_bufs");
        hashes_reversed_bufs[i] = clCreateBuffer(ctxs[i], CL_MEM_WRITE_ONLY, hashes_reversed_len, NULL, &errcode);
        die_iferr(errcode, "failed to create hashes_reversed_bufs");

/*
        __kernel void permut(
            __global const permut_template *permut_templates,
                     const uint iters_per_item,
            __global uint *hashes,
                     uint hashes_num,
            __global uint *hashes_reversed)
*/
        clSetKernelArg(permut_kernels[i], 0, sizeof (cl_mem), &permut_templates_bufs[i]);
        clSetKernelArg(permut_kernels[i], 1, sizeof (iters_per_item), &iters_per_item);
        clSetKernelArg(permut_kernels[i], 2, sizeof (cl_mem), &hashes_bufs[i]);
        clSetKernelArg(permut_kernels[i], 3, sizeof (hashes_num), &hashes_num);
        clSetKernelArg(permut_kernels[i], 4, sizeof (cl_mem), &hashes_reversed_bufs[i]);

        queue[i] = clCreateCommandQueue(ctxs[i], device_ids[i], NULL, &errcode);
        die_iferr(errcode, "failed to create command queue");
    }

    struct timeval t0;
    gettimeofday(&t0, 0);

    const size_t globalWorkSize[] = {num_templates, 0, 0};
    cl_event evt[num_devices];

    for (int i=0; i<num_devices; i++) {
        errcode = clEnqueueNDRangeKernel(queue[i], permut_kernels[i], 1, NULL, globalWorkSize, NULL, 0, NULL, &evt[i]);
        die_iferr(errcode, "failed to enqueue kernel");
    }

    errcode = clWaitForEvents(num_devices, evt);
    die_iferr(errcode, "failed to wait for completion");

    struct timeval t1;
    gettimeofday(&t1, 0);
    long elapsed_millis = (t1.tv_sec - t0.tv_sec) * 1000 + (t1.tv_usec - t0.tv_usec) / 1000;

    format_bignum(1000L * iters_per_item * num_templates * num_devices / elapsed_millis, char_buf, 1000);
    printf("kernel took %.2fsec, speed: ~%sHash/s\n", elapsed_millis / 10 / 100.0, char_buf);

    errcode = clEnqueueReadBuffer (queue[0], hashes_reversed_bufs[0], CL_TRUE, 0, hashes_reversed_len, hashes_reversed, 0, NULL, NULL);
    die_iferr(errcode, "failed to read hashes_reversed");

    for(int i=0; i<hashes_num; i++) {
        char hash_ascii[33];
        hash_to_ascii(&hashes[i*4], hash_ascii);
        printf("%s:  %s\n", hash_ascii, (char*)&hashes_reversed[i*MAX_STR_LENGTH/4]);
    }

    for (int i=0; i<num_devices; i++) {
        clReleaseCommandQueue (queue[i]);

        clReleaseMemObject(permut_templates_bufs[i]);
        clReleaseMemObject(hashes_bufs[i]);
        clReleaseMemObject(hashes_reversed_bufs[i]);

        clReleaseKernel(permut_kernels[i]);

        clReleaseProgram(programs[i]);

        clReleaseContext(ctxs[i]);
    }

    return 0;
}