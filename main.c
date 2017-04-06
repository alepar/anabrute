#ifdef __APPLE__
#include "OpenCL/opencl.h"
#else
#include "CL/cl.h"
#endif

#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <unitypes.h>
#include <time.h>
#include <event.h>

#define die_iferr(val, msg) \
if (val) {\
    fprintf(stderr, "FATAL: %d - %s\n", val, msg);\
    return val;\
}

uint32_t num_cpu_cores() {
    // posix-way
    return (uint32_t) sysconf(_SC_NPROCESSORS_ONLN);
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

void hash_to_ascii(uint32_t *hash_ints, char *buf) {
    int di = 0;
    for(int si=0; si<4; si++) {
        buf[di++] = (hash_ints[si] & 0x000000f0) >>  4;
        buf[di++] = (hash_ints[si] & 0x0000000f)      ;

        buf[di++] = (hash_ints[si] & 0x0000f000) >> 12;
        buf[di++] = (hash_ints[si] & 0x00000f00) >>  8;

        buf[di++] = (hash_ints[si] & 0x00f00000) >> 20;
        buf[di++] = (hash_ints[si] & 0x000f0000) >> 16;

        buf[di++] = (hash_ints[si] & 0xf0000000) >> 28;
        buf[di++] = (hash_ints[si] & 0x0f000000) >> 24;
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

int main(int argc, char *argv[]) {
    const uint32_t num_cpus = num_cpu_cores();
    printf("Cpu cores: %d\n", num_cpus);

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

    const cl_context_properties ctx_props [] = { CL_CONTEXT_PLATFORM, platform_id, 0, 0 };

    cl_int errcode;
    cl_context ctx = clCreateContext (ctx_props, num_devices, device_ids, NULL, NULL, &errcode);
    die_iferr(errcode, "failed to create context");

    // loading kernel
    char *const kernel_source = read_file("kernels/md5.cl");
    die_iferr(!kernel_source, "failed to read kernel source");
    size_t lengths[] = {strlen(kernel_source)};
    const char *sources[] = {kernel_source};
    cl_program program = clCreateProgramWithSource(ctx, 1, sources, lengths, &errcode);
    die_iferr(errcode, "failed to create program");
    errcode = clBuildProgram (program, num_devices, device_ids, NULL, NULL, NULL);
    die_iferr(errcode, "failed to build program");
    cl_kernel md5_kernel = clCreateKernel (program, "md5", &errcode);
    die_iferr(errcode, "failed to create kernel");

    // loading data
    uint32_t data_info[2];
    uint32_t keys[16];
    uint32_t hashes[4];

    data_info[0] = 36; // KEY_LENGTH, 9 uints per key
    data_info[1] = 1;  // num_keys
    char *key = "tyranous pluto twits";
    memcpy(keys, key, strlen(key)+1);

    cl_mem data_info_buf = clCreateBuffer (ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof (data_info), data_info, &errcode);
    die_iferr(errcode, "failed to create data_info_buf");
    cl_mem keys_buf = clCreateBuffer (ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof (keys), keys, &errcode);
    die_iferr(errcode, "failed to create keys_buf");
    cl_mem hashes_buf = clCreateBuffer (ctx, CL_MEM_WRITE_ONLY | CL_MEM_COPY_HOST_PTR, sizeof (hashes), hashes, &errcode);
    die_iferr(errcode, "failed to create hashes_buf");

    // running kernel
    cl_command_queue queue = clCreateCommandQueue(ctx, device_ids[0], NULL, &errcode);
    die_iferr(errcode, "failed to create command queue");

    clSetKernelArg(md5_kernel, 0, sizeof (cl_mem), &data_info_buf);
    clSetKernelArg(md5_kernel, 1, sizeof (cl_mem), &keys_buf);
    clSetKernelArg(md5_kernel, 2, sizeof (cl_mem), &hashes_buf);


    struct timeval t0;
    gettimeofday(&t0, 0);
    const size_t globalWorkSize [] = { 1024, 1024, 1024 };
    errcode = clEnqueueNDRangeKernel(queue, md5_kernel, 3, NULL, globalWorkSize, NULL, 0, NULL, NULL);
    die_iferr(errcode, "failed to enqueue kernel");

    errcode = clEnqueueReadBuffer (queue, hashes_buf, CL_TRUE, 0, sizeof (hashes), hashes, 0, NULL, NULL);
    die_iferr(errcode, "failed to read hashes");

    struct timeval t1;
    gettimeofday(&t1, 0);
    long elapsed_millis = (t1.tv_sec-t0.tv_sec)*1000 + (t1.tv_usec-t0.tv_usec)/1000;
    format_bignum(1000L*1024*1024*1024/elapsed_millis, char_buf, 1000);
    printf("kernel took %.2fsec, speed: ~%sHash/s\n", elapsed_millis/10/100.0, char_buf);

    hash_to_ascii(hashes, char_buf);
    printf("hash: %s\n", char_buf);

    clReleaseCommandQueue (queue);

    clReleaseMemObject(data_info_buf);
    clReleaseMemObject(keys_buf);
    clReleaseMemObject(hashes_buf);

    clReleaseKernel(md5_kernel);
    clReleaseProgram(program);

    clReleaseContext(ctx);

    return 0;
}