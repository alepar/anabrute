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

    char *const kernel_source = read_file("kernels/md5.cl");
    die_iferr(!kernel_source, "failed to read kernel source");
    size_t lengths[] = {strlen(kernel_source)};
    const char *sources[] = {kernel_source};

    const cl_context_properties ctx_props [] = { CL_CONTEXT_PLATFORM, platform_id, 0, 0 };
    cl_context ctxs[num_devices];

    cl_program programs[num_devices];
    cl_kernel md5_kernels[num_devices];

    uint32_t data_info[2];
    uint32_t keys[16];
    uint32_t hashes[4];

    data_info[0] = 36; // KEY_LENGTH, 9 uints per key
    data_info[1] = 1;  // num_keys
    char *key = "tyranous pluto twits";
    memcpy(keys, key, strlen(key) + 1);

    cl_mem data_info_bufs[num_devices];
    cl_mem keys_bufs[num_devices];
    cl_mem hashes_bufs[num_devices];

    cl_command_queue queue[num_devices];

    cl_int errcode;
    for (int i=0; i<num_devices; i++) {
        ctxs[i] = clCreateContext(ctx_props, 1, &device_ids[i], NULL, NULL, &errcode);
        die_iferr(errcode, "failed to create context");

        // loading kernel
        programs[i] = clCreateProgramWithSource(ctxs[i], 1, sources, lengths, &errcode);
        die_iferr(errcode, "failed to create program");
        errcode = clBuildProgram(programs[i], 0, NULL, NULL, NULL, NULL);
        die_iferr(errcode, "failed to build program");

        md5_kernels[i] = clCreateKernel(programs[i], "md5", &errcode);
        die_iferr(errcode, "failed to create kernel");

        data_info_bufs[i] = clCreateBuffer(ctxs[i], CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(data_info), data_info, &errcode);
        die_iferr(errcode, "failed to create data_info_buf");
        keys_bufs[i] = clCreateBuffer(ctxs[i], CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(keys), keys, &errcode);
        die_iferr(errcode, "failed to create keys_buf");
        hashes_bufs[i] = clCreateBuffer(ctxs[i], CL_MEM_WRITE_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(hashes), hashes, &errcode);
        die_iferr(errcode, "failed to create hashes_buf");

        clSetKernelArg(md5_kernels[i], 0, sizeof (cl_mem), &data_info_bufs[i]);
        clSetKernelArg(md5_kernels[i], 1, sizeof (cl_mem), &keys_bufs[i]);
        clSetKernelArg(md5_kernels[i], 2, sizeof (cl_mem), &hashes_bufs[i]);

        queue[i] = clCreateCommandQueue(ctxs[i], device_ids[i], NULL, &errcode);
        die_iferr(errcode, "failed to create command queue");
    }

    while(1) {
        struct timeval t0;
        gettimeofday(&t0, 0);

        const size_t globalWorkSize[] = {32, 1024, 1024};
        cl_event evt[num_devices];
        for (int i=0; i<num_devices; i++) {
            errcode = clEnqueueNDRangeKernel(queue[i], md5_kernels[i], 3, NULL, globalWorkSize, NULL, 0, NULL, &evt[i]);
            die_iferr(errcode, "failed to enqueue kernel");
        }

        errcode = clWaitForEvents(num_devices, evt);
        die_iferr(errcode, "failed to wait for completion");

//        errcode = clEnqueueReadBuffer (queue, hashes_buf, CL_TRUE, 0, sizeof (hashes), hashes, 0, NULL, NULL);
//        die_iferr(errcode, "failed to read hashes");

        struct timeval t1;
        gettimeofday(&t1, 0);
        long elapsed_millis = (t1.tv_sec - t0.tv_sec) * 1000 + (t1.tv_usec - t0.tv_usec) / 1000;

        format_bignum(1000L * 500 * globalWorkSize[0] * globalWorkSize[1] * globalWorkSize[2] * num_devices / elapsed_millis, char_buf, 1000);
        printf("kernel took %.2fsec, speed: ~%sHash/s\n", elapsed_millis / 10 / 100.0, char_buf);
    }

    hash_to_ascii(hashes, char_buf);
    printf("hash: %s\n", char_buf);

    for (int i=0; i<num_devices; i++) {
        clReleaseCommandQueue (queue[i]);

        clReleaseMemObject(data_info_bufs[i]);
        clReleaseMemObject(keys_bufs[i]);
        clReleaseMemObject(hashes_bufs[i]);

        clReleaseKernel(md5_kernels[i]);

        clReleaseProgram(programs[i]);

        clReleaseContext(ctxs[i]);
    }

    return 0;
}