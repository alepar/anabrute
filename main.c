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

typedef struct {
    char all_strs[38]; // 36 max
    char offsets[18]; // positives - permutable, negatives - fixed, zeroes - empty; abs(offset)-1 to get offset in all_strs
    uint64_t start_from;
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

permut_template* permut_templates_create(const int num_templates) {
    permut_template *templates = malloc(num_templates*sizeof(permut_template));

    if (templates == NULL) {
        return NULL;
    }

    char all_strs[38] = "x\0a\0b\0c\0d\0e\0f\0g\0h\0i\0j\0k\0l\0m\0";
    char offsets[18] = {-1, 3, 5, 7, 9, -1, 11, 13, 15, 17, 19, 21, 23, -1, 25, 27};

    for (int i=0; i<num_templates; i++) {
        templates[i].start_from = 1024L*i+1;
        memcpy(templates[i].all_strs, all_strs, 38);
        memcpy(templates[i].offsets, offsets, 18);
    }

    return templates;
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

    const uint32_t num_templates = 2048; // 13!
    const uint32_t iters_per_item = 256;
    permut_template *permut_templates = permut_templates_create(num_templates);
    die_iferr(!permut_templates, "failed to create permut_templates");
    uint32_t *hashes = malloc(num_templates*iters_per_item*4);
    die_iferr(!hashes, "failed to allocate hashes");

    cl_mem permut_templates_bufs[num_devices];
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
        hashes_bufs[i] = clCreateBuffer(ctxs[i], CL_MEM_WRITE_ONLY, num_templates*iters_per_item*4, NULL, &errcode);
        die_iferr(errcode, "failed to create hashes_bufs");

        clSetKernelArg(permut_kernels[i], 0, sizeof (cl_mem), &permut_templates_bufs[i]);
        clSetKernelArg(permut_kernels[i], 1, sizeof (iters_per_item), &iters_per_item);
        clSetKernelArg(permut_kernels[i], 2, sizeof (cl_mem), &hashes_bufs[i]);

        queue[i] = clCreateCommandQueue(ctxs[i], device_ids[i], NULL, &errcode);
        die_iferr(errcode, "failed to create command queue");
    }

    struct timeval t0;
    gettimeofday(&t0, 0);

    const size_t globalWorkSize[] = {num_templates, 0, 0};
    cl_event evt[num_devices];

//    while (1) {
        for (int i=0; i<num_devices; i++) {
            errcode = clEnqueueNDRangeKernel(queue[i], permut_kernels[i], 1, NULL, globalWorkSize, NULL, 0, NULL, &evt[i]);
            die_iferr(errcode, "failed to enqueue kernel");
        }

        errcode = clWaitForEvents(num_devices, evt);
        die_iferr(errcode, "failed to wait for completion");

        struct timeval t1;
        gettimeofday(&t1, 0);
        long elapsed_millis = (t1.tv_sec - t0.tv_sec) * 1000 + (t1.tv_usec - t0.tv_usec) / 1000;

        format_bignum(1000L * iters_per_item * globalWorkSize[0] /** globalWorkSize[1] * globalWorkSize[2]*/ * num_devices / elapsed_millis, char_buf, 1000);
        printf("kernel took %.2fsec, speed: ~%sHash/s\n", elapsed_millis / 10 / 100.0, char_buf);
//    }

    errcode = clEnqueueReadBuffer (queue[0], hashes_bufs[0], CL_TRUE, 0, num_templates*iters_per_item*4, hashes, 0, NULL, NULL);
    die_iferr(errcode, "failed to read hashes");

    char hash_ascii_buf[33];
    FILE *f = fopen("perms_gpu_hashed.txt", "w");
    for (int i=0; i<num_templates*iters_per_item; i++) {
        hash_to_ascii(&hashes[i*4], hash_ascii_buf);
        fprintf(f, "%s\n", hash_ascii_buf);
    }
    fclose(f);

    for (int i=0; i<num_devices; i++) {
        clReleaseCommandQueue (queue[i]);

        clReleaseMemObject(permut_templates_bufs[i]);
        clReleaseMemObject(hashes_bufs[i]);

        clReleaseKernel(permut_kernels[i]);

        clReleaseProgram(programs[i]);

        clReleaseContext(ctxs[i]);
    }

    return 0;
}