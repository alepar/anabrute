#include "CL/cl.h"
#include <stdio.h>
#include <string.h>
#include <unistd.h>

#define die_unless(val, msg) \
if (!(val)) {\
    fprintf(stderr, "FATAL: %s\n", msg);\
    return -2;\
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


static const char* size_suffixes[] = {"", "KiB", "MiB", "GiB", "TiB", "PiB"};
void format_size(uint64_t size, char* dst) {
    int divs = 0;
    while (size/1024 > 1) {
        size = size/1024;
        divs++;
    }
    sprintf(dst, "%lu%s", size, size_suffixes[divs]);
}

int main(int argc, char *argv[]) {
    printf("Cpu cores: %d\n", num_cpu_cores());

    cl_platform_id platform_id;
    cl_uint num_platforms;
    clGetPlatformIDs (1, &platform_id, &num_platforms);
    die_unless(num_platforms, "no platforms");

    cl_uint num_devices;
    clGetDeviceIDs (platform_id, CL_DEVICE_TYPE_ALL, 0, NULL, &num_devices);
    cl_device_id device_ids[num_devices];
    clGetDeviceIDs (platform_id, CL_DEVICE_TYPE_ALL, num_devices, device_ids, &num_devices);
    die_unless(num_devices, "no devices");

    char char_buf[1024];
    printf("OpenCL devices: ");
    for (int i=0; i<num_devices; i++) {
        cl_ulong local_mem; char local_mem_str[32];
        cl_ulong global_mem; char global_mem_str[32];

        clGetDeviceInfo(device_ids[0], CL_DEVICE_GLOBAL_MEM_SIZE, 8, &global_mem, NULL);
        clGetDeviceInfo(device_ids[0], CL_DEVICE_LOCAL_MEM_SIZE, 8, &local_mem, NULL);
        clGetDeviceInfo (device_ids[i], CL_DEVICE_NAME, 1024, char_buf, NULL);

        format_size(global_mem, global_mem_str);
        format_size(local_mem, local_mem_str);
        printf("%s (g:%s l:%s), ", char_buf, global_mem_str, local_mem_str);
    }
    printf("\n");

    const cl_context_properties ctx_props [] = { CL_CONTEXT_PLATFORM, platform_id, 0, 0 };

    cl_int errcode;
    cl_context ctx = clCreateContext (ctx_props, num_devices, device_ids, NULL, NULL, &errcode);
    die_unless(errcode == CL_SUCCESS, "failed to create context");

    char *const kernel_source = read_file("kernels/saxpy.cl");
    die_unless(kernel_source, "failed to read kernel source");
    
    size_t lengths[] = {strlen(kernel_source)};
    const char *sources[] = {kernel_source};

    cl_program program = clCreateProgramWithSource(ctx, 1, sources, lengths, &errcode);
    die_unless(errcode == CL_SUCCESS, "failed to create program");

    errcode = clBuildProgram (program, num_devices, device_ids, NULL, NULL, NULL);
    die_unless(errcode == CL_SUCCESS, "failed to build program");

    cl_kernel kernel = clCreateKernel (program, "SAXPY", &errcode);
    die_unless(errcode == CL_SUCCESS, "failed to create kernel");

    float a[1024], b[1024];
    for(int i=0; i<1024; i++ ) {
        a[i] = i;
        b[i] = 1024-i;
    }

    cl_mem a_buf = clCreateBuffer (ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof (a), a, &errcode);
    die_unless(errcode == CL_SUCCESS, "failed to copy a");
    cl_mem b_buf = clCreateBuffer (ctx, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof (b), b, &errcode);
    die_unless(errcode == CL_SUCCESS, "failed to copy b");

    cl_command_queue queue = clCreateCommandQueueWithProperties(ctx, device_ids[0], NULL, &errcode);
    die_unless(errcode == CL_SUCCESS, "failed to create command queue");

    clSetKernelArg(kernel, 0, sizeof (cl_mem), &a_buf);
    clSetKernelArg(kernel, 1, sizeof (cl_mem), &b_buf);
    const float two = 2.0f;
    clSetKernelArg (kernel, 2, sizeof (float), &two);

    const size_t globalWorkSize [] = { 1024, 0, 0 };
    errcode = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, globalWorkSize, NULL, 0, NULL, NULL);
    die_unless(errcode == CL_SUCCESS, "failed to enqueue kernel");

    float b_res[1024];
    errcode = clEnqueueReadBuffer (queue, b_buf, CL_TRUE, 0, sizeof (float) * 1024, b_res, 0, NULL, NULL);
    die_unless(errcode == CL_SUCCESS, "failed to read buffer");

    for(int i=0; i<1024; i++ ) {
        printf("%.2f\t%.2f\t%.2f\n", a[i], b[i], b_res[i]);
    }

    clReleaseCommandQueue (queue);

    clReleaseMemObject(b_buf);
    clReleaseMemObject(a_buf);

    clReleaseKernel(kernel);
    clReleaseProgram(program);

    clReleaseContext(ctx);

    return 0;
}