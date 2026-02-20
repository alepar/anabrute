#include "opencl_cruncher.h"

#ifdef HAVE_OPENCL

#include "gpu_cruncher.h"

#define MAX_OPENCL_DEVICES 16
#define MAX_OPENCL_PLATFORMS 8
static cl_platform_id s_platform_ids[MAX_OPENCL_DEVICES]; // per-device platform
static cl_device_id s_device_ids[MAX_OPENCL_DEVICES];
static uint32_t s_num_devices = 0;

static uint32_t opencl_probe(void) {
    cl_platform_id platforms[MAX_OPENCL_PLATFORMS];
    cl_uint num_platforms;
    clGetPlatformIDs(MAX_OPENCL_PLATFORMS, platforms, &num_platforms);
    if (!num_platforms) return 0;

    // Gather all GPU devices from all platforms
    s_num_devices = 0;
    for (cl_uint p = 0; p < num_platforms && s_num_devices < MAX_OPENCL_DEVICES; p++) {
        cl_uint num_devs;
        if (clGetDeviceIDs(platforms[p], CL_DEVICE_TYPE_GPU, 0, NULL, &num_devs) != CL_SUCCESS)
            continue;
        cl_device_id devs[MAX_OPENCL_DEVICES];
        uint32_t to_get = num_devs > MAX_OPENCL_DEVICES ? MAX_OPENCL_DEVICES : num_devs;
        clGetDeviceIDs(platforms[p], CL_DEVICE_TYPE_GPU, to_get, devs, &num_devs);
        for (cl_uint d = 0; d < num_devs && s_num_devices < MAX_OPENCL_DEVICES; d++) {
            s_platform_ids[s_num_devices] = platforms[p];
            s_device_ids[s_num_devices] = devs[d];
            s_num_devices++;
        }
    }

    // No GPU devices found (e.g. POCL CPU-only) — don't fall back to CPU devices,
    // native AVX/scalar backends are always faster than OpenCL-on-CPU.
    if (s_num_devices == 0) return 0;

    // Skip integrated GPUs if any discrete GPU is present
    bool have_discrete = false;
    for (uint32_t i = 0; i < s_num_devices; i++) {
        cl_bool unified = CL_FALSE;
        clGetDeviceInfo(s_device_ids[i], CL_DEVICE_HOST_UNIFIED_MEMORY, sizeof(unified), &unified, NULL);
        if (!unified) { have_discrete = true; break; }
    }
    if (have_discrete) {
        uint32_t dst = 0;
        for (uint32_t i = 0; i < s_num_devices; i++) {
            cl_bool unified = CL_FALSE;
            clGetDeviceInfo(s_device_ids[i], CL_DEVICE_HOST_UNIFIED_MEMORY, sizeof(unified), &unified, NULL);
            if (!unified) {
                s_platform_ids[dst] = s_platform_ids[i];
                s_device_ids[dst] = s_device_ids[i];
                dst++;
            } else {
                char name[256];
                clGetDeviceInfo(s_device_ids[i], CL_DEVICE_NAME, sizeof(name), name, NULL);
                printf("  opencl: skipping integrated GPU: %s\n", name);
            }
        }
        s_num_devices = dst;
    }

    for (uint32_t i = 0; i < s_num_devices; i++) {
        char name[256];
        clGetDeviceInfo(s_device_ids[i], CL_DEVICE_NAME, sizeof(name), name, NULL);
        printf("  opencl[%d]: %s\n", i, name);
    }

    return s_num_devices;
}

static int opencl_create(void *ctx, cruncher_config *cfg, uint32_t instance_id) {
    gpu_cruncher_ctx *gctx = ctx;
    int err = gpu_cruncher_ctx_create(gctx, s_platform_ids[instance_id], s_device_ids[instance_id],
                                       cfg->tasks_buffs, cfg->hashes, cfg->hashes_num);
    if (err) return err;
    gctx->cfg = cfg;
    return 0;
}

static void *opencl_run(void *ctx) {
    return run_gpu_cruncher_thread(ctx);
}

static void opencl_get_stats(void *ctx, float *busy_pct, float *anas_per_sec) {
    gpu_cruncher_get_stats(ctx, busy_pct, anas_per_sec);
}

static uint64_t opencl_get_total_anas(void *ctx) {
    return ((gpu_cruncher_ctx *)ctx)->consumed_anas;
}

static bool opencl_is_running(void *ctx) {
    return ((gpu_cruncher_ctx *)ctx)->is_running;
}

static int opencl_destroy(void *ctx) {
    return gpu_cruncher_ctx_free(ctx);
}

cruncher_ops opencl_cruncher_ops = {
    .name = "opencl",
    .probe = opencl_probe,
    .create = opencl_create,
    .run = opencl_run,
    .get_stats = opencl_get_stats,
    .get_total_anas = opencl_get_total_anas,
    .is_running = opencl_is_running,
    .destroy = opencl_destroy,
    .ctx_size = sizeof(gpu_cruncher_ctx),
};

#else /* !HAVE_OPENCL */

static uint32_t opencl_probe_stub(void) { return 0; }

cruncher_ops opencl_cruncher_ops = {
    .name = "opencl",
    .probe = opencl_probe_stub,
    .ctx_size = 0,
};

#endif /* HAVE_OPENCL */
