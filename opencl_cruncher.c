#include "opencl_cruncher.h"

#ifdef HAVE_OPENCL

#include "gpu_cruncher.h"

#define MAX_OPENCL_DEVICES 16
static cl_platform_id s_platform_id;
static cl_device_id s_device_ids[MAX_OPENCL_DEVICES];
static uint32_t s_num_devices = 0;

static uint32_t opencl_probe(void) {
    cl_uint num_platforms;
    clGetPlatformIDs(1, &s_platform_id, &num_platforms);
    if (!num_platforms) return 0;

    cl_uint num_all;
    clGetDeviceIDs(s_platform_id, CL_DEVICE_TYPE_ALL, 0, NULL, &num_all);
    if (!num_all) return 0;

    cl_device_id all_ids[MAX_OPENCL_DEVICES];
    uint32_t to_get = num_all > MAX_OPENCL_DEVICES ? MAX_OPENCL_DEVICES : num_all;
    clGetDeviceIDs(s_platform_id, CL_DEVICE_TYPE_ALL, to_get, all_ids, &num_all);

    // Prefer GPU devices over CPU
    s_num_devices = 0;
    uint32_t num_gpus = 0;
    for (uint32_t i = 0; i < num_all && i < MAX_OPENCL_DEVICES; i++) {
        cl_device_type dtype;
        clGetDeviceInfo(all_ids[i], CL_DEVICE_TYPE, sizeof(dtype), &dtype, NULL);
        if (dtype > CL_DEVICE_TYPE_CPU) num_gpus++;
    }

    if (num_gpus > 0) {
        for (uint32_t i = 0; i < num_all && i < MAX_OPENCL_DEVICES; i++) {
            cl_device_type dtype;
            clGetDeviceInfo(all_ids[i], CL_DEVICE_TYPE, sizeof(dtype), &dtype, NULL);
            if (dtype > CL_DEVICE_TYPE_CPU)
                s_device_ids[s_num_devices++] = all_ids[i];
        }
    } else {
        for (uint32_t i = 0; i < num_all && i < MAX_OPENCL_DEVICES; i++)
            s_device_ids[s_num_devices++] = all_ids[i];
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
    int err = gpu_cruncher_ctx_create(gctx, s_platform_id, s_device_ids[instance_id],
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
