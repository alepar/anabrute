#import <Metal/Metal.h>
#import <Foundation/Foundation.h>

#include "metal_cruncher.h"
#include "os.h"
#include "fact.h"

extern char* read_file(const char* filename);

typedef struct {
    cruncher_config *cfg;
    id<MTLDevice> device;
    id<MTLCommandQueue> queue;
    id<MTLComputePipelineState> pipeline;
    id<MTLBuffer> buf_hashes;           // hashes (read-only for GPU)
    id<MTLBuffer> buf_hashes_reversed;  // shared output (GPU writes matches)

    volatile bool is_running;
    volatile uint64_t consumed_bufs;
    volatile uint64_t consumed_anas;
    uint64_t task_time_start;
    uint64_t task_time_end;
    uint64_t times_start[TIMES_WINDOW_LENGTH];
    uint64_t times_end[TIMES_WINDOW_LENGTH];
    uint64_t times_anas[TIMES_WINDOW_LENGTH];
    uint32_t times_idx;
} metal_cruncher_ctx;

/* ---- merge hashes_reversed from local Metal buffer to shared output ---- */
static void merge_hashes_reversed(metal_cruncher_ctx *mctx) {
    uint32_t *local = (uint32_t *)[mctx->buf_hashes_reversed contents];
    for (uint32_t i = 0; i < mctx->cfg->hashes_num; i++) {
        if (local[i * MAX_STR_LENGTH / 4]) {
            memcpy(mctx->cfg->hashes_reversed + i * MAX_STR_LENGTH / 4,
                   local + i * MAX_STR_LENGTH / 4, MAX_STR_LENGTH);
        }
    }
}

/* ---- vtable functions ---- */

static uint32_t metal_probe(void) {
    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (device == nil) return 0;
        printf("  metal[0]: %s\n", [[device name] UTF8String]);
        return 1;
    }
}

static int metal_create(void *ctx, cruncher_config *cfg, uint32_t instance_id) {
    @autoreleasepool {
        metal_cruncher_ctx *mctx = ctx;
        mctx->cfg = cfg;
        mctx->is_running = false;
        mctx->consumed_bufs = 0;
        mctx->consumed_anas = 0;
        mctx->task_time_start = 0;
        mctx->task_time_end = 0;
        memset(mctx->times_start, 0, sizeof(mctx->times_start));
        memset(mctx->times_end, 0, sizeof(mctx->times_end));
        memset(mctx->times_anas, 0, sizeof(mctx->times_anas));
        mctx->times_idx = 0;

        mctx->device = MTLCreateSystemDefaultDevice();
        if (mctx->device == nil) {
            fprintf(stderr, "FATAL: Metal device not available\n");
            return -1;
        }

        mctx->queue = [mctx->device newCommandQueue];
        if (mctx->queue == nil) {
            fprintf(stderr, "FATAL: failed to create Metal command queue\n");
            return -1;
        }

        /* Read and compile the Metal kernel source */
        char *source = read_file("kernels/permut.metal");
        if (source == NULL) {
            fprintf(stderr, "FATAL: failed to read kernels/permut.metal\n");
            return -1;
        }

        NSString *sourceStr = [NSString stringWithUTF8String:source];
        free(source);

        NSError *error = nil;
        id<MTLLibrary> library = [mctx->device newLibraryWithSource:sourceStr options:nil error:&error];
        if (library == nil) {
            fprintf(stderr, "FATAL: failed to compile Metal kernel: %s\n",
                    [[error localizedDescription] UTF8String]);
            return -1;
        }

        id<MTLFunction> function = [library newFunctionWithName:@"permut"];
        if (function == nil) {
            fprintf(stderr, "FATAL: Metal kernel function 'permut' not found\n");
            return -1;
        }

        mctx->pipeline = [mctx->device newComputePipelineStateWithFunction:function error:&error];
        if (mctx->pipeline == nil) {
            fprintf(stderr, "FATAL: failed to create Metal pipeline: %s\n",
                    [[error localizedDescription] UTF8String]);
            return -1;
        }

        /* Create hashes buffer (read-only for GPU) */
        mctx->buf_hashes = [mctx->device newBufferWithBytes:cfg->hashes
                                                      length:cfg->hashes_num * 16
                                                     options:MTLResourceStorageModeShared];
        if (mctx->buf_hashes == nil) {
            fprintf(stderr, "FATAL: failed to create Metal hashes buffer\n");
            return -1;
        }

        /* Create hashes_reversed buffer (zeroed, GPU writes matches) */
        mctx->buf_hashes_reversed = [mctx->device newBufferWithLength:cfg->hashes_num * MAX_STR_LENGTH
                                                              options:MTLResourceStorageModeShared];
        if (mctx->buf_hashes_reversed == nil) {
            fprintf(stderr, "FATAL: failed to create Metal hashes_reversed buffer\n");
            return -1;
        }
        memset([mctx->buf_hashes_reversed contents], 0, cfg->hashes_num * MAX_STR_LENGTH);

        return 0;
    }
}

static void *metal_run(void *ctx) {
    @autoreleasepool {
        metal_cruncher_ctx *mctx = ctx;
        mctx->is_running = true;
        mctx->task_time_start = current_micros();

        tasks_buffer *buf;
        while (1) {
            tasks_buffers_get_buffer(mctx->cfg->tasks_buffs, &buf);
            if (buf == NULL) break;

            uint32_t num_tasks = buf->num_tasks;
            uint64_t buf_num_anas = buf->num_anas;

            /* Create task buffer for GPU */
            id<MTLBuffer> buf_tasks = [mctx->device newBufferWithBytes:buf->permut_tasks
                                                                length:num_tasks * sizeof(permut_task)
                                                               options:MTLResourceStorageModeShared];

            /* Re-dispatch loop: keep running until all tasks have i >= n */
            bool has_incomplete = true;
            while (has_incomplete) {
                @autoreleasepool {
                    uint64_t dispatch_start = current_micros();
                    uint64_t dispatch_anas = buf_num_anas;

                    /* Encode and dispatch */
                    id<MTLCommandBuffer> cmdBuf = [mctx->queue commandBuffer];
                    id<MTLComputeCommandEncoder> encoder = [cmdBuf computeCommandEncoder];

                    [encoder setComputePipelineState:mctx->pipeline];
                    [encoder setBuffer:buf_tasks offset:0 atIndex:0];
                    uint32_t iters_per_task = MAX_ITERS_IN_KERNEL_TASK;
                    [encoder setBytes:&iters_per_task length:sizeof(iters_per_task) atIndex:1];
                    [encoder setBuffer:mctx->buf_hashes offset:0 atIndex:2];
                    uint32_t hashes_num = mctx->cfg->hashes_num;
                    [encoder setBytes:&hashes_num length:sizeof(hashes_num) atIndex:3];
                    [encoder setBuffer:mctx->buf_hashes_reversed offset:0 atIndex:4];

                    MTLSize gridSize = MTLSizeMake(num_tasks, 1, 1);
                    NSUInteger threadGroupSize = MIN(mctx->pipeline.maxTotalThreadsPerThreadgroup, (NSUInteger)num_tasks);
                    MTLSize threadgroupSize = MTLSizeMake(threadGroupSize, 1, 1);
                    [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];

                    [encoder endEncoding];
                    [cmdBuf commit];
                    [cmdBuf waitUntilCompleted];

                    uint64_t dispatch_end = current_micros();

                    /* Check if any tasks need re-dispatch */
                    permut_task *tasks_out = (permut_task *)[buf_tasks contents];
                    has_incomplete = false;
                    buf_num_anas = 0;
                    for (uint32_t i = 0; i < num_tasks; i++) {
                        if (tasks_out[i].i < tasks_out[i].n) {
                            has_incomplete = true;
                            uint64_t iters_left = fact(tasks_out[i].n) - tasks_out[i].iters_done;
                            buf_num_anas += iters_left > MAX_ITERS_IN_KERNEL_TASK ? MAX_ITERS_IN_KERNEL_TASK : iters_left;
                        }
                    }

                    /* Record windowed stats for this dispatch (after computing
                     * next-dispatch anas, so dispatch_anas reflects THIS dispatch) */
                    mctx->times_start[mctx->times_idx] = dispatch_start;
                    mctx->times_end[mctx->times_idx] = dispatch_end;
                    mctx->times_anas[mctx->times_idx] = dispatch_anas;
                    mctx->times_idx = (mctx->times_idx + 1) % TIMES_WINDOW_LENGTH;
                } /* @autoreleasepool — drains command buffer/encoder each dispatch */
            }

            mctx->consumed_anas += buf->num_anas;
            mctx->consumed_bufs++;

            /* Merge results to shared buffer */
            merge_hashes_reversed(mctx);

            tasks_buffer_free(buf);
        }

        /* Final merge */
        merge_hashes_reversed(mctx);

        mctx->task_time_end = current_micros();
        mctx->is_running = false;
        return NULL;
    }
}

static void metal_get_stats(void *ctx, float *busy_pct, float *anas_per_sec) {
    metal_cruncher_ctx *mctx = ctx;
    uint64_t calculated_anas = 0;
    uint64_t min_time_start = (uint64_t)-1L, max_time_end = 0;
    uint64_t micros_in_kernel = 0;

    for (int i = 0; i < TIMES_WINDOW_LENGTH; i++) {
        if (mctx->times_start[i] > 0) {
            calculated_anas += mctx->times_anas[i];
            micros_in_kernel += mctx->times_end[i] - mctx->times_start[i];

            if (mctx->times_start[i] < min_time_start) {
                min_time_start = mctx->times_start[i];
            }
            if (mctx->times_end[i] > max_time_end) {
                max_time_end = mctx->times_end[i];
            }
        }
    }

    if (max_time_end <= min_time_start) {
        *busy_pct = 0;
        *anas_per_sec = 0;
        return;
    }

    uint64_t wall_time = max_time_end - min_time_start;
    *busy_pct = (float)micros_in_kernel / (float)wall_time * 100.0f;
    *anas_per_sec = (float)calculated_anas / ((float)wall_time / 1000000.0f);
}

static bool metal_is_running(void *ctx) {
    return ((metal_cruncher_ctx *)ctx)->is_running;
}

static int metal_destroy(void *ctx) {
    metal_cruncher_ctx *mctx = ctx;
    mctx->buf_hashes = nil;
    mctx->buf_hashes_reversed = nil;
    mctx->pipeline = nil;
    mctx->queue = nil;
    mctx->device = nil;
    return 0;
}

cruncher_ops metal_cruncher_ops = {
    .name = "metal",
    .probe = metal_probe,
    .create = metal_create,
    .run = metal_run,
    .get_stats = metal_get_stats,
    .is_running = metal_is_running,
    .destroy = metal_destroy,
    .ctx_size = sizeof(metal_cruncher_ctx),
};
