# Pluggable Cruncher Abstraction Design

## Goal

Abstract the hash-crunching consumer role behind a pluggable interface so multiple backend types (OpenCL GPU, AVX CPU, Metal GPU) can run simultaneously, pulling jobs from the same task queue.

## Backends

| Backend | Hardware | Scaling | Estimated throughput |
|---------|----------|---------|---------------------|
| OpenCL GPU | Discrete/integrated GPU | 1 instance per `cl_device_id` | ~1B hashes/sec (current) |
| AVX CPU | x86 with AVX2/AVX-512 | 1 instance per available core | 8-12B (AVX2 8-core), 25-40B (AVX-512 16-core) |
| Metal GPU | Apple Silicon | 1 instance per `MTLDevice` | M1: 3-5B, M2 Ultra: 15-25B |

Apple Neural Engine was evaluated and rejected — it's a MAC accelerator for INT8/FP16 tensors with no support for bitwise ops, bit rotations, or 32-bit integer arithmetic needed by MD5.

## Cruncher Interface (vtable)

New file `cruncher.h`:

```c
typedef struct cruncher_config_s {
    tasks_buffers *tasks_buffs;    // shared input queue
    uint32_t *hashes;              // target hashes (read-only)
    uint32_t hashes_num;
    uint32_t *hashes_reversed;     // shared output buffer
    pthread_mutex_t *output_mutex; // for hashes_reversed writes
} cruncher_config;

typedef struct cruncher_ops_s {
    const char *name;              // "opencl", "avx", "metal"

    // Returns 0 if unavailable, else recommended instance count
    uint32_t (*probe)(void);

    // Lifecycle per instance
    int   (*create)(void *ctx, cruncher_config *cfg, uint32_t instance_id);
    void* (*run)(void *ctx);       // thread entry point
    void  (*get_stats)(void *ctx, float *busy_pct, float *anas_per_sec);
    int   (*destroy)(void *ctx);

    size_t ctx_size;               // sizeof(backend-specific context)
} cruncher_ops;
```

Backend suggests instance count via `probe()`, orchestrator caps it (GPU/Metal uncapped since they don't consume CPU; AVX capped to leave cores for dict enumeration).

## Orchestrator (main.c)

Startup sequence:
1. Probe all registered backends
2. GPU/Metal get full ask (they don't consume CPU meaningfully)
3. AVX gets `min(probe_result, num_cores - num_dict_threads - 1)`
4. Allocate contexts via `malloc(ops->ctx_size)`, call `create(ctx, cfg, instance_id)`
5. Start all threads — dict threads at higher OS priority via `setpriority()`

Backend registry is compile-time conditional:
```c
cruncher_ops *all_backends[] = {
#ifdef __APPLE__
    &metal_cruncher_ops,
#endif
    &opencl_cruncher_ops,
    &avx_cruncher_ops,
    NULL
};
```

Shutdown: existing `tasks_buffers_close()` poison-pill mechanism. All cruncher threads see empty queue + closed flag and exit.

Stats loop iterates all instances uniformly via `get_stats()`.

## Shared Output Buffer

`hashes_reversed` is allocated once by the orchestrator and shared across all cruncher instances via `cruncher_config`. Each hash slot is written by at most one cruncher (first to find a match). Writes are idempotent — protected by `output_mutex` or lock-free (each slot is independent, written atomically).

## CPU Scheduling

- Dictionary enumeration threads and AVX cruncher threads together fill all CPU cores
- Dictionary threads run at higher OS priority (`setpriority` or `pthread_setschedparam`)
- The task queue (`tasks_buffers`) provides natural backpressure — when empty, consumers block; when full, producers block
- GPU/Metal threads don't compete for CPU (they sleep on OpenCL/Metal events)

## Backend Details

### OpenCL (`opencl_cruncher.c`)

Refactor of existing `gpu_cruncher.c`. `probe()` enumerates OpenCL platforms and devices, prefers GPU over CPU device types. Kernel logic, dispatch loop, and carryover mechanism unchanged.

### AVX (`avx_cruncher.c`)

Each instance is one thread that pulls tasks from the queue, runs Heap's permutation in C, computes MD5 using AVX2 intrinsics (8 hashes in parallel per instruction, 16 with AVX-512). `probe()` checks `__builtin_cpu_supports("avx2")` at runtime.

Key simplification vs GPU: no upload/download cycle, no carryover. CPU reads `permut_task` directly from memory and iterates all `fact(n)` permutations in one go.

### Metal (`metal_cruncher.c`)

Port of OpenCL kernel to Metal Compute Shaders (MSL). `probe()` calls `MTLCreateSystemDefaultDevice()`. Main advantage: unified memory = zero-copy buffers.

Requires Objective-C (`.m` files) for host API. Kernel is pure MSL.

## Test Suite

Generic test harness `tests/test_cruncher.c` that runs identical inputs through each available backend and verifies identical results.

Test cases:
- Single word task, hash matches → match found
- Two-word task, both orderings' hashes present → matches found
- Non-matching hash → no false positives
- Multiple tasks in buffer → only correct matches

Backends unavailable at runtime are skipped. Uses existing `test_cpu_enumeration` pattern for task construction.

## Implementation Order

1. **Cruncher abstraction** — `cruncher.h`, refactor GPU to `opencl_cruncher.c` behind vtable. Pure refactor, no new functionality.
2. **Orchestrator** — Probe/cap/create loop in main.c, shared `hashes_reversed`, thread priority.
3. **Cruncher test suite** — `test_cruncher.c` using OpenCL as first testee.
4. **AVX cruncher** — `avx_cruncher.c` with vectorized MD5. Highest-value new backend.
5. **Metal cruncher** — `metal_cruncher.c` with MSL kernel port. Apple Silicon only.

Steps 4 and 5 are independent.

## Out of Scope (YAGNI)

- Batch-by-N GPU optimization (separate IDEAS.md item)
- Buffer pool / free-list optimization
- SIMD `char_counts_subtract`
- Dynamic thread count tuning (static cap at startup is sufficient)
