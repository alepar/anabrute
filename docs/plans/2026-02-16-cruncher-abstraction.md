# Pluggable Cruncher Abstraction Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Abstract the hash-crunching consumer behind a vtable interface so OpenCL GPU, AVX CPU, and Metal GPU backends can run simultaneously.

**Architecture:** Each backend implements a `cruncher_ops` vtable (probe/create/run/stats/destroy). The orchestrator in `main.c` probes available backends at startup, creates instances, starts threads, and monitors a shared `hashes_reversed` output buffer. All backends pull from the same `tasks_buffers` queue.

**Tech Stack:** C99, pthreads, OpenCL, AVX2/AVX-512 intrinsics, Metal + Objective-C (macOS only)

**Design doc:** `docs/plans/2026-02-16-cruncher-abstraction-design.md`

**Build:** `mkdir -p cmake-build-debug && cd cmake-build-debug && cmake -DCMAKE_POLICY_VERSION_MINIMUM=3.5 .. && make`

**Tests:** `cd cmake-build-debug && ctest --output-on-failure`

---

## Task 1: Cruncher Interface + OpenCL Backend Refactor

Introduce the `cruncher.h` abstraction and wrap the existing GPU cruncher behind it. After this task, the code works exactly as before — pure refactor, no new functionality.

**Files:**
- Create: `cruncher.h`
- Create: `opencl_cruncher.c`
- Create: `opencl_cruncher.h`
- Modify: `gpu_cruncher.c` — minor changes (accept shared hashes_reversed, final merge in run)
- Modify: `gpu_cruncher.h` — add `cruncher_config` pointer to ctx
- Modify: `kernel_debug.c` — update for gpu_cruncher.h changes
- Modify: `CMakeLists.txt` — add `opencl_cruncher.c` to targets

**Context you need:**
- `gpu_cruncher.h` defines `gpu_cruncher_ctx` with OpenCL-specific fields. It also has `krnl_permut` for kernel dispatch. These remain internal.
- `gpu_cruncher.c:run_gpu_cruncher_thread` is the main consumer loop. It calls `gpu_cruncher_ctx_refresh_hashes_reversed` periodically to read GPU results back to host.
- `main.c:52-82` has OpenCL device enumeration logic that will move to `opencl_probe()`.
- `kernel_debug.c` directly uses `gpu_cruncher_ctx_create`, `run_gpu_cruncher_thread`, etc. It must keep working.

### Step 1: Create `cruncher.h`

```c
// cruncher.h
#ifndef ANABRUTE_CRUNCHER_H
#define ANABRUTE_CRUNCHER_H

#include "common.h"
#include "task_buffers.h"

typedef struct cruncher_config_s {
    tasks_buffers *tasks_buffs;     // shared input queue (all crunchers pull from this)
    uint32_t *hashes;               // target hash values (read-only, 4 uint32s per hash)
    uint32_t hashes_num;            // number of target hashes
    uint32_t *hashes_reversed;      // shared output buffer (hashes_num * MAX_STR_LENGTH bytes)
} cruncher_config;

typedef struct cruncher_ops_s {
    const char *name;               // human-readable: "opencl", "avx", "metal"

    // Query hardware. Returns recommended instance count (0 = unavailable).
    uint32_t (*probe)(void);

    // Initialize one cruncher instance. instance_id is 0..probe()-1.
    // ctx points to a caller-allocated block of ctx_size bytes.
    int (*create)(void *ctx, cruncher_config *cfg, uint32_t instance_id);

    // Thread entry point. Pulls tasks from cfg->tasks_buffs, processes them,
    // writes matches to cfg->hashes_reversed. Returns NULL on completion.
    void *(*run)(void *ctx);

    // Report throughput stats for this instance.
    void (*get_stats)(void *ctx, float *busy_pct, float *anas_per_sec);

    // Check if this instance is still running.
    bool (*is_running)(void *ctx);

    // Release resources. Called after thread join.
    int (*destroy)(void *ctx);

    // Size of the backend-specific context struct.
    size_t ctx_size;
} cruncher_ops;

#endif // ANABRUTE_CRUNCHER_H
```

### Step 2: Modify `gpu_cruncher.h` — add config pointer

Add a `cruncher_config *cfg` field to `gpu_cruncher_ctx`:

```c
// In gpu_cruncher_ctx, add after the existing hashes_reversed field:
    cruncher_config *cfg;           // shared config (set by opencl_cruncher)
```

Also add a `uint32_t *local_hashes_reversed` field for the GPU readback temp buffer (separate from the shared one):

```c
    uint32_t *local_hashes_reversed;  // temp buffer for GPU readback
```

### Step 3: Modify `gpu_cruncher.c` — shared output support

**3a.** In `gpu_cruncher_ctx_create`: allocate `local_hashes_reversed` as the temp readback buffer. Keep `ctx->hashes_reversed` as the pointer to the shared buffer (set by opencl_cruncher):

```c
ctx->local_hashes_reversed = malloc(hashes_num * MAX_STR_LENGTH);
ret_iferr(!ctx->local_hashes_reversed, "failed to malloc local_hashes_reversed");
memset(ctx->local_hashes_reversed, 0, hashes_num * MAX_STR_LENGTH);
```

The `clCreateBuffer` for `mem_hashes_reversed` should use `local_hashes_reversed`:

```c
ctx->mem_hashes_reversed = clCreateBuffer(ctx->cl_ctx, CL_MEM_WRITE_ONLY | CL_MEM_COPY_HOST_PTR,
    hashes_num * MAX_STR_LENGTH, ctx->local_hashes_reversed, &errcode);
```

**3b.** Change `gpu_cruncher_ctx_read_hashes_reversed` to read into `local_hashes_reversed`, then merge non-zero slots to `cfg->hashes_reversed`:

```c
cl_int gpu_cruncher_ctx_read_hashes_reversed(gpu_cruncher_ctx *ctx) {
    cl_int err = clEnqueueReadBuffer(ctx->queue, ctx->mem_hashes_reversed, CL_TRUE, 0,
        ctx->hashes_num * MAX_STR_LENGTH, ctx->local_hashes_reversed, 0, NULL, NULL);
    if (err != CL_SUCCESS) return err;

    // Merge to shared buffer
    if (ctx->cfg) {
        for (uint32_t i = 0; i < ctx->hashes_num; i++) {
            if (ctx->local_hashes_reversed[i * MAX_STR_LENGTH / 4]) {
                memcpy(ctx->cfg->hashes_reversed + i * MAX_STR_LENGTH / 4,
                       ctx->local_hashes_reversed + i * MAX_STR_LENGTH / 4,
                       MAX_STR_LENGTH);
            }
        }
    }
    return CL_SUCCESS;
}
```

**3c.** At the end of `run_gpu_cruncher_thread`, before `ctx->is_running = false`, add a final readback to ensure all results are merged:

```c
    gpu_cruncher_ctx_read_hashes_reversed(ctx);
    ctx->is_running = false;
    return NULL;
```

**3d.** In `gpu_cruncher_ctx_free`, also free `local_hashes_reversed`:

```c
if (ctx->local_hashes_reversed) {
    free(ctx->local_hashes_reversed);
}
```

**3e.** Initialize `cfg` to NULL in `gpu_cruncher_ctx_create` so kernel_debug (which doesn't use cruncher_config) still works:

```c
ctx->cfg = NULL;
```

And keep the existing `ctx->hashes_reversed` allocation as a fallback when `cfg == NULL`:

```c
if (ctx->cfg == NULL) {
    ctx->hashes_reversed = ctx->local_hashes_reversed;  // kernel_debug path
}
```

### Step 4: Create `opencl_cruncher.h`

```c
// opencl_cruncher.h
#ifndef ANABRUTE_OPENCL_CRUNCHER_H
#define ANABRUTE_OPENCL_CRUNCHER_H

#include "cruncher.h"

extern cruncher_ops opencl_cruncher_ops;

#endif // ANABRUTE_OPENCL_CRUNCHER_H
```

### Step 5: Create `opencl_cruncher.c`

This is a thin wrapper that delegates to `gpu_cruncher.h` functions:

```c
// opencl_cruncher.c
#include "opencl_cruncher.h"
#include "gpu_cruncher.h"

// Statics populated by probe()
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
    clGetDeviceIDs(s_platform_id, CL_DEVICE_TYPE_ALL,
                   num_all > MAX_OPENCL_DEVICES ? MAX_OPENCL_DEVICES : num_all,
                   all_ids, &num_all);

    // Prefer GPU devices over CPU
    s_num_devices = 0;
    uint32_t num_gpus = 0;
    for (uint32_t i = 0; i < num_all; i++) {
        cl_device_type dtype;
        clGetDeviceInfo(all_ids[i], CL_DEVICE_TYPE, sizeof(dtype), &dtype, NULL);
        if (dtype > CL_DEVICE_TYPE_CPU) num_gpus++;
    }

    if (num_gpus > 0) {
        for (uint32_t i = 0; i < num_all; i++) {
            cl_device_type dtype;
            clGetDeviceInfo(all_ids[i], CL_DEVICE_TYPE, sizeof(dtype), &dtype, NULL);
            if (dtype > CL_DEVICE_TYPE_CPU)
                s_device_ids[s_num_devices++] = all_ids[i];
        }
    } else {
        for (uint32_t i = 0; i < num_all; i++)
            s_device_ids[s_num_devices++] = all_ids[i];
    }

    // Print device info
    for (uint32_t i = 0; i < s_num_devices; i++) {
        char name[256];
        clGetDeviceInfo(s_device_ids[i], CL_DEVICE_NAME, sizeof(name), name, NULL);
        printf("  opencl[%d]: %s\n", i, name);
    }

    return s_num_devices;
}

static int opencl_create(void *ctx, cruncher_config *cfg, uint32_t instance_id) {
    gpu_cruncher_ctx *gctx = ctx;
    gctx->cfg = cfg;
    gctx->hashes_reversed = cfg->hashes_reversed;
    return gpu_cruncher_ctx_create(gctx, s_platform_id, s_device_ids[instance_id],
                                   cfg->tasks_buffs, cfg->hashes, cfg->hashes_num);
}

static void *opencl_run(void *ctx) {
    return run_gpu_cruncher_thread(ctx);
}

static void opencl_get_stats(void *ctx, float *busy_pct, float *anas_per_sec) {
    gpu_cruncher_get_stats(ctx, busy_pct, anas_per_sec);
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
    .is_running = opencl_is_running,
    .destroy = opencl_destroy,
    .ctx_size = sizeof(gpu_cruncher_ctx),
};
```

### Step 6: Update `CMakeLists.txt`

Add `opencl_cruncher.c` to both `anabrute` and `kernel_debug` source lists:

```cmake
add_executable(anabrute main.c opencl_cruncher.c gpu_cruncher.c hashes.c dict.c permut_types.c seedphrase.c fact.c cpu_cruncher.c os.c task_buffers.c)
add_executable(kernel_debug kernel_debug.c opencl_cruncher.c gpu_cruncher.c hashes.c dict.c permut_types.c seedphrase.c fact.c cpu_cruncher.c os.c task_buffers.c)
```

### Step 7: Build and verify

```bash
cd cmake-build-debug && cmake -DCMAKE_POLICY_VERSION_MINIMUM=3.5 .. && make anabrute kernel_debug
```

Both targets must compile. Run existing tests to verify nothing broke:

```bash
ctest --output-on-failure
```

All 3 existing test suites (hash_parsing, dict_parsing, cpu_enumeration) must still pass.

### Step 8: Commit

```bash
git add cruncher.h opencl_cruncher.h opencl_cruncher.c gpu_cruncher.h gpu_cruncher.c kernel_debug.c CMakeLists.txt
git commit -m "refactor: introduce cruncher vtable, wrap OpenCL behind it"
```

---

## Task 2: Orchestrator (main.c rewrite)

Rewrite main.c to use the cruncher abstraction: probe backends, create instances, start threads with priority, unified monitoring loop.

**Files:**
- Modify: `main.c` — major rewrite of cruncher setup + monitoring sections
- Modify: `os.h` / `os.c` — add thread priority helper

**Context you need:**
- `main.c:52-82` (OpenCL device enumeration) moves to `opencl_probe()` (done in Task 1).
- `main.c:104-109` (gpu_cruncher_ctx creation) becomes generic `ops->create()` calls.
- `main.c:132-136` (GPU thread creation) becomes generic.
- `main.c:142-213` (monitoring loop) currently iterates `gpu_cruncher_ctxs[]` directly. Must change to iterate `cruncher_instance` array.
- `main.c:158-168` (hashes_reversed checking) now reads from shared buffer directly.
- Dict loading (lines 20-42) and CPU cruncher setup (lines 113-117) are UNCHANGED.

### Step 1: Add thread priority helper to `os.h`/`os.c`

```c
// os.h — add:
void set_thread_high_priority(void);

// os.c — add:
#include <sys/resource.h>

void set_thread_high_priority(void) {
#ifdef __APPLE__
    // macOS: use setpriority (lower nice = higher priority)
    setpriority(PRIO_PROCESS, 0, -5);
#else
    // Linux: use nice
    nice(-5);
#endif
}
```

### Step 2: Rewrite main.c cruncher setup

Replace the OpenCL device enumeration (lines 52-82), device info printing (84-96), and gpu_cruncher creation (104-109) with:

```c
// === probe and create crunchers ===
#include "opencl_cruncher.h"
// Future: #include "avx_cruncher.h"
// Future: #include "metal_cruncher.h"

cruncher_ops *all_backends[] = {
    &opencl_cruncher_ops,
    // Future: &avx_cruncher_ops,
    // Future: &metal_cruncher_ops,
    NULL
};

// Shared output buffer
uint32_t *hashes_reversed = calloc(hashes_num, MAX_STR_LENGTH);
ret_iferr(!hashes_reversed, "failed to allocate hashes_reversed");

cruncher_config cruncher_cfg = {
    .tasks_buffs = &tasks_buffs,
    .hashes = hashes,
    .hashes_num = hashes_num,
    .hashes_reversed = hashes_reversed,
};

// Probe all backends
#define MAX_CRUNCHER_INSTANCES 64
typedef struct {
    cruncher_ops *ops;
    void *ctx;
    pthread_t thread;
} cruncher_instance;

cruncher_instance crunchers[MAX_CRUNCHER_INSTANCES];
uint32_t num_crunchers = 0;

printf("Probing cruncher backends:\n");
for (int bi = 0; all_backends[bi]; bi++) {
    cruncher_ops *ops = all_backends[bi];
    uint32_t count = ops->probe();
    if (!count) continue;

    printf("  %s: %d instance(s)\n", ops->name, count);

    // TODO: cap AVX instances to leave cores for dict threads
    // uint32_t cap = (is_cpu_backend) ? num_cores - num_cpu_crunchers - 1 : count;

    for (uint32_t i = 0; i < count && num_crunchers < MAX_CRUNCHER_INSTANCES; i++) {
        cruncher_instance *ci = &crunchers[num_crunchers];
        ci->ops = ops;
        ci->ctx = calloc(1, ops->ctx_size);
        int err = ops->create(ci->ctx, &cruncher_cfg, i);
        ret_iferr(err, "failed to create cruncher instance");
        num_crunchers++;
    }
}
printf("%d cruncher instance(s) total\n\n", num_crunchers);
```

### Step 3: Rewrite thread creation

Replace GPU thread creation (lines 132-136) with:

```c
// Start cruncher threads
for (uint32_t i = 0; i < num_crunchers; i++) {
    int err = pthread_create(&crunchers[i].thread, NULL, crunchers[i].ops->run, crunchers[i].ctx);
    ret_iferr(err, "failed to create cruncher thread");
}
```

For CPU cruncher threads, add priority setting. Modify `run_cpu_cruncher_thread` in `cpu_cruncher.c` to call `set_thread_high_priority()` at the start, or set it from main after creation:

```c
// Start CPU (dict enumeration) threads — higher priority
for (int i = 0; i < num_cpu_crunchers; i++) {
    int err = pthread_create(cpu_threads+i, NULL, run_cpu_cruncher_thread, cpu_cruncher_ctxs+i);
    ret_iferr(err, "failed to create cpu thread");
}
// Note: priority is set inside run_cpu_cruncher_thread via set_thread_high_priority()
```

Add `set_thread_high_priority()` call at the top of `run_cpu_cruncher_thread` in `cpu_cruncher.c`.

### Step 4: Rewrite monitoring loop

Replace the monitoring loop (lines 142-213) to use the cruncher abstraction:

```c
bool hash_is_printed[hashes_num];
memset(hash_is_printed, 0, sizeof(bool) * hashes_num);

while (1) {
    sleep(1);

    // Check if any cruncher still running
    bool any_running = false;
    for (uint32_t i = 0; i < num_crunchers; i++) {
        any_running |= crunchers[i].ops->is_running(crunchers[i].ctx);
    }

    // Print newly found hashes from shared buffer
    for (int hi = 0; hi < hashes_num; hi++) {
        if (!hash_is_printed[hi] && hashes_reversed[hi * MAX_STR_LENGTH / 4]) {
            hash_to_ascii(hashes + hi * 4, strbuf);
            printf("%s:  %s\n", strbuf, (char *)(hashes_reversed + hi * MAX_STR_LENGTH / 4));
            hash_is_printed[hi] = true;
        }
    }

    // Aggregate stats from all crunchers
    float overall_busy = 0, overall_anas_per_sec = 0;
    for (uint32_t i = 0; i < num_crunchers; i++) {
        float busy, aps;
        crunchers[i].ops->get_stats(crunchers[i].ctx, &busy, &aps);
        overall_busy += busy;
        overall_anas_per_sec += aps;
    }
    if (num_crunchers > 0) overall_busy /= num_crunchers;

    // ... (existing progress printing logic, adapted)

    // Poison pill when CPU threads done
    if (min >= dict_by_char_len[0] && max >= dict_by_char_len[0]) {
        tasks_buffers_close(&tasks_buffs);
    }

    if (!any_running) {
        printf("\n");
        break;
    }
}
```

### Step 5: Rewrite cleanup

```c
for (uint32_t i = 0; i < num_cpu_crunchers; i++) {
    pthread_join(cpu_threads[i], NULL);
}
for (uint32_t i = 0; i < num_crunchers; i++) {
    pthread_join(crunchers[i].thread, NULL);
    crunchers[i].ops->destroy(crunchers[i].ctx);
    free(crunchers[i].ctx);
}
free(hashes_reversed);
```

### Step 6: Build and verify

```bash
cd cmake-build-debug && make anabrute kernel_debug && ctest --output-on-failure
```

Both targets must build. Existing tests must pass. If you have OpenCL hardware available, verify `anabrute` runs and produces the same output as before.

### Step 7: Commit

```bash
git add main.c os.h os.c cpu_cruncher.c
git commit -m "refactor: orchestrator uses cruncher vtable for probe/create/run/stats"
```

---

## Task 3: Cruncher Test Suite

Create a generic test that validates cruncher correctness: given known tasks + known target hashes, verify the cruncher finds the right matches.

**Files:**
- Create: `tests/test_cruncher.c`
- Modify: `CMakeLists.txt` — add test target

**Context you need:**
- `kernel_debug.c:16-72` has `create_and_fill_task_buffer()` which manually constructs `permut_task` structs. This is the pattern to follow.
- A `permut_task` encodes: `all_strs` (concatenated null-terminated words), `offsets` (positive = permutable index into `a[]`, negative = fixed position, zero = end), `a[]` (Heap's permutation state — initial word offsets), `n` (number of permutable positions).
- The OpenCL kernel constructs candidate strings by iterating offsets, looking up words in `all_strs`, spacing them, computing MD5, and comparing.
- For testing, we need known plaintext → MD5 pairs. We can compute these offline. E.g., MD5("tyranousplutotwits") can be computed with `echo -n "tyranousplutotwits" | md5` on macOS.
- The cruncher test should work with ANY backend. Use `probe()` to check availability and skip unavailable backends.

### Step 1: Compute known MD5 hashes for test strings

Run these on the command line to get test vectors:

```bash
echo -n "tyranousplutotwits" | md5    # single word, n=1
echo -n "tyranous plutotwits" | md5   # two words order A, n=2
echo -n "plutotwits tyranous" | md5   # two words order B, n=2
```

Record the hex hashes. These will be hardcoded in the test.

### Step 2: Create `tests/test_cruncher.c`

```c
#include <assert.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include "cruncher.h"
#include "opencl_cruncher.h"
#include "hashes.h"
#include "task_buffers.h"
#include "fact.h"
#include "os.h"

/*
 * Helper: construct a tasks_buffer with a single manually-built task.
 * words[] is a NULL-terminated array of strings.
 * Returns a tasks_buffer ready to submit.
 */
static tasks_buffer *make_task_buffer(const char *words[], int num_words) {
    tasks_buffer *buf = tasks_buffer_allocate();

    permut_task *task = buf->permut_tasks;
    memset(task, 0, sizeof(permut_task));

    // Pack words into all_strs
    int8_t off = 0;
    for (int i = 0; i < num_words; i++) {
        task->a[i] = off + 1;  // 1-based offset into all_strs
        int len = strlen(words[i]);
        memcpy(task->all_strs + off, words[i], len + 1);
        off += len + 1;
    }

    // All positions are permutable
    for (int i = 0; i < num_words; i++) {
        task->offsets[i] = i + 1;  // positive = permutable, index into a[]
    }
    task->offsets[num_words] = 0;  // terminator

    task->n = num_words;
    task->i = 0;
    task->iters_done = 0;
    memset(task->c, 0, MAX_OFFSETS_LENGTH);

    buf->num_tasks = 1;
    buf->num_anas = fact(num_words);

    return buf;
}

/*
 * Helper: run a single cruncher backend on given tasks, return hashes_reversed.
 * Submits buf to tasks_buffers, runs cruncher thread, collects results.
 */
static void run_cruncher_on_tasks(cruncher_ops *ops, tasks_buffer *buf,
                                   uint32_t *hashes, uint32_t hashes_num,
                                   uint32_t *hashes_reversed) {
    tasks_buffers tasks_buffs;
    tasks_buffers_create(&tasks_buffs);

    cruncher_config cfg = {
        .tasks_buffs = &tasks_buffs,
        .hashes = hashes,
        .hashes_num = hashes_num,
        .hashes_reversed = hashes_reversed,
    };

    void *ctx = calloc(1, ops->ctx_size);
    assert(ctx);
    int err = ops->create(ctx, &cfg, 0);
    assert(err == 0);

    // Submit task buffer, then close queue
    tasks_buffers_add_buffer(&tasks_buffs, buf);
    tasks_buffers_close(&tasks_buffs);

    // Run cruncher synchronously
    ops->run(ctx);

    ops->destroy(ctx);
    free(ctx);
    tasks_buffers_free(&tasks_buffs);
}

/*
 * Test: single word "tyranousplutotwits" with n=1.
 * Target hash = MD5("tyranousplutotwits").
 * Cruncher should find the match.
 */
static void test_single_word_match(cruncher_ops *ops) {
    // Target: MD5("tyranousplutotwits") — compute with: echo -n "tyranousplutotwits" | md5
    const char *hash_hex = "<FILL_IN_FROM_STEP_1>";
    uint32_t hashes[4];
    ascii_to_hash(hash_hex, hashes);

    uint32_t hashes_reversed[MAX_STR_LENGTH / 4];
    memset(hashes_reversed, 0, MAX_STR_LENGTH);

    const char *words[] = {"tyranousplutotwits", NULL};
    tasks_buffer *buf = make_task_buffer(words, 1);

    run_cruncher_on_tasks(ops, buf, hashes, 1, hashes_reversed);

    // Verify match was found
    assert(hashes_reversed[0] != 0 && "should find single-word match");
    printf("    PASS: single word match\n");
}

/*
 * Test: two words "tyranous" + "plutotwits" with n=2.
 * Target hashes = MD5("tyranous plutotwits") and MD5("plutotwits tyranous").
 * Cruncher should find both.
 */
static void test_two_word_match(cruncher_ops *ops) {
    const char *hash_hexes[] = {
        "<FILL_IN_MD5_OF_tyranous_plutotwits>",
        "<FILL_IN_MD5_OF_plutotwits_tyranous>",
    };
    uint32_t hashes[8];
    ascii_to_hash(hash_hexes[0], hashes);
    ascii_to_hash(hash_hexes[1], hashes + 4);

    uint32_t hashes_reversed[2 * MAX_STR_LENGTH / 4];
    memset(hashes_reversed, 0, 2 * MAX_STR_LENGTH);

    const char *words[] = {"tyranous", "plutotwits", NULL};
    tasks_buffer *buf = make_task_buffer(words, 2);

    run_cruncher_on_tasks(ops, buf, hashes, 2, hashes_reversed);

    assert(hashes_reversed[0] != 0 && "should find first ordering");
    assert(hashes_reversed[MAX_STR_LENGTH / 4] != 0 && "should find second ordering");
    printf("    PASS: two word match\n");
}

/*
 * Test: no matching hash. Should produce zero matches.
 */
static void test_no_match(cruncher_ops *ops) {
    // Use a hash that won't match anything
    const char *hash_hex = "00000000000000000000000000000000";
    uint32_t hashes[4];
    ascii_to_hash(hash_hex, hashes);

    uint32_t hashes_reversed[MAX_STR_LENGTH / 4];
    memset(hashes_reversed, 0, MAX_STR_LENGTH);

    const char *words[] = {"tyranousplutotwits", NULL};
    tasks_buffer *buf = make_task_buffer(words, 1);

    run_cruncher_on_tasks(ops, buf, hashes, 1, hashes_reversed);

    assert(hashes_reversed[0] == 0 && "should NOT find match for zero hash");
    printf("    PASS: no false matches\n");
}

static void run_backend_tests(cruncher_ops *ops) {
    printf("  Testing %s backend:\n", ops->name);
    test_single_word_match(ops);
    test_two_word_match(ops);
    test_no_match(ops);
}

int main(void) {
    printf("test_cruncher:\n");

    cruncher_ops *backends[] = {
        &opencl_cruncher_ops,
        // Future: &avx_cruncher_ops,
        // Future: &metal_cruncher_ops,
        NULL
    };

    int tested = 0;
    for (int i = 0; backends[i]; i++) {
        uint32_t count = backends[i]->probe();
        if (count == 0) {
            printf("  %s: not available, skipping\n", backends[i]->name);
            continue;
        }
        run_backend_tests(backends[i]);
        tested++;
    }

    if (tested == 0) {
        printf("  WARNING: no backends available, all tests skipped\n");
    }

    printf("All cruncher tests passed!\n");
    return 0;
}
```

### Step 3: Add test target to `CMakeLists.txt`

```cmake
add_executable(test_cruncher tests/test_cruncher.c
    opencl_cruncher.c gpu_cruncher.c task_buffers.c hashes.c permut_types.c seedphrase.c fact.c os.c)
set_property(TARGET test_cruncher PROPERTY C_STANDARD 99)
target_include_directories(test_cruncher PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_options(test_cruncher PRIVATE -fsanitize=address -fsanitize=undefined -fno-omit-frame-pointer)
target_link_options(test_cruncher PRIVATE -fsanitize=address -fsanitize=undefined)
target_link_libraries(test_cruncher pthread ${OpenCL_LIBRARY})
add_test(NAME cruncher COMMAND test_cruncher)
```

### Step 4: Build and run

```bash
cd cmake-build-debug && cmake -DCMAKE_POLICY_VERSION_MINIMUM=3.5 .. && make test_cruncher
ctest -R cruncher --output-on-failure
```

### Step 5: Commit

```bash
git add tests/test_cruncher.c CMakeLists.txt
git commit -m "test: generic cruncher correctness test suite"
```

---

## Task 4: AVX Cruncher

Implement a CPU-based cruncher using AVX2 intrinsics for parallel MD5 computation. Each instance is one thread that pulls tasks from the queue, runs Heap's permutation in scalar C, and computes 8 MD5 hashes simultaneously using AVX2.

**Files:**
- Create: `avx_cruncher.h`
- Create: `avx_cruncher.c`
- Create: `md5_avx2.h` — AVX2 vectorized MD5 (header-only for inlining)
- Modify: `cruncher.h` — no changes needed (already generic)
- Modify: `main.c` — add `avx_cruncher_ops` to backend registry
- Modify: `CMakeLists.txt` — add avx sources, `-mavx2` flag
- Modify: `tests/test_cruncher.c` — add avx backend to test list

**Context you need:**
- The OpenCL kernel (`kernels/permut.cl`) is the reference implementation. The AVX cruncher does the same algorithm in C + AVX2.
- The kernel's inner loop: (1) construct candidate string from `offsets[]` + `a[]` + `all_strs`, (2) MD5-pad it, (3) compute MD5, (4) compare against target hashes, (5) advance Heap's permutation.
- AVX2 processes 8 independent MD5 hashes per instruction. So we batch: generate 8 permutations, construct 8 strings, pack into SIMD lanes, compute 8 MD5s, compare.
- For strings < 56 bytes (always true for our 18-char seed phrase), entire message fits in one MD5 block (512 bits = 64 bytes).
- `probe()` uses `__builtin_cpu_supports("avx2")` on x86, returns 0 on ARM.

### Step 1: Create `md5_avx2.h`

AVX2 MD5 computing 8 hashes in parallel. Each of the 4 hash state variables (a, b, c, d) is a `__m256i` holding 8 uint32 values.

```c
// md5_avx2.h — AVX2 vectorized MD5 (8 hashes in parallel)
#ifndef ANABRUTE_MD5_AVX2_H
#define ANABRUTE_MD5_AVX2_H

#ifdef __x86_64__
#include <immintrin.h>

// MD5 round functions operating on 8 lanes
#define MD5_F(x, y, z) _mm256_or_si256(_mm256_and_si256(x, y), _mm256_andnot_si256(x, z))
#define MD5_G(x, y, z) _mm256_or_si256(_mm256_and_si256(z, x), _mm256_andnot_si256(z, y))
#define MD5_H(x, y, z) _mm256_xor_si256(_mm256_xor_si256(x, y), z)
#define MD5_I(x, y, z) _mm256_xor_si256(y, _mm256_or_si256(x, _mm256_xor_si256(z, _mm256_set1_epi32(-1))))

#define MD5_ROTATE_LEFT(x, n) \
    _mm256_or_si256(_mm256_slli_epi32(x, n), _mm256_srli_epi32(x, 32 - (n)))

#define MD5_STEP(f, a, b, c, d, x, t, s) do { \
    a = _mm256_add_epi32(a, _mm256_add_epi32(_mm256_add_epi32(f(b, c, d), x), _mm256_set1_epi32(t))); \
    a = MD5_ROTATE_LEFT(a, s); \
    a = _mm256_add_epi32(a, b); \
} while(0)

/*
 * Compute 8 MD5 hashes in parallel.
 * keys: array of 8 pointers to uint32_t[16] (pre-padded MD5 blocks)
 * out:  array of 8 uint32_t[4] results
 */
static inline void md5_avx2(const uint32_t *keys[8], uint32_t out[8][4]) {
    // Load message words — keys[lane][word_index]
    __m256i w[16];
    for (int i = 0; i < 16; i++) {
        w[i] = _mm256_set_epi32(
            keys[7][i], keys[6][i], keys[5][i], keys[4][i],
            keys[3][i], keys[2][i], keys[1][i], keys[0][i]);
    }

    __m256i a = _mm256_set1_epi32(0x67452301);
    __m256i b = _mm256_set1_epi32(0xefcdab89);
    __m256i c = _mm256_set1_epi32(0x98badcfe);
    __m256i d = _mm256_set1_epi32(0x10325476);

    // Round 1 (F)
    MD5_STEP(MD5_F, a, b, c, d, w[0],  0xd76aa478, 7);
    MD5_STEP(MD5_F, d, a, b, c, w[1],  0xe8c7b756, 12);
    MD5_STEP(MD5_F, c, d, a, b, w[2],  0x242070db, 17);
    MD5_STEP(MD5_F, b, c, d, a, w[3],  0xc1bdceee, 22);
    MD5_STEP(MD5_F, a, b, c, d, w[4],  0xf57c0faf, 7);
    MD5_STEP(MD5_F, d, a, b, c, w[5],  0x4787c62a, 12);
    MD5_STEP(MD5_F, c, d, a, b, w[6],  0xa8304613, 17);
    MD5_STEP(MD5_F, b, c, d, a, w[7],  0xfd469501, 22);
    MD5_STEP(MD5_F, a, b, c, d, w[8],  0x698098d8, 7);
    MD5_STEP(MD5_F, d, a, b, c, w[9],  0x8b44f7af, 12);
    MD5_STEP(MD5_F, c, d, a, b, w[10], 0xffff5bb1, 17);
    MD5_STEP(MD5_F, b, c, d, a, w[11], 0x895cd7be, 22);
    MD5_STEP(MD5_F, a, b, c, d, w[12], 0x6b901122, 7);
    MD5_STEP(MD5_F, d, a, b, c, w[13], 0xfd987193, 12);
    MD5_STEP(MD5_F, c, d, a, b, w[14], 0xa679438e, 17);
    MD5_STEP(MD5_F, b, c, d, a, w[15], 0x49b40821, 22);

    // Round 2 (G)
    MD5_STEP(MD5_G, a, b, c, d, w[1],  0xf61e2562, 5);
    MD5_STEP(MD5_G, d, a, b, c, w[6],  0xc040b340, 9);
    MD5_STEP(MD5_G, c, d, a, b, w[11], 0x265e5a51, 14);
    MD5_STEP(MD5_G, b, c, d, a, w[0],  0xe9b6c7aa, 20);
    MD5_STEP(MD5_G, a, b, c, d, w[5],  0xd62f105d, 5);
    MD5_STEP(MD5_G, d, a, b, c, w[10], 0x02441453, 9);
    MD5_STEP(MD5_G, c, d, a, b, w[15], 0xd8a1e681, 14);
    MD5_STEP(MD5_G, b, c, d, a, w[4],  0xe7d3fbc8, 20);
    MD5_STEP(MD5_G, a, b, c, d, w[9],  0x21e1cde6, 5);
    MD5_STEP(MD5_G, d, a, b, c, w[14], 0xc33707d6, 9);
    MD5_STEP(MD5_G, c, d, a, b, w[3],  0xf4d50d87, 14);
    MD5_STEP(MD5_G, b, c, d, a, w[8],  0x455a14ed, 20);
    MD5_STEP(MD5_G, a, b, c, d, w[13], 0xa9e3e905, 5);
    MD5_STEP(MD5_G, d, a, b, c, w[2],  0xfcefa3f8, 9);
    MD5_STEP(MD5_G, c, d, a, b, w[7],  0x676f02d9, 14);
    MD5_STEP(MD5_G, b, c, d, a, w[12], 0x8d2a4c8a, 20);

    // Round 3 (H)
    MD5_STEP(MD5_H, a, b, c, d, w[5],  0xfffa3942, 4);
    MD5_STEP(MD5_H, d, a, b, c, w[8],  0x8771f681, 11);
    MD5_STEP(MD5_H, c, d, a, b, w[11], 0x6d9d6122, 16);
    MD5_STEP(MD5_H, b, c, d, a, w[14], 0xfde5380c, 23);
    MD5_STEP(MD5_H, a, b, c, d, w[1],  0xa4beea44, 4);
    MD5_STEP(MD5_H, d, a, b, c, w[4],  0x4bdecfa9, 11);
    MD5_STEP(MD5_H, c, d, a, b, w[7],  0xf6bb4b60, 16);
    MD5_STEP(MD5_H, b, c, d, a, w[10], 0xbebfbc70, 23);
    MD5_STEP(MD5_H, a, b, c, d, w[13], 0x289b7ec6, 4);
    MD5_STEP(MD5_H, d, a, b, c, w[0],  0xeaa127fa, 11);
    MD5_STEP(MD5_H, c, d, a, b, w[3],  0xd4ef3085, 16);
    MD5_STEP(MD5_H, b, c, d, a, w[6],  0x04881d05, 23);
    MD5_STEP(MD5_H, a, b, c, d, w[9],  0xd9d4d039, 4);
    MD5_STEP(MD5_H, d, a, b, c, w[12], 0xe6db99e5, 11);
    MD5_STEP(MD5_H, c, d, a, b, w[15], 0x1fa27cf8, 16);
    MD5_STEP(MD5_H, b, c, d, a, w[2],  0xc4ac5665, 23);

    // Round 4 (I)
    MD5_STEP(MD5_I, a, b, c, d, w[0],  0xf4292244, 6);
    MD5_STEP(MD5_I, d, a, b, c, w[7],  0x432aff97, 10);
    MD5_STEP(MD5_I, c, d, a, b, w[14], 0xab9423a7, 15);
    MD5_STEP(MD5_I, b, c, d, a, w[5],  0xfc93a039, 21);
    MD5_STEP(MD5_I, a, b, c, d, w[12], 0x655b59c3, 6);
    MD5_STEP(MD5_I, d, a, b, c, w[3],  0x8f0ccc92, 10);
    MD5_STEP(MD5_I, c, d, a, b, w[10], 0xffeff47d, 15);
    MD5_STEP(MD5_I, b, c, d, a, w[1],  0x85845dd1, 21);
    MD5_STEP(MD5_I, a, b, c, d, w[8],  0x6fa87e4f, 6);
    MD5_STEP(MD5_I, d, a, b, c, w[15], 0xfe2ce6e0, 10);
    MD5_STEP(MD5_I, c, d, a, b, w[6],  0xa3014314, 15);
    MD5_STEP(MD5_I, b, c, d, a, w[13], 0x4e0811a1, 21);
    MD5_STEP(MD5_I, a, b, c, d, w[4],  0xf7537e82, 6);
    MD5_STEP(MD5_I, d, a, b, c, w[11], 0xbd3af235, 10);
    MD5_STEP(MD5_I, c, d, a, b, w[2],  0x2ad7d2bb, 15);
    MD5_STEP(MD5_I, b, c, d, a, w[9],  0xeb86d391, 21);

    // Add initial values
    a = _mm256_add_epi32(a, _mm256_set1_epi32(0x67452301));
    b = _mm256_add_epi32(b, _mm256_set1_epi32(0xefcdab89));
    c = _mm256_add_epi32(c, _mm256_set1_epi32(0x98badcfe));
    d = _mm256_add_epi32(d, _mm256_set1_epi32(0x10325476));

    // Extract results — store as [lane][hash_word]
    uint32_t tmp_a[8], tmp_b[8], tmp_c[8], tmp_d[8];
    _mm256_storeu_si256((__m256i *)tmp_a, a);
    _mm256_storeu_si256((__m256i *)tmp_b, b);
    _mm256_storeu_si256((__m256i *)tmp_c, c);
    _mm256_storeu_si256((__m256i *)tmp_d, d);
    for (int i = 0; i < 8; i++) {
        out[i][0] = tmp_a[i];
        out[i][1] = tmp_b[i];
        out[i][2] = tmp_c[i];
        out[i][3] = tmp_d[i];
    }
}

#endif // __x86_64__
#endif // ANABRUTE_MD5_AVX2_H
```

### Step 2: Create `avx_cruncher.h`

```c
// avx_cruncher.h
#ifndef ANABRUTE_AVX_CRUNCHER_H
#define ANABRUTE_AVX_CRUNCHER_H

#include "cruncher.h"

extern cruncher_ops avx_cruncher_ops;

#endif // ANABRUTE_AVX_CRUNCHER_H
```

### Step 3: Create `avx_cruncher.c`

This implements the full cruncher loop: pull tasks, permute, batch MD5, compare.

```c
// avx_cruncher.c
#include "avx_cruncher.h"
#include "os.h"
#include "fact.h"

#ifdef __x86_64__
#include "md5_avx2.h"
#endif

typedef struct {
    cruncher_config *cfg;
    volatile bool is_running;
    volatile uint64_t consumed_bufs;
    volatile uint64_t consumed_anas;
    uint64_t task_time_start;
    uint64_t task_time_end;
} avx_cruncher_ctx;

// Scalar PUTCHAR matching OpenCL kernel behavior
#define PUTCHAR_SCALAR(buf, index, val) \
    (buf)[(index) >> 2] = ((buf)[(index) >> 2] & ~(0xffU << (((index) & 3) << 3))) + ((val) << (((index) & 3) << 3))

/*
 * Construct candidate string from task + current permutation state.
 * Returns string length (excluding null, before padding).
 * key[] is zeroed and filled as uint32 array (matching kernel PUTCHAR layout).
 */
static int construct_string(permut_task *task, uint32_t *key) {
    memset(key, 0, 64);  // 16 uint32s = 64 bytes
    int wcs = 0;
    for (int io = 0; task->offsets[io]; io++) {
        int8_t off = task->offsets[io];
        if (off < 0) {
            off = -off - 1;
        } else {
            off = task->a[off - 1] - 1;
        }
        while (task->all_strs[off]) {
            PUTCHAR_SCALAR(key, wcs, (uint8_t)task->all_strs[off]);
            wcs++;
            off++;
        }
        PUTCHAR_SCALAR(key, wcs, ' ');
        wcs++;
    }
    wcs--;  // remove trailing space
    // MD5 padding
    PUTCHAR_SCALAR(key, wcs, 0x80);
    PUTCHAR_SCALAR(key, 56, wcs << 3);
    PUTCHAR_SCALAR(key, 57, wcs >> 5);
    return wcs;
}

/*
 * Advance Heap's algorithm by one step. Returns false if no more permutations.
 */
static bool heap_next(permut_task *task) {
    while (task->i < task->n) {
        if (task->c[task->i] < task->i) {
            if (task->i % 2 == 0) {
                uint8_t tmp = task->a[0];
                task->a[0] = task->a[task->i];
                task->a[task->i] = tmp;
            } else {
                uint8_t tmp = task->a[task->c[task->i]];
                task->a[task->c[task->i]] = task->a[task->i];
                task->a[task->i] = tmp;
            }
            task->c[task->i]++;
            task->i = 0;
            return true;
        } else {
            task->c[task->i] = 0;
            task->i++;
        }
    }
    return false;
}

static void process_task(avx_cruncher_ctx *actx, permut_task *task) {
    if (task->i >= task->n) return;  // already completed

    cruncher_config *cfg = actx->cfg;

#ifdef __x86_64__
    // AVX2 path: batch 8 permutations at a time
    uint32_t keys[8][16];
    const uint32_t *key_ptrs[8];
    int batch_count = 0;
    int wcs_vals[8];

    for (int i = 0; i < 8; i++) key_ptrs[i] = keys[i];

    // Process first permutation (current state)
    wcs_vals[batch_count] = construct_string(task, keys[batch_count]);
    batch_count++;

    while (heap_next(task)) {
        wcs_vals[batch_count] = construct_string(task, keys[batch_count]);
        batch_count++;

        if (batch_count == 8) {
            // Compute 8 MD5s in parallel
            uint32_t hashes[8][4];
            md5_avx2(key_ptrs, hashes);

            // Compare each against targets
            for (int lane = 0; lane < 8; lane++) {
                for (uint32_t ih = 0; ih < cfg->hashes_num; ih++) {
                    if (hashes[lane][0] == cfg->hashes[4*ih] &&
                        hashes[lane][1] == cfg->hashes[4*ih+1] &&
                        hashes[lane][2] == cfg->hashes[4*ih+2] &&
                        hashes[lane][3] == cfg->hashes[4*ih+3]) {
                        // Match! Write string to shared buffer
                        PUTCHAR_SCALAR(keys[lane], wcs_vals[lane], 0);
                        memcpy(cfg->hashes_reversed + ih * MAX_STR_LENGTH / 4,
                               keys[lane], MAX_STR_LENGTH);
                    }
                }
            }
            batch_count = 0;
        }
    }

    // Process remaining batch (< 8)
    if (batch_count > 0) {
        // Pad unused lanes with lane 0 data (to avoid uninitialized reads)
        for (int i = batch_count; i < 8; i++) {
            memcpy(keys[i], keys[0], 64);
        }
        uint32_t hashes[8][4];
        md5_avx2(key_ptrs, hashes);
        for (int lane = 0; lane < batch_count; lane++) {
            for (uint32_t ih = 0; ih < cfg->hashes_num; ih++) {
                if (hashes[lane][0] == cfg->hashes[4*ih] &&
                    hashes[lane][1] == cfg->hashes[4*ih+1] &&
                    hashes[lane][2] == cfg->hashes[4*ih+2] &&
                    hashes[lane][3] == cfg->hashes[4*ih+3]) {
                    PUTCHAR_SCALAR(keys[lane], wcs_vals[lane], 0);
                    memcpy(cfg->hashes_reversed + ih * MAX_STR_LENGTH / 4,
                           keys[lane], MAX_STR_LENGTH);
                }
            }
        }
    }
#else
    // Scalar fallback (ARM or no AVX)
    // Same algorithm but one hash at a time — use the scalar MD5 from hashes.c
    // or inline the md5() function from the OpenCL kernel
    // (This path is for correctness, not performance)
#endif
}

static uint32_t avx_probe(void) {
#ifdef __x86_64__
    if (__builtin_cpu_supports("avx2")) {
        uint32_t cores = num_cpu_cores();
        printf("  avx: AVX2 detected, suggesting %d threads\n", cores > 1 ? cores - 1 : 1);
        return cores > 1 ? cores - 1 : 1;
    }
#endif
    return 0;
}

static int avx_create(void *ctx, cruncher_config *cfg, uint32_t instance_id) {
    avx_cruncher_ctx *actx = ctx;
    actx->cfg = cfg;
    actx->is_running = false;
    actx->consumed_bufs = 0;
    actx->consumed_anas = 0;
    return 0;
}

static void *avx_run(void *ctx) {
    avx_cruncher_ctx *actx = ctx;
    actx->is_running = true;
    actx->task_time_start = current_micros();

    tasks_buffer *buf;
    while (1) {
        tasks_buffers_get_buffer(actx->cfg->tasks_buffs, &buf);
        if (buf == NULL) break;  // queue closed

        for (uint32_t i = 0; i < buf->num_tasks; i++) {
            process_task(actx, &buf->permut_tasks[i]);
        }

        actx->consumed_anas += buf->num_anas;
        actx->consumed_bufs++;
        tasks_buffer_free(buf);
    }

    actx->task_time_end = current_micros();
    actx->is_running = false;
    return NULL;
}

static void avx_get_stats(void *ctx, float *busy_pct, float *anas_per_sec) {
    avx_cruncher_ctx *actx = ctx;
    uint64_t elapsed = current_micros() - actx->task_time_start;
    if (elapsed == 0) elapsed = 1;
    *busy_pct = 100.0f;  // CPU is always "busy" while running
    *anas_per_sec = (float)actx->consumed_anas / (elapsed / 1000000.0f);
}

static bool avx_is_running(void *ctx) {
    return ((avx_cruncher_ctx *)ctx)->is_running;
}

static int avx_destroy(void *ctx) {
    return 0;  // nothing to free
}

cruncher_ops avx_cruncher_ops = {
    .name = "avx",
    .probe = avx_probe,
    .create = avx_create,
    .run = avx_run,
    .get_stats = avx_get_stats,
    .is_running = avx_is_running,
    .destroy = avx_destroy,
    .ctx_size = sizeof(avx_cruncher_ctx),
};
```

### Step 4: Update `main.c` backend registry

```c
#include "avx_cruncher.h"

cruncher_ops *all_backends[] = {
    &opencl_cruncher_ops,
    &avx_cruncher_ops,
    NULL
};
```

### Step 5: Update `CMakeLists.txt`

Add avx sources to anabrute and test targets. Add `-mavx2` flag for avx files:

```cmake
# Add to anabrute sources
add_executable(anabrute main.c opencl_cruncher.c gpu_cruncher.c avx_cruncher.c
    hashes.c dict.c permut_types.c seedphrase.c fact.c cpu_cruncher.c os.c task_buffers.c)

# AVX compile flags (only for avx_cruncher.c)
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
    set_source_files_properties(avx_cruncher.c PROPERTIES COMPILE_FLAGS "-mavx2")
endif()

# Update test_cruncher to include avx
add_executable(test_cruncher tests/test_cruncher.c
    opencl_cruncher.c gpu_cruncher.c avx_cruncher.c
    task_buffers.c hashes.c permut_types.c seedphrase.c fact.c os.c)
```

### Step 6: Build and test

```bash
cd cmake-build-debug && cmake -DCMAKE_POLICY_VERSION_MINIMUM=3.5 .. && make test_cruncher
ctest -R cruncher --output-on-failure
```

Both OpenCL and AVX backends should pass all tests. If running on ARM (Apple Silicon), AVX tests will be skipped.

### Step 7: Commit

```bash
git add avx_cruncher.h avx_cruncher.c md5_avx2.h main.c CMakeLists.txt tests/test_cruncher.c
git commit -m "feat: AVX2 cruncher backend with vectorized MD5"
```

---

## Task 5: Metal Cruncher (macOS only)

Port the OpenCL kernel to Metal Compute Shaders. The kernel logic (Heap's algorithm + MD5) is identical; only the host API and shader language differ.

**Files:**
- Create: `metal_cruncher.h`
- Create: `metal_cruncher.m` (Objective-C for Metal API)
- Create: `kernels/permut.metal` (Metal Shading Language kernel)
- Modify: `main.c` — add metal backend to registry
- Modify: `CMakeLists.txt` — add Metal framework, handle `.m` files
- Modify: `tests/test_cruncher.c` — add metal backend to test list

**Context you need:**
- `kernels/permut.cl` is the reference. Metal Shading Language (MSL) is C++-based, very close to OpenCL C.
- Key syntax differences: `__kernel void` → `kernel void`, `__global` → `device`, `get_global_id(0)` → pass `uint id [[thread_position_in_grid]]` as parameter, `uchar` → `uint8_t` (or keep uchar via Metal stdlib).
- Metal host API is Objective-C: `MTLCreateSystemDefaultDevice()`, `[device newCommandQueue]`, `[device newLibraryWithSource:...]`, etc.
- Metal supports runtime shader compilation from source (like OpenCL), which matches our current pattern.
- On Apple Silicon, Metal buffers use shared memory (zero-copy): `[device newBufferWithBytesNoCopy:...]` or `MTLResourceStorageModeShared`.

### Step 1: Create `kernels/permut.metal`

Translation of `kernels/permut.cl` to MSL. Key changes:
- Replace `__kernel void permut(__global ...)` with `kernel void permut(device ... [[buffer(N)]], uint id [[thread_position_in_grid]])`
- Replace `get_global_id(0)` with the `id` parameter
- Replace `__global` with `device`
- Keep all MD5 macros, `PUTCHAR`/`GETCHAR`, and algorithm logic identical
- MSL uses `#include <metal_stdlib>` and `using namespace metal;`

The full kernel is ~280 lines (same as the OpenCL version with syntax changes). Provide complete MSL source in this file.

### Step 2: Create `metal_cruncher.h`

```c
// metal_cruncher.h
#ifndef ANABRUTE_METAL_CRUNCHER_H
#define ANABRUTE_METAL_CRUNCHER_H

#include "cruncher.h"

extern cruncher_ops metal_cruncher_ops;

#endif
```

### Step 3: Create `metal_cruncher.m`

Objective-C implementation using Metal API. Structure mirrors `opencl_cruncher.c`:

- `metal_probe()`: calls `MTLCreateSystemDefaultDevice()`, returns 1 if available
- `metal_create()`: compiles shader from `kernels/permut.metal`, creates command queue, allocates buffers with `MTLResourceStorageModeShared`
- `metal_run()`: same dispatch loop as GPU cruncher — pull tasks, encode compute command, dispatch threads, wait, read results, merge to shared buffer
- Key advantage: shared memory mode means no explicit upload/download. Buffers are accessible from both CPU and GPU.

The context struct:

```objc
typedef struct {
    cruncher_config *cfg;
    id<MTLDevice> device;
    id<MTLCommandQueue> queue;
    id<MTLComputePipelineState> pipeline;
    id<MTLBuffer> buf_hashes;
    id<MTLBuffer> buf_hashes_reversed;
    // ... same stats fields as gpu_cruncher_ctx
    volatile bool is_running;
    volatile uint64_t consumed_bufs;
    volatile uint64_t consumed_anas;
} metal_cruncher_ctx;
```

The dispatch loop (inside `metal_run`):

```objc
while (1) {
    tasks_buffer *src_buf;
    tasks_buffers_get_buffer(cfg->tasks_buffs, &src_buf);
    if (!src_buf) break;

    // Create buffer with task data (shared memory — zero copy)
    id<MTLBuffer> buf_tasks = [device newBufferWithBytes:src_buf->permut_tasks
        length:src_buf->num_tasks * sizeof(permut_task)
        options:MTLResourceStorageModeShared];

    id<MTLCommandBuffer> cmdBuf = [queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [cmdBuf computeCommandEncoder];
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:buf_tasks offset:0 atIndex:0];
    // ... set other buffers and args
    [encoder dispatchThreads:MTLSizeMake(src_buf->num_tasks, 1, 1)
       threadsPerThreadgroup:MTLSizeMake(MIN(pipeline.maxTotalThreadsPerThreadgroup, src_buf->num_tasks), 1, 1)];
    [encoder endEncoding];
    [cmdBuf commit];
    [cmdBuf waitUntilCompleted];

    // Results already in shared memory — merge to hashes_reversed
    // ... same merge logic as OpenCL backend
}
```

### Step 4: Update `CMakeLists.txt`

```cmake
if(APPLE)
    enable_language(OBJC)
    add_executable(anabrute main.c opencl_cruncher.c gpu_cruncher.c avx_cruncher.c metal_cruncher.m
        hashes.c dict.c permut_types.c seedphrase.c fact.c cpu_cruncher.c os.c task_buffers.c)
    target_link_libraries(anabrute "-framework Metal" "-framework Foundation")
endif()
```

### Step 5: Update `main.c` backend registry

```c
#ifdef __APPLE__
#include "metal_cruncher.h"
#endif

cruncher_ops *all_backends[] = {
#ifdef __APPLE__
    &metal_cruncher_ops,
#endif
    &opencl_cruncher_ops,
    &avx_cruncher_ops,
    NULL
};
```

### Step 6: Build and test

```bash
cd cmake-build-debug && cmake -DCMAKE_POLICY_VERSION_MINIMUM=3.5 .. && make test_cruncher
ctest -R cruncher --output-on-failure
```

On macOS: all three backends (Metal, OpenCL, AVX) should pass. On Linux: Metal skipped.

### Step 7: Commit

```bash
git add metal_cruncher.h metal_cruncher.m kernels/permut.metal main.c CMakeLists.txt tests/test_cruncher.c
git commit -m "feat: Metal cruncher backend for Apple Silicon"
```
