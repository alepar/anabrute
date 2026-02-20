# Backend Selection Redesign — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Enable runtime AVX-512 detection and implement priority-based cruncher backend selection so the fastest available hardware is always used.

**Architecture:** Compile `avx_cruncher.c` with both `-mavx2 -mavx512f` so both SIMD code paths exist in the binary. Use `__builtin_cpu_supports()` at runtime to choose AVX-512 vs AVX2. Replace the ad-hoc backend selection loop in `main.c` with a priority-ordered probe (Metal/discrete GPU > AVX-512 > iGPU > AVX2 > scalar). Filter POCL CPU-only devices from OpenCL.

**Tech Stack:** C99, GCC/Clang CPUID builtins, pthreads, OpenCL API

---

### Task 1: Enable AVX-512 compilation and runtime detection

**Files:**
- Modify: `CMakeLists.txt:27-29`
- Modify: `avx_cruncher.c:298,426,550,610,654-668,739-761`
- Modify: `avx_cruncher.h`

**Step 1: Update CMake flags**

In `CMakeLists.txt`, change line 28 from:
```
set_source_files_properties(avx_cruncher.c PROPERTIES COMPILE_FLAGS "-mavx2")
```
to:
```
set_source_files_properties(avx_cruncher.c PROPERTIES COMPILE_FLAGS "-mavx2 -mavx512f")
```

**Step 2: Add runtime CPUID detection helpers**

In `avx_cruncher.c`, after the includes (after line 8), add:

```c
#if defined(__x86_64__) || defined(_M_AMD64)
static bool cpu_has_avx512f(void) {
    return __builtin_cpu_supports("avx512f");
}
static bool cpu_has_avx2(void) {
    return __builtin_cpu_supports("avx2");
}
#endif
```

**Step 3: Convert compile-time `#if` to runtime dispatch in `process_task`**

Replace the three `#if`/`#elif`/`#else` blocks in `process_task()` (lines 550-649) with runtime dispatch. The function body becomes:

```c
static void process_task(avx_cruncher_ctx *actx, permut_task *task) {
    if (task->i >= task->n) return;
    cruncher_config *cfg = actx->cfg;

#if defined(__x86_64__) || defined(_M_AMD64)
    if (cpu_has_avx512f()) {
        // [existing AVX-512 code block from lines 551-609, unchanged]
        ...
        return;
    }
    // AVX2 path
    {
        // [existing AVX2 code block from lines 611-640, unchanged]
        ...
        return;
    }
#else
    // Scalar fallback
    do {
        uint32_t key[16];
        int wcs = construct_string(task, key);
        uint32_t hash[4];
        md5_scalar(key, hash);
        check_hashes(cfg, hash, key, wcs);
    } while (heap_next(task));
#endif
}
```

Similarly, remove the `#if defined(__x86_64__) && !defined(__AVX512F__)` guard on `md5_check_avx2` (line 298) — change it to just `#if defined(__x86_64__) || defined(_M_AMD64)`. Remove the `#endif` on line 417.

Remove the `#if defined(__x86_64__) && defined(__AVX512F__)` guard on `md5_check_avx512` (line 426) — change it to just `#if defined(__x86_64__) || defined(_M_AMD64)`. Remove the `#endif` on line 544.

**Step 4: Split probe functions and vtables**

Replace `avx_probe` (lines 654-662) with two probes:

```c
static uint32_t avx512_probe(void) {
#if defined(__x86_64__) || defined(_M_AMD64)
    if (!cpu_has_avx512f()) return 0;
    uint32_t cores = num_cpu_cores();
    printf("  avx512: %d cores, avx-512 available\n", cores);
    return cores;
#else
    return 0;
#endif
}

static uint32_t avx2_probe(void) {
#if defined(__x86_64__) || defined(_M_AMD64)
    if (!cpu_has_avx2()) return 0;
    uint32_t cores = num_cpu_cores();
    printf("  avx2: %d cores, avx2 available\n", cores);
    return cores;
#else
    return 0;
#endif
}
```

Replace the two vtable structs (lines 739-761) with three:

```c
cruncher_ops avx512_cruncher_ops = {
    .name = "avx512",
    .probe = avx512_probe,
    .create = avx_create,
    .run = avx_run,
    .get_stats = avx_get_stats,
    .get_total_anas = avx_get_total_anas,
    .is_running = avx_is_running,
    .destroy = avx_destroy,
    .ctx_size = sizeof(avx_cruncher_ctx),
};

cruncher_ops avx2_cruncher_ops = {
    .name = "avx2",
    .probe = avx2_probe,
    .create = avx_create,
    .run = avx_run,
    .get_stats = avx_get_stats,
    .get_total_anas = avx_get_total_anas,
    .is_running = avx_is_running,
    .destroy = avx_destroy,
    .ctx_size = sizeof(avx_cruncher_ctx),
};

cruncher_ops scalar_cruncher_ops = {
    .name = "cpu",
    .probe = scalar_probe,
    .create = avx_create,
    .run = avx_run,
    .get_stats = avx_get_stats,
    .get_total_anas = avx_get_total_anas,
    .is_running = avx_is_running,
    .destroy = avx_destroy,
    .ctx_size = sizeof(avx_cruncher_ctx),
};
```

**Step 5: Update header**

In `avx_cruncher.h`, replace:
```c
extern cruncher_ops avx_cruncher_ops;
extern cruncher_ops scalar_cruncher_ops;
```
with:
```c
extern cruncher_ops avx512_cruncher_ops;
extern cruncher_ops avx2_cruncher_ops;
extern cruncher_ops scalar_cruncher_ops;
```

**Step 6: Build and run tests**

```bash
cd build && cmake .. && make -j$(nproc)
ctest --output-on-failure
```

Expected: all 4 tests pass. The `test_cruncher` test currently references `&avx_cruncher_ops` — this needs updating in the test too (see Step 7).

**Step 7: Update test_cruncher.c**

In `tests/test_cruncher.c`, change the backends array (lines 228-235) from:
```c
cruncher_ops *backends[] = {
#ifdef __APPLE__
    &metal_cruncher_ops,
#endif
    &opencl_cruncher_ops,
    &avx_cruncher_ops,
    NULL
};
```
to:
```c
cruncher_ops *backends[] = {
#ifdef __APPLE__
    &metal_cruncher_ops,
#endif
    &opencl_cruncher_ops,
    &avx512_cruncher_ops,
    &avx2_cruncher_ops,
    NULL
};
```

**Step 8: Rebuild and verify**

```bash
cd build && make -j$(nproc) && ctest --output-on-failure
```

Expected: all tests pass. On a machine with AVX-512, the `avx512` backend tests run. On one without, `avx512` probe returns 0 and is skipped; `avx2` runs instead.

**Step 9: Commit**

```bash
git add CMakeLists.txt avx_cruncher.c avx_cruncher.h tests/test_cruncher.c
git commit -m "feat: runtime AVX-512 detection, split avx512/avx2 backends"
```

---

### Task 2: Filter POCL CPU devices from OpenCL

**Files:**
- Modify: `opencl_cruncher.c:35-50`

**Step 1: Remove the CL_DEVICE_TYPE_ALL fallback**

In `opencl_cruncher.c`, delete the "fall back to all devices" block (lines 35-50):
```c
    // If no GPUs found, fall back to all devices from all platforms
    if (s_num_devices == 0) {
        ...
    }
```

This removes the path where POCL CPU devices get picked up. If no GPU devices exist, `opencl_probe` returns 0, and the AVX/scalar fallback handles it.

**Step 2: Build and verify**

```bash
cd build && make -j$(nproc) && ctest --output-on-failure
```

Expected: all tests pass. On a system with only POCL (no real GPU), `opencl_probe` returns 0.

**Step 3: Commit**

```bash
git add opencl_cruncher.c
git commit -m "fix: skip POCL CPU-only devices in OpenCL probe"
```

---

### Task 3: Priority-based backend selection in main.c

**Files:**
- Modify: `main.c:2-3,88-169`

**Step 1: Update includes and backend list**

The include already has `avx_cruncher.h`. No change needed there.

Replace the backend list and selection logic (lines 88-163) with:

```c
    // === probe and create crunchers (priority order: best first) ===

    cruncher_ops *all_backends[] = {
#ifdef __APPLE__
        &metal_cruncher_ops,       // P1: Metal (discrete GPU)
#endif
        &opencl_cruncher_ops,      // P1/P3: discrete GPU or iGPU
        &avx512_cruncher_ops,      // P2: AVX-512
        &avx2_cruncher_ops,        // P4: AVX2
        &scalar_cruncher_ops,      // P5: scalar fallback
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

    #define MAX_CRUNCHER_INSTANCES 64
    typedef struct {
        cruncher_ops *ops;
        void *ctx;
        pthread_t thread;
    } cruncher_instance;

    cruncher_instance crunchers[MAX_CRUNCHER_INSTANCES];
    uint32_t num_crunchers = 0;

    printf("Probing cruncher backends:\n");
    bool have_gpu = false;
    bool have_cpu_accel = false;  // AVX-512 or AVX2
    for (int bi = 0; all_backends[bi]; bi++) {
        cruncher_ops *ops = all_backends[bi];

        bool is_gpu = (ops != &avx512_cruncher_ops && ops != &avx2_cruncher_ops && ops != &scalar_cruncher_ops);
        bool is_cpu_accel = (ops == &avx512_cruncher_ops || ops == &avx2_cruncher_ops);

        // Skip OpenCL if native GPU (Metal) already active
        if (have_gpu && ops == &opencl_cruncher_ops) {
            printf("  %s: skipped (native GPU already active)\n", ops->name);
            continue;
        }

        // Skip CPU-bound backends if GPU is available (they'd just contend)
        if (have_gpu && !is_gpu) {
            printf("  %s: skipped (GPU backend active)\n", ops->name);
            continue;
        }

        // Skip lower-priority CPU backends if we already have an accelerated one
        if (have_cpu_accel && (ops == &avx2_cruncher_ops || ops == &scalar_cruncher_ops)) {
            printf("  %s: skipped (faster CPU backend active)\n", ops->name);
            continue;
        }

        // Scalar only as absolute fallback
        if (have_cpu_accel && ops == &scalar_cruncher_ops) {
            printf("  %s: skipped (accelerated backend active)\n", ops->name);
            continue;
        }

        uint32_t count = ops->probe();
        if (!count) continue;

        printf("  %s: %d instance(s)\n", ops->name, count);

        if (is_gpu) have_gpu = true;
        if (is_cpu_accel) have_cpu_accel = true;

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

**Step 2: Update enumerator thread count**

Replace the enum thread count logic (line 169) from:
```c
    uint32_t num_cpu_crunchers = have_gpu ? 8 : 4;
```
to:
```c
    uint32_t total_cores = num_cpu_cores();
    // GPU backends don't compete for CPU — use all cores for enumeration.
    // CPU-bound crunchers (AVX-512, AVX2, scalar) share cores — limit enumerators to 2.
    uint32_t num_cpu_crunchers = have_gpu ? total_cores : 2;
```

**Step 3: Build and run**

```bash
cd build && cmake .. && make -j$(nproc) && ctest --output-on-failure
```

Expected: all tests pass. Run `./anabrute` from the project root and verify the probe output shows the correct backend selected and the correct number of enumeration threads.

**Step 4: Commit**

```bash
git add main.c
git commit -m "feat: priority-based backend selection, dynamic enum thread count"
```

---

### Task 4: Verify end-to-end

**Step 1: Run anabrute from project root**

```bash
cd /home/debian/AleCode/anabrute_cl && ./build/anabrute
```

Verify:
- Probe output shows correct backend (avx512 on Zen 4, avx2 on older x86, opencl on GPU systems)
- Correct number of enum threads (2 for CPU backends, N for GPU)
- Hashes are found (reversed hashes printed)

**Step 2: Run full test suite**

```bash
cd build && ctest --output-on-failure
```

Expected: all 4 tests pass.
