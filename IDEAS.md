# IDEAS.md — Performance Optimization Design Notes

## Current Performance Bottleneck Analysis

Measured: ~1B hashes/sec. Theoretical GPU peak for MD5: 10-50B hashes/sec (depending on GPU).

The gap comes from GPU underutilization, not memory bandwidth. Transfer overhead is ~3% of dispatch time. The real problems are:

### 1. Warp Divergence from Mixed Task Sizes

Tasks with different `n` values (number of permutable words) land in the same warp. A task with `n=2` (2 permutations) finishes almost instantly but waits for `n=5` (120 permutations) in the same warp. GPU pays for the slowest work item per warp (32 threads on NVIDIA, 64 on AMD).

### 2. Excessive Kernel Re-dispatch

`MAX_ITERS_IN_KERNEL_TASK=512` means a task with `n=8` (40320 permutations) needs ~79 kernel dispatches to complete. Each dispatch cycle: read back 24MB task state -> re-upload 24MB -> kernel launch -> wait. That's ~630ms of pure transfer overhead for work that should run continuously.

### 3. String Construction Divergence (minor)

The `PUTCHAR` loop in the kernel (lines 196-212 of `permut.cl`) has variable iterations depending on word lengths. Less impactful than the `n`-value divergence but contributes.

### 4. Task Struct is Bandwidth-Efficient

The `permut_task` struct (96 bytes) is a compressed representation of up to `fact(n)` candidate strings. For `n=8`, that's 40320 candidates from 96 bytes — 12,600x compression vs sending raw strings. This is a strength of the current design, not a weakness.

---

## Quick Win: Raise `MAX_ITERS_IN_KERNEL_TASK`

The current value of 512 is extremely conservative. Here's the timeout math:

```
Current (512 iters):    256K tasks × 512 × ~500 cycles/MD5 / (1.5 GHz × 2048 cores) ≈ 21ms
At 8192:                256K tasks × 8192 × ~500 cycles / (1.5 GHz × 2048)           ≈ 341ms
At 40320 (=8!):         256K tasks × 40320 × ~500 cycles / (1.5 GHz × 2048)          ≈ 1.7s
```

GPU watchdog timeouts: ~2s on Windows (TDR), configurable on Linux. In practice most tasks have n=3-5 and exit early, so real kernel time is much lower than the worst case.

**Recommendation:** Set to `8192` or `16384` for a safe ~16-32x reduction in re-dispatch overhead. This is a one-line change in `common.h`:

```c
#define MAX_ITERS_IN_KERNEL_TASK (8*1024)  // was 512
```

**Impact for n=8 tasks (worst case):**

| Cap | Dispatches needed | Overhead |
|-----|-------------------|----------|
| 512 | 79 | 79 × (readback + upload + launch) |
| 8192 | 5 | 5 × (readback + upload + launch) |
| 16384 | 3 | 3 × (readback + upload + launch) |
| 40320 | 1 | zero re-dispatch |

This doesn't fix warp divergence (mixed-n tasks still waste threads) but eliminates most re-dispatch overhead immediately. Combine with batch-by-N below for the full win.

---

## Proposed: Batch-by-N GPU Optimization

**Core idea:** Sort tasks by their `n` value and dispatch separate kernels per batch. Set `iters_per_task = fact(n)` for each batch so every task completes in one dispatch.

### Changes Required

#### A. CPU Side — Task Batching in `cpu_cruncher.c` / `task_buffers.c`

Currently `tasks_buffer` is a flat array of mixed-`n` tasks. Change to:

1. **Per-N local buffers in CPU cruncher.** Instead of one `local_buffer`, maintain `local_buffers[MAX_WORD_LENGTH+1]` (one per possible `n` value, since `n` = word count = up to `MAX_WORD_LENGTH=5`).

2. **`tasks_buffer_add_task`** routes each task to the buffer matching its `n` value.

3. **`tasks_buffers` queue** becomes N separate queues (one per `n` value), or tasks_buffer gets a `uint8_t n_value` field so the GPU consumer can pick intelligently.

Alternative simpler approach: tag each `tasks_buffer` with its `n` value (all tasks in a buffer have the same `n`). GPU consumer groups by `n` when filling kernel dispatch batches.

#### B. GPU Side — Per-Batch Iteration Limit in `gpu_cruncher.c`

Currently `MAX_ITERS_IN_KERNEL_TASK` is a fixed constant. Change to:

1. When dispatching a batch of `n=K` tasks, set `iters_per_task = fact(K)`.
2. Since all tasks in the batch have the same `n`, they all complete in exactly `fact(K)` iterations.
3. **No read-back of task state needed.** Tasks always complete in one dispatch. Only read `hashes_reversed` (tiny).
4. Removes the carryover logic in `run_gpu_cruncher_thread` (lines 183-211 of `gpu_cruncher.c`).

#### C. Kernel — Minor Simplification in `kernels/permut.cl`

1. The write-back of task state (lines 273-275) becomes unnecessary since tasks always complete.
2. Could add `__attribute__((reqd_work_group_size(X, 1, 1)))` hint since we know all items do equal work.
3. The `if (task.i >= task.n)` early-exit check (line 183) stays as a safety guard.

#### D. Double Buffering (Optional Enhancement)

Use two GPU task buffers. While kernel N executes, async-upload batch N+1 via `clEnqueueWriteBuffer` with `CL_FALSE` (non-blocking). Hides the remaining ~2ms upload latency.

### Expected Performance Gains

| Change | Mechanism | Estimated Gain |
|--------|-----------|---------------|
| Batch by N | Eliminates warp divergence — all work items do equal iterations | 2-3x |
| fact(N) iterations | Eliminates re-dispatch overhead (79 dispatches -> 1) | 1.5-2x |
| No read-back | Halves PCIe transfer (but transfer was only 3%) | 1.05x |
| Double buffer | Hides upload latency behind compute | 1.1x |
| **Combined** | | **~4-8x → 4-8B hashes/sec** |

### Risk Assessment

- **Low risk:** The kernel logic doesn't change. We're only changing how tasks are grouped and what `iters_per_task` is set to.
- **Kernel timeout risk:** For `n=8`, `fact(8)=40320` iterations per task. With 256K tasks that's ~10B iterations per dispatch. If the GPU driver has a timeout (common on Windows, ~2 seconds), this might trigger it. Mitigation: cap `iters_per_task` at a safe maximum (e.g. 8192) and allow partial completion for large `n`, but only for those batches.
- **Load balancing:** With separate queues per `n`, smaller-`n` queues might drain faster. The GPU consumer should prioritize whichever queue has work, not round-robin.

### Implementation Sequence

1. Add `n_value` tag to `tasks_buffer` struct
2. Split `local_buffer` into per-N buffers in `cpu_cruncher.c`
3. Route tasks to correct buffer in `tasks_buffer_add_task`
4. In `gpu_cruncher.c`, compute `iters_per_task = fact(n_value)` per batch
5. Remove carryover/re-dispatch logic from `run_gpu_cruncher_thread`
6. Remove task state write-back from kernel (or leave as dead code, harmless)
7. Add double buffering (separate PR, clean diff)

---

## GPU Kernel Optimizations

### GPU-1. Target Hashes in Local Memory (MEDIUM IMPACT)

**Problem:** Every work item reads `hashes[]` from global memory for every permutation. There's a TODO at `permut.cl:222`. All work items in a work group check the same targets, so this is redundant global reads.

**Fix:** Copy target hashes to `__local` memory once per work group:

```c
__local uint local_hashes[MAX_HASHES * 4];  // MAX_HASHES = max expected target count
uint lid = get_local_id(0);
uint lsize = get_local_size(0);
for (uint i = lid; i < hashes_num * 4; i += lsize)
    local_hashes[i] = hashes[i];
barrier(CLK_LOCAL_MEM_FENCE);
// ... use local_hashes instead of hashes in comparison loop
```

With a typical work group of 64-256 threads, this turns N×64 global reads per permutation into 1 local read. Local memory bandwidth is ~10-50x higher than global.

### GPU-2. Reuse Persistent GPU Buffers (MEDIUM IMPACT)

**Problem:** `krnl_permut_create` calls `clCreateBuffer(..., CL_MEM_COPY_HOST_PTR, ...)` every dispatch, and `krnl_permut_free` releases it. Buffer allocation involves driver-side bookkeeping and possibly page table manipulation.

**Fix:** Allocate `cl_mem mem_permut_tasks` once at `gpu_cruncher_ctx_create` time (at max size = `PERMUT_TASKS_IN_KERNEL_TASK * sizeof(permut_task)`). Use `clEnqueueWriteBuffer` to upload new data each dispatch. Similarly, keep the `cl_kernel` object alive and just re-set args when buffer contents change.

Changes to `gpu_cruncher.c`:
- Move `clCreateBuffer` and `clCreateKernel` to `gpu_cruncher_ctx_create`
- `krnl_permut_create` becomes just `clEnqueueWriteBuffer` + `clSetKernelArg`
- `krnl_permut_free` becomes a no-op (or just `clReleaseEvent`)
- `gpu_cruncher_ctx_free` releases the persistent objects

### GPU-3. Early-Exit Hash Comparison (LOW IMPACT)

**Problem:** The inner hash comparison loop already breaks on first mismatch, but the compiler may not optimize the branch pattern for the common case (non-match on first uint32).

**Fix:** Unroll the comparison to make the fast path explicit:

```c
for (uchar ih = 0; ih < hashes_num; ih++) {
    if (local_hashes[4*ih] != computed_hash[0]) continue;  // eliminates 15/16 of non-matches
    if (local_hashes[4*ih+1] != computed_hash[1]) continue;
    if (local_hashes[4*ih+2] != computed_hash[2]) continue;
    if (local_hashes[4*ih+3] != computed_hash[3]) continue;
    // match — write to hashes_reversed
}
```

---

## CPU-Side Optimizations

The CPU pipeline has 3 nested recursion levels before a single task is submitted to the GPU:

1. `recurse_dict_words` — find word combinations whose char_counts sum to seed phrase
2. `recurse_string_combs` — distribute word counts across synonym strings (same char_counts, different strings like "list"/"slit")
3. `recurse_combs` — assign words to positional slots in the permutation array

### CPU-1. Buffer Pool — Eliminate 24MB calloc/free Churn (HIGH IMPACT)

**Problem:** Every time a CPU thread fills its 256K-task buffer, it submits it and calls `tasks_buffer_allocate()` which does `calloc(256K, 96) = 24MB`. The GPU consumer then `free()`s consumed buffers. This is constant 24MB malloc/free churn hitting the kernel page allocator, plus zero-filling 24MB of memory.

**Fix:** Create a free-list of pre-allocated buffers. GPU consumer returns emptied buffers to the free-list (just resets `num_tasks=0`). CPU producer pulls from free-list instead of calloc. After warmup, zero allocations.

```
                 tasks_buffers (ready queue)
CPU producer ──────────────────────────────> GPU consumer
     ^                                           │
     │          free_buffers (free-list)          │
     └───────────────────────────────────────────┘
```

Changes:
- Add `tasks_buffers_return_buffer(buffs, buf)` — puts buffer on free-list
- Add `tasks_buffers_get_free_buffer(buffs)` — returns a buffer from free-list, or allocates if none available
- GPU consumer calls `return_buffer` instead of `free`
- CPU producer calls `get_free_buffer` instead of `allocate`

### CPU-2. SIMD `char_counts_subtract` (MEDIUM IMPACT)

**Problem:** `char_counts_subtract` is called in the hottest loop of the search — the inner for-loop of `recurse_dict_words` (line 200 of `cpu_cruncher.c`). It iterates over `CHARCOUNT=12` uint8 elements doing scalar compare + subtract. Called millions of times per thread.

**Fix:** With `CHARCOUNT=12`, the entire `counts[12]` array fits in a single 128-bit SSE/NEON register. Replace the scalar loop with vectorized packed operations:

```c
#include <immintrin.h>

bool char_counts_subtract_simd(char_counts *from, char_counts *what) {
    if (from->length < what->length) return false;

    // Load 16 bytes (12 used, 4 padding — struct must be 16-byte aligned or use unaligned load)
    __m128i from_v = _mm_loadu_si128((__m128i*)from->counts);
    __m128i what_v = _mm_loadu_si128((__m128i*)what->counts);

    // Check for underflow: if any from[i] < what[i], saturating subtract differs from real subtract
    __m128i saturated = _mm_subs_epu8(from_v, what_v);
    __m128i real_sub  = _mm_sub_epi8(from_v, what_v);
    __m128i diff      = _mm_xor_si128(saturated, real_sub);

    // Mask to only check first 12 bytes (ignore padding bytes 12-15)
    // 0xFFF = bits for bytes 0-11
    if (_mm_movemask_epi8(diff) & 0xFFF) return false;

    _mm_storeu_si128((__m128i*)from->counts, saturated);
    from->length -= what->length;
    return true;
}
```

This replaces 12 comparisons + 12 subtractions + 12 branches with ~6 SIMD instructions + 1 branch. On ARM/NEON, equivalent using `vqsubq_u8` + `vceqq_u8`.

**Prerequisite:** Pad `char_counts.counts` to 16 bytes (currently 12). Or use unaligned loads (already shown above). The struct is 13 bytes currently; padding to 16 is natural.

### CPU-3. Better Work Distribution (MEDIUM IMPACT)

**Problem:** Strided parallelization at the top level:
```c
for (int i=curdictidx; i<ctx->dict_by_char_len[curchar]; i+=step)
```
Thread 0 gets entries {0, N, 2N...}. If entry 0 leads to 10M combinations and entry 1 leads to 100, thread 0 runs 100,000x longer. Other threads finish and idle.

**Options:**
- **Simple:** Pre-shuffle dictionary entries randomly so hot entries are distributed across threads
- **Better:** Use an atomic counter for dynamic work stealing — each thread atomically increments a shared index to get the next top-level entry
- **Best:** Work-stealing deque (but adds significant complexity)

The atomic counter approach is simplest:
```c
// shared across threads
volatile uint32_t next_l0_index = 0;

// in recurse_dict_words, when stack_len == 0:
int i = __sync_fetch_and_add(&next_l0_index, 1);
if (i >= dict_by_char_len[curchar]) break;
```

### CPU-4. Ring Buffer for `tasks_buffers` Queue (LOW-MEDIUM)

**Problem:** `tasks_buffers_add_buffer` and `_get_buffer` do linear scans through 64 slots under mutex:
```c
for (int i=0; i<TASKS_BUFFERS_SIZE; i++) {
    if (buffs->arr[i] == NULL) { ... }
}
```

**Fix:** Replace with a proper ring buffer (head/tail indices). O(1) insert/remove:
```c
typedef struct tasks_buffers_s {
    tasks_buffer* arr[TASKS_BUFFERS_SIZE];
    uint32_t head, tail;  // head = next write, tail = next read
    // ...
} tasks_buffers;
```

### CPU-5. Avoid Redundant `strlen` in Hot Path (LOW)

**Problem:** In `recurse_string_combs` line 125:
```c
for (int j=0; j<=strlen(scs[i].str); j++) {
    all_strs[all_offs++] = scs[i].str[j];
}
```
`strlen` is recomputed every iteration of `j`. The compiler may not hoist it since `scs[i].str` is a pointer that could theoretically alias `all_strs`.

**Fix:** Cache the length:
```c
int len = strlen(scs[i].str);
for (int j=0; j<=len; j++) { ... }
```
Or better, use `memcpy`:
```c
int len = strlen(scs[i].str) + 1; // include null terminator
memcpy(all_strs + all_offs, scs[i].str, len);
all_offs += len;
```

### CPU-6. `char_counts_copy` as memcpy (LOW)

**Problem:** Element-by-element copy loop for 13 bytes.

**Fix:** `memcpy(dst, src, sizeof(char_counts))` — compiler emits 1-2 instructions.

### CPU-7. Pre-sort Dictionary by Descending Length (LOW-MEDIUM)

**Problem:** Dictionary entries within each character group are unsorted. Longer words consume more of the remainder budget and hit the `word_count > MAX_WORD_LENGTH` or `remainder.length == 0` cutoffs sooner.

**Fix:** Sort `dict_by_char[ci]` entries by descending `counts.length` before starting the search. This helps the recursive search prune earlier, reducing the number of nodes explored.

### CPU-8. Precompute Compatibility Matrix (SPECULATIVE)

Before searching, for each pair of dictionary entries (i, j), check if `counts[i] + counts[j] <= seed_phrase`. Store as a bitset per entry. During recursion, intersect the candidate set with the bitset to skip entries that can never fit.

Setup cost: O(dict_size² × CHARCOUNT) = ~50M operations for 2048 entries. One-time.

Benefit depends on how many entries are incompatible. If the seed phrase is short and most pairs conflict, this could dramatically reduce the search tree. If the phrase is long and most pairs are compatible, the benefit is minimal.

### CPU Optimization Summary

| # | Optimization | Impact | Effort | Changes |
|---|-------------|--------|--------|---------|
| 1 | Buffer pool | HIGH | Low | `task_buffers.c`: add free-list, `gpu_cruncher.c`: return instead of free |
| 2 | SIMD char_counts | MEDIUM | Medium | `permut_types.c`: SSE/NEON subtract, pad struct to 16 bytes |
| 3 | Atomic work stealing | MEDIUM | Low | `cpu_cruncher.c`: atomic counter at top level |
| 4 | Ring buffer queue | LOW-MED | Low | `task_buffers.c`: head/tail indices |
| 5 | Cache strlen | LOW | Trivial | `cpu_cruncher.c`: 2 lines |
| 6 | memcpy char_counts | LOW | Trivial | `permut_types.c`: 1 line |
| 7 | Sort dict by length | LOW-MED | Low | `main.c`: qsort after loading dict |
| 8 | Compatibility matrix | SPECULATIVE | Medium | `main.c` + `cpu_cruncher.c`: bitset check |

---

## Alternative Approaches Evaluated

### AVX2/AVX-512 CPU-Only MD5

**Concept:** Eliminate GPU entirely. Each CPU thread does recursive backtracking + SIMD MD5. AVX2 = 8 parallel MD5 hashes per instruction, AVX-512 = 16.

**Throughput:** ~8-12B MD5/sec on an 8-core AVX2 desktop. ~25-40B on a 16-core AVX-512 server.

**Pros:**
- Eliminates entire producer-consumer pipeline (no buffers, no mutex, no PCIe)
- Dramatically simpler codebase (~500 lines vs ~1500)
- Each thread owns its search subspace end-to-end
- No kernel timeout issues

**Cons:**
- Must write/integrate AVX2 MD5 intrinsics (verbose but straightforward)
- x86-only (no ARM/Apple Silicon)
- Can't leverage discrete GPU if available

**Verdict:** Strong alternative if targeting x86 servers. Could coexist with GPU path.

### MD5-Only GPU (CPU generates full strings)

**Concept:** Move all permutation generation to CPU, send fully-formed candidate strings to GPU, GPU only does MD5.

**Why rejected:** Destroys the bandwidth advantage. The `permut_task` struct (96 bytes) encodes up to `fact(n)` candidates (e.g. 40320 for n=8). Sending raw strings would be 12,600x more data. The GPU would become memory-bandwidth starved.

### Apple Neural Engine

**Why rejected:** The ANE is a matrix multiply-accumulate (MAC) accelerator for INT8/FP16 tensor operations. It has no native support for bitwise operations (AND, OR, XOR, NOT), bit rotations, or 32-bit integer arithmetic — all core MD5 primitives. Performance would be orders of magnitude worse than CPU, if mappable at all.

### Apple M-series GPU (Metal)

**Concept:** Port from OpenCL to Metal Compute Shaders. Unified memory eliminates PCIe overhead.

**Throughput:** M1=3-5B, M2 Ultra=15-25B MD5/sec.

**Pros:** Zero-copy buffers, no transfer overhead. Kernel ports almost directly.
**Cons:** Apple deprecated OpenCL, Metal rewrite required. Fewer cores than discrete GPU.

**Verdict:** Viable if targeting macOS. Orthogonal to the batch-by-N optimization (which would also benefit Metal).



See `BUGS.md` for known bugs — fix before or alongside optimization work.
