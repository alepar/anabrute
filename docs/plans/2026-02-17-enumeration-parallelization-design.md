# CPU Enumeration Parallelization Design

## Problem

Multi-thread enumeration efficiency drops to 57% at 8 threads (baseline: 266.3M anas/s single-thread, 1.102s at 8 threads). The Metal cruncher on Apple Silicon is starved for task buffers because CPU enumeration can't produce fast enough.

Root causes:
1. **Load imbalance** (HIGH): Strided top-level assignment (`i += num_cpu_crunchers`) gives each thread fixed dictionary entries. Subtree sizes vary enormously — short common words produce orders of magnitude more combinations than long rare words.
2. **Single global mutex** (MEDIUM): All CPU producers and GPU consumers share one mutex for add_buffer, get_buffer, obtain, and recycle operations.
3. **O(64) linear scan under lock** (LOW): Finding empty/occupied slots requires scanning the full array while holding the mutex.

## Solution: Four Optimizations

### 1. Atomic Work Stealing (cpu_cruncher.c)

Replace strided loop at stack_len==0 with a shared atomic counter:

```c
// Shared across all CPU cruncher threads:
_Atomic uint32_t shared_l0_index;

// In recurse_dict_words, at stack_len == 0:
uint32_t i;
while ((i = __sync_fetch_and_add(ctx->shared_l0_index, 1)) < dict_by_char_len[curchar]) {
    // process dictionary entry i
}
```

Threads that finish small subtrees immediately grab the next work item. Atomic increment is ~1 cycle overhead per top-level entry — negligible vs recursion cost.

### 2. Pre-sort Dictionary by Descending Length (main.c/bench_enum.c)

Sort each `dict_by_char[ci][]` array by descending `counts.length` after populating. Longer words create smaller subtrees (subtract more characters per step). Processing heavy items first improves work-stealing balance: early threads get large subtrees, late threads get light ones.

One-time `qsort` at startup. No runtime cost.

### 3. Ring Buffer Queue (task_buffers.c)

Replace linear O(64) scan with circular buffer:

```c
tasks_buffer *ring[TASKS_BUFFERS_SIZE];
uint32_t head;   // producer writes here
uint32_t tail;   // consumer reads here
uint32_t count;  // occupied slots
```

`add_buffer`: `ring[head++ % SIZE]`, O(1). `get_buffer`: `ring[tail++ % SIZE]`, O(1). Mutex retained for condvar signaling but hold time drops from O(64) to O(1).

### 4. Per-Thread Free-List (cpu_cruncher.c)

Each CPU thread keeps a small local stack (capacity 2-4) of recycled buffers:

```c
// In cpu_cruncher_ctx:
tasks_buffer *local_free[4];
int local_free_count;
```

`obtain` checks local stack first (no lock). Falls back to global free-list only when empty. GPU consumers recycle to global free-list (they don't need local stacks since they don't produce buffers).

Eliminates ~50% of mutex acquisitions in the produce/recycle cycle.

## Files Modified

| File | Change |
|---|---|
| `cpu_cruncher.c` | Atomic counter loop, local free-list obtain |
| `cpu_cruncher.h` | Add `shared_l0_index` pointer, `local_free[]` fields |
| `task_buffers.c` | Ring buffer internals, per-thread free-list API |
| `task_buffers.h` | Ring buffer fields (head/tail/count), free-list API |
| `main.c` | Allocate shared atomic counter, sort dict_by_char, pass to crunchers |
| `bench_enum.c` | Update for shared counter setup, add dict sorting |

## Expected Impact

| Optimization | Single-thread | Multi-thread |
|---|---|---|
| Atomic work stealing | No change | Major — eliminates load imbalance |
| Pre-sort descending | Slight improvement | Improves stealing balance |
| Ring buffer | Negligible | Reduces lock hold time |
| Per-thread free-list | Negligible | Eliminates ~50% mutex acquisitions |

Target: 8-thread efficiency from 57% to 80%+.

## Verification

1. `bench_enum` before changes (baseline already captured)
2. Apply changes, build, `ctest --output-on-failure` — all tests pass
3. `bench_enum` after — compare single-thread throughput and multi-thread efficiency
4. End-to-end `./anabrute` — correct results
