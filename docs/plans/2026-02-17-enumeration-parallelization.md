# Enumeration Parallelization Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Improve 8-thread CPU enumeration efficiency from 57% to 80%+ by eliminating load imbalance, reducing lock contention, and optimizing buffer management.

**Architecture:** Four independent optimizations applied incrementally: (1) atomic work stealing replaces strided top-level loop, (2) pre-sort dictionary by descending word length for better stealing balance, (3) ring buffer replaces O(64) linear scan in task queue, (4) per-thread free-list eliminates mutex acquisitions for buffer obtain.

**Tech Stack:** C99, pthreads, `__sync_fetch_and_add` (GCC/Clang built-in), `qsort`

**Baseline benchmark (post-SIMD):**
- 1 thread: 5.060s, 266.3M anas/s
- 2 threads: 2.544s, 99% efficiency
- 4 threads: 1.566s, 81% efficiency
- 8 threads: 1.102s, 57% efficiency

---

### Task 1: Ring Buffer Queue

Replace O(64) linear scan in `tasks_buffers_add_buffer` and `tasks_buffers_get_buffer` with O(1) head/tail ring buffer. This is the safest change — purely internal to task_buffers.c/h, no API changes.

**Files:**
- Modify: `task_buffers.h:28-40` (struct fields)
- Modify: `task_buffers.c:58-77` (create), `task_buffers.c:79-99` (free), `task_buffers.c:101-132` (add_buffer), `task_buffers.c:135-173` (get_buffer)
- Test: existing `tests/test_cpu_enumeration.c`

**Step 1: Modify `tasks_buffers` struct in `task_buffers.h`**

Replace the `arr[]` + `num_ready` fields with ring buffer fields:

```c
typedef struct tasks_buffers_s {
    // Ring buffer for ready tasks (replaces linear-scan arr[])
    tasks_buffer* ring[TASKS_BUFFERS_SIZE];
    uint32_t ring_head;    // producer writes at ring[head % SIZE]
    uint32_t ring_tail;    // consumer reads at ring[tail % SIZE]
    volatile uint32_t ring_count;  // number of occupied slots (volatile for peek)
    volatile bool is_closed;

    // Free-list: returned buffers available for reuse
    tasks_buffer* free_arr[TASKS_BUFFERS_SIZE];
    uint32_t num_free;

    pthread_mutex_t mutex;
    pthread_cond_t not_full;   // rename inc_cond → not_full for clarity
    pthread_cond_t not_empty;  // rename dec_cond → not_empty for clarity
} tasks_buffers;
```

**Step 2: Update `tasks_buffers_create` in `task_buffers.c`**

Initialize ring buffer fields:

```c
int tasks_buffers_create(tasks_buffers* buffs) {
    buffs->ring_head = 0;
    buffs->ring_tail = 0;
    buffs->ring_count = 0;
    buffs->num_free = 0;
    buffs->is_closed = false;

    for (int i = 0; i < TASKS_BUFFERS_SIZE; i++) {
        buffs->ring[i] = NULL;
        buffs->free_arr[i] = NULL;
    }

    int errcode;
    errcode = pthread_mutex_init(&buffs->mutex, NULL);
    ret_iferr(errcode, "failed to init mutex while creating tasks buffers");
    errcode = pthread_cond_init(&buffs->not_full, NULL);
    ret_iferr(errcode, "failed to init not_full cond while creating tasks buffers");
    errcode = pthread_cond_init(&buffs->not_empty, NULL);
    ret_iferr(errcode, "failed to init not_empty cond while creating tasks buffers");

    return 0;
}
```

**Step 3: Update `tasks_buffers_free` in `task_buffers.c`**

Drain ring buffer entries:

```c
int tasks_buffers_free(tasks_buffers* buffs) {
    // Free any buffers still in the ring
    while (buffs->ring_count > 0) {
        uint32_t idx = buffs->ring_tail % TASKS_BUFFERS_SIZE;
        tasks_buffer_free(buffs->ring[idx]);
        buffs->ring[idx] = NULL;
        buffs->ring_tail++;
        buffs->ring_count--;
    }
    // Free any buffers in the free-list
    for (uint32_t i = 0; i < buffs->num_free; i++) {
        tasks_buffer_free(buffs->free_arr[i]);
        buffs->free_arr[i] = NULL;
    }
    buffs->num_free = 0;

    int errcode = 0;
    errcode |= pthread_mutex_destroy(&buffs->mutex);
    errcode |= pthread_cond_destroy(&buffs->not_full);
    errcode |= pthread_cond_destroy(&buffs->not_empty);
    return errcode;
}
```

**Step 4: Update `tasks_buffers_add_buffer` — O(1) ring insert**

```c
int tasks_buffers_add_buffer(tasks_buffers* buffs, tasks_buffer* buf) {
    int errcode = 0;
    errcode = pthread_mutex_lock(&buffs->mutex);
    ret_iferr(errcode, "failed to lock mutex while adding buffer");

    while (buffs->ring_count >= TASKS_BUFFERS_SIZE) {
        errcode = pthread_cond_wait(&buffs->not_full, &buffs->mutex);
        if (errcode) {
            pthread_mutex_unlock(&buffs->mutex);
            ret_iferr(errcode, "failed to wait for free space while adding buffer");
        }
    }

    buffs->ring[buffs->ring_head % TASKS_BUFFERS_SIZE] = buf;
    buffs->ring_head++;
    buffs->ring_count++;

    errcode = pthread_cond_signal(&buffs->not_empty);
    if (errcode) {
        pthread_mutex_unlock(&buffs->mutex);
        ret_iferr(errcode, "failed to signal not_empty while adding buffer");
    }

    pthread_mutex_unlock(&buffs->mutex);
    return 0;
}
```

**Step 5: Update `tasks_buffers_get_buffer` — O(1) ring remove**

```c
int tasks_buffers_get_buffer(tasks_buffers* buffs, tasks_buffer** buf) {
    int errcode = 0;
    errcode = pthread_mutex_lock(&buffs->mutex);
    ret_iferr(errcode, "failed to lock mutex while removing buffer");

    while (buffs->ring_count == 0) {
        if (buffs->is_closed) {
            pthread_mutex_unlock(&buffs->mutex);
            *buf = NULL;
            return 0;
        }
        errcode = pthread_cond_wait(&buffs->not_empty, &buffs->mutex);
        if (errcode) {
            pthread_mutex_unlock(&buffs->mutex);
            ret_iferr(errcode, "failed to wait for available buffer while removing buffer");
        }
    }

    *buf = buffs->ring[buffs->ring_tail % TASKS_BUFFERS_SIZE];
    buffs->ring[buffs->ring_tail % TASKS_BUFFERS_SIZE] = NULL;
    buffs->ring_tail++;
    buffs->ring_count--;

    errcode = pthread_cond_signal(&buffs->not_full);
    if (errcode) {
        pthread_mutex_unlock(&buffs->mutex);
        ret_iferr(errcode, "failed to signal not_full while removing buffer");
    }

    pthread_mutex_unlock(&buffs->mutex);
    return 0;
}
```

**Step 6: Update `tasks_buffers_close` and `tasks_buffers_num_ready`**

In `tasks_buffers_close`: rename `inc_cond` → `not_empty` in the broadcast call.

In `tasks_buffers_num_ready`: change `buffs->num_ready` → `buffs->ring_count`.

**Step 7: Update all references to old field names**

Search for `num_ready`, `inc_cond`, `dec_cond`, `arr[` in the codebase. Key locations:
- `main.c:198` — `tasks_buffs.num_ready` → `tasks_buffs.ring_count`

**Step 8: Build and test**

Run: `cd /home/debian/AleCode/anabrute_cl/build && cmake .. && make -j$(nproc) 2>&1`
Expected: Clean compile

Run: `cd /home/debian/AleCode/anabrute_cl/build && ctest --output-on-failure`
Expected: All 4 tests pass (hash_parsing, dict_parsing, cpu_enumeration, cruncher)

**Step 9: Quick benchmark sanity check**

Run: `cd /home/debian/AleCode/anabrute_cl && ./build/bench_enum 8`
Expected: Similar numbers to baseline (ring buffer alone shouldn't change throughput significantly)

**Step 10: Commit**

```bash
git add task_buffers.c task_buffers.h main.c
git commit -m "refactor: ring buffer queue replaces O(64) linear scan in task_buffers"
```

---

### Task 2: Pre-sort Dictionary by Descending Length

Sort each `dict_by_char[ci][]` array by descending word length after populating it. Longer words create smaller recursion subtrees, improving work distribution.

**Files:**
- Modify: `main.c:40-48` (after dict_by_char population)
- Modify: `bench_enum.c:111-119` (same location)
- Modify: `tests/test_cpu_enumeration.c:33-40` (same location in helper)

**Step 1: Add sort comparator and sort call to `main.c`**

After the `dict_by_char` population loop (line 47), add:

```c
/* Sort each dict_by_char bucket by descending word length for work-stealing balance */
static int cmp_ccs_length_desc(const void *a, const void *b) {
    const char_counts_strings *ca = *(const char_counts_strings *const *)a;
    const char_counts_strings *cb = *(const char_counts_strings *const *)b;
    return (int)cb->counts.length - (int)ca->counts.length;
}

// ... after the population loop:
for (int ci = 0; ci < CHARCOUNT; ci++) {
    if (dict_by_char_len[ci] > 1) {
        qsort(dict_by_char[ci], dict_by_char_len[ci], sizeof(char_counts_strings*), cmp_ccs_length_desc);
    }
}
```

Note: the comparator takes `char_counts_strings**` because the array stores pointers. Each element is a `char_counts_strings*`, so `a` points to a `char_counts_strings*`.

**Step 2: Add same sort to `bench_enum.c`**

After the dict_by_char population loop (line 119), add the same comparator and sort loop. Since the comparator is small, duplicate it rather than creating a shared header.

**Step 3: Add same sort to `tests/test_cpu_enumeration.c`**

After the dict_by_char population loop (line 40), add the same sort. This ensures tests exercise the sorted path.

**Step 4: Build and test**

Run: `cd /home/debian/AleCode/anabrute_cl/build && make -j$(nproc) 2>&1`
Expected: Clean compile

Run: `cd /home/debian/AleCode/anabrute_cl/build && ctest --output-on-failure`
Expected: All 4 tests pass

**Step 5: Benchmark**

Run: `cd /home/debian/AleCode/anabrute_cl && ./build/bench_enum 8`
Expected: Possible slight single-thread improvement; multi-thread numbers depend on interaction with strided assignment (may not improve much until work stealing is added)

**Step 6: Commit**

```bash
git add main.c bench_enum.c tests/test_cpu_enumeration.c
git commit -m "perf: pre-sort dictionary by descending length for work-stealing balance"
```

---

### Task 3: Atomic Work Stealing

Replace strided `i += num_cpu_crunchers` top-level loop with `__sync_fetch_and_add` on a shared atomic counter. This is the highest-impact change for multi-thread efficiency.

**Files:**
- Modify: `cpu_cruncher.h:7-26` (add shared counter pointer to ctx)
- Modify: `cpu_cruncher.c:4-20` (ctx_create accepts counter), `cpu_cruncher.c:182-211` (atomic loop)
- Modify: `main.c:131-137` (allocate shared counter, pass to ctx_create)
- Modify: `bench_enum.c:41-49` (same)
- Modify: `tests/test_cpu_enumeration.c:47-48` (same, single-thread uses counter too)

**Step 1: Add shared counter to `cpu_cruncher_ctx` in `cpu_cruncher.h`**

Add a pointer to the shared atomic counter:

```c
typedef struct cpu_cruncher_ctx_s {
    // parallelization
    uint32_t num_cpu_crunchers;
    uint32_t cpu_cruncher_id;

    // atomic work stealing: shared counter for top-level dict entries
    volatile uint32_t *shared_l0_counter;

    // ... rest unchanged
} cpu_cruncher_ctx;
```

Update the `cpu_cruncher_ctx_create` signature to accept the counter:

```c
void cpu_cruncher_ctx_create(cpu_cruncher_ctx* cruncher, uint32_t cpu_cruncher_id, uint32_t num_cpu_crunchers,
                             char_counts* seed_phrase, char_counts_strings* (*dict_by_char)[CHARCOUNT][MAX_DICT_SIZE], int* dict_by_char_len,
                             tasks_buffers* tasks_buffs, volatile uint32_t *shared_l0_counter);
```

**Step 2: Update `cpu_cruncher_ctx_create` in `cpu_cruncher.c`**

Accept and store the counter:

```c
void cpu_cruncher_ctx_create(cpu_cruncher_ctx* cruncher, uint32_t cpu_cruncher_id, uint32_t num_cpu_crunchers,
                             char_counts* seed_phrase, char_counts_strings* (*dict_by_char)[CHARCOUNT][MAX_DICT_SIZE], int* dict_by_char_len,
                             tasks_buffers* tasks_buffs, volatile uint32_t *shared_l0_counter)
{
    cruncher->num_cpu_crunchers = num_cpu_crunchers;
    cruncher->cpu_cruncher_id = cpu_cruncher_id;
    cruncher->shared_l0_counter = shared_l0_counter;
    // ... rest unchanged
}
```

**Step 3: Replace strided loop with atomic grab in `recurse_dict_words`**

Current code (cpu_cruncher.c:182-211):
```c
int step = 1;
if (stack_len == 0) {
    step = ctx->num_cpu_crunchers;
}

int errcode=0;
for (int i=curdictidx; i<ctx->dict_by_char_len[curchar]; i+=step) {
    if (stack_len == 0) {
        ctx->progress_l0_index = i;
    }
    // ... process entry i
}
```

Replace with:
```c
int errcode=0;
if (stack_len == 0) {
    // Atomic work stealing: each thread grabs the next available index
    uint32_t i;
    while ((i = __sync_fetch_and_add(ctx->shared_l0_counter, 1)) < (uint32_t)ctx->dict_by_char_len[curchar]) {
        ctx->progress_l0_index = i;

        stack[stack_len].ccs = (*ctx->dict_by_char)[curchar][i];

        char_counts next_remainder;
        char_counts_copy(remainder, &next_remainder);
        for (uint8_t ccs_count=1; char_counts_subtract(&next_remainder, &(*ctx->dict_by_char)[curchar][i]->counts); ccs_count++) {
            stack[stack_len].count = ccs_count;

            int next_char = curchar;
            int next_idx = i+1;

            if(next_remainder.counts[next_char] == 0) {
                next_char++;
                next_idx = 0;
            }

            errcode = recurse_dict_words(ctx, &next_remainder, next_char, next_idx, stack, stack_len + 1, scs);
            if (errcode) return errcode;
        }
    }
} else {
    for (int i=curdictidx; i<ctx->dict_by_char_len[curchar]; i++) {
        stack[stack_len].ccs = (*ctx->dict_by_char)[curchar][i];

        char_counts next_remainder;
        char_counts_copy(remainder, &next_remainder);
        for (uint8_t ccs_count=1; char_counts_subtract(&next_remainder, &(*ctx->dict_by_char)[curchar][i]->counts); ccs_count++) {
            stack[stack_len].count = ccs_count;

            int next_char = curchar;
            int next_idx = i+1;

            if(next_remainder.counts[next_char] == 0) {
                next_char++;
                next_idx = 0;
            }

            errcode = recurse_dict_words(ctx, &next_remainder, next_char, next_idx, stack, stack_len + 1, scs);
            if (errcode) return errcode;
        }
    }
}

return errcode;
```

Note: the `step` variable is removed entirely. At `stack_len > 0`, `step` was always 1, so the loop is unchanged. At `stack_len == 0`, the atomic counter replaces strided access. The `curdictidx` parameter is ignored at stack_len==0 (the counter handles starting position).

**Step 4: Update `run_cpu_cruncher_thread` initial call**

The initial call `recurse_dict_words(ctx, &local_remainder, 0, ctx->cpu_cruncher_id, stack, 0, scs)` passes `cpu_cruncher_id` as `curdictidx`. With work stealing, this parameter is ignored at stack_len==0, so it can stay as-is (or change to 0 for clarity).

**Step 5: Update `main.c` — allocate shared counter, update ctx_create calls**

After the task_buffers creation (line 53), add:

```c
volatile uint32_t shared_l0_counter = 0;
```

Update the ctx_create call (line 136):

```c
cpu_cruncher_ctx_create(cpu_cruncher_ctxs+id, id, num_cpu_crunchers, &seed_phrase, &dict_by_char, dict_by_char_len, &tasks_buffs, &shared_l0_counter);
```

Update progress display (main.c:184-189). Instead of min/max across threads, show shared counter:

```c
// CPU progress (shared counter = items completed globally)
uint32_t cpu_progress = shared_l0_counter;
if (cpu_progress > (uint32_t)dict_by_char_len[0]) cpu_progress = dict_by_char_len[0];
```

Update progress format string to show single value:
```c
int pos = sprintf(strbuf, "%02ld:%02ld:%02ld | %d cpus: %u/%d | %d buffs",
       elapsed_secs/3600, (elapsed_secs/60)%60, elapsed_secs%60,
       num_cpu_crunchers, cpu_progress, dict_by_char_len[0],
       tasks_buffs.ring_count);
```

Update "done" check (main.c:232). All threads are done when every thread has set progress_l0_index to dict_by_char_len[0]:

```c
// Check: all CPU threads done?
bool cpus_done = true;
for (int i = 0; i < num_cpu_crunchers; i++) {
    if (cpu_cruncher_ctxs[i].progress_l0_index < dict_by_char_len[0]) {
        cpus_done = false;
        break;
    }
}
if (cpus_done) {
    tasks_buffers_close(&tasks_buffs);
}
```

**Step 6: Update `bench_enum.c` — allocate shared counter**

In `run_benchmark`, after creating tasks_buffers, add shared counter and pass to ctx_create:

```c
volatile uint32_t shared_l0_counter = 0;

cpu_cruncher_ctx ctxs[num_threads];
for (int i = 0; i < num_threads; i++) {
    cpu_cruncher_ctx_create(&ctxs[i], i, num_threads, seed, &dict_by_char, dict_by_char_len, &buffs, &shared_l0_counter);
}
```

**Step 7: Update `tests/test_cpu_enumeration.c` — shared counter for single-thread**

In `run_cruncher_with_dict`, add counter and pass to ctx_create:

```c
volatile uint32_t shared_l0_counter = 0;

cpu_cruncher_ctx ctx;
cpu_cruncher_ctx_create(&ctx, 0, 1, &seed, &dict_by_char, dict_by_char_len, &tasks_buffs, &shared_l0_counter);
```

**Step 8: Build and test**

Run: `cd /home/debian/AleCode/anabrute_cl/build && cmake .. && make -j$(nproc) 2>&1`
Expected: Clean compile (may need `-Wno-int-conversion` already in CMakeLists.txt)

Run: `cd /home/debian/AleCode/anabrute_cl/build && ctest --output-on-failure`
Expected: All 4 tests pass. Task counts may differ slightly from before if dict sorting changes enumeration order, but the key properties hold (correct n values, non-zero counts for valid anagrams).

**Step 9: Benchmark — this is the big one**

Run: `cd /home/debian/AleCode/anabrute_cl && ./build/bench_enum 8`
Expected: 8-thread efficiency should jump from ~57% to ~75-85%. Single-thread should be unchanged or slightly better (no step multiplication).

**Step 10: Commit**

```bash
git add cpu_cruncher.c cpu_cruncher.h main.c bench_enum.c tests/test_cpu_enumeration.c
git commit -m "perf: atomic work stealing replaces strided enumeration for better parallelism"
```

---

### Task 4: Per-Thread Free-List

Each CPU thread maintains a local cache of recycled buffers to avoid locking the global free-list on every `obtain` call.

**Files:**
- Modify: `cpu_cruncher.h:7-26` (add local_free fields to ctx)
- Modify: `cpu_cruncher.c:22-48` (submit_tasks uses local obtain)
- Modify: `cpu_cruncher.c:216-241` (flush local_free on exit)

**Step 1: Add local free-list to `cpu_cruncher_ctx` in `cpu_cruncher.h`**

```c
#define LOCAL_FREE_CAP 4

typedef struct cpu_cruncher_ctx_s {
    // ... existing fields ...

    // Per-thread buffer free-list (avoids global mutex for obtain)
    tasks_buffer* local_free[LOCAL_FREE_CAP];
    int local_free_count;
} cpu_cruncher_ctx;
```

**Step 2: Initialize in `cpu_cruncher_ctx_create`**

```c
cruncher->local_free_count = 0;
```

**Step 3: Add `cpu_obtain_buffer` helper in `cpu_cruncher.c`**

Add a static helper that checks local cache first, then bulk-grabs from global:

```c
static tasks_buffer* cpu_obtain_buffer(cpu_cruncher_ctx *ctx) {
    // Check local cache first (no lock)
    if (ctx->local_free_count > 0) {
        tasks_buffer *buf = ctx->local_free[--ctx->local_free_count];
        tasks_buffer_reset(buf);
        return buf;
    }

    // Bulk grab from global free-list
    pthread_mutex_lock(&ctx->tasks_buffs->mutex);
    while (ctx->local_free_count < LOCAL_FREE_CAP && ctx->tasks_buffs->num_free > 0) {
        ctx->local_free[ctx->local_free_count++] = ctx->tasks_buffs->free_arr[--ctx->tasks_buffs->num_free];
    }
    pthread_mutex_unlock(&ctx->tasks_buffs->mutex);

    if (ctx->local_free_count > 0) {
        tasks_buffer *buf = ctx->local_free[--ctx->local_free_count];
        tasks_buffer_reset(buf);
        return buf;
    }

    // Nothing available — allocate fresh
    return tasks_buffer_allocate();
}
```

**Step 4: Replace `tasks_buffers_obtain` calls in `submit_tasks`**

In `submit_tasks` (cpu_cruncher.c:42), change:
```c
*bufp = tasks_buffers_obtain(ctx->tasks_buffs);
```
to:
```c
*bufp = cpu_obtain_buffer(ctx);
```

**Step 5: Free local buffers on thread exit**

At the end of `run_cpu_cruncher_thread`, after flushing per-N buffers, free any cached local buffers:

```c
// Free local free-list buffers
for (int i = 0; i < ctx->local_free_count; i++) {
    tasks_buffer_free(ctx->local_free[i]);
}
ctx->local_free_count = 0;
```

**Step 6: Build and test**

Run: `cd /home/debian/AleCode/anabrute_cl/build && make -j$(nproc) 2>&1`
Expected: Clean compile

Run: `cd /home/debian/AleCode/anabrute_cl/build && ctest --output-on-failure`
Expected: All 4 tests pass

**Step 7: Final benchmark**

Run: `cd /home/debian/AleCode/anabrute_cl && ./build/bench_enum 8`
Expected: Marginal improvement on top of atomic work stealing. Total improvement should be significant.

**Step 8: Commit**

```bash
git add cpu_cruncher.c cpu_cruncher.h
git commit -m "perf: per-thread buffer free-list reduces mutex contention"
```

---

### Task 5: Final Verification

**Step 1: Run full benchmark comparison**

Run: `cd /home/debian/AleCode/anabrute_cl && ./build/bench_enum 8`
Compare with baseline:
- Single-thread: should be >= 266M anas/s
- 8-thread efficiency: target >= 80%

**Step 2: Run full test suite**

Run: `cd /home/debian/AleCode/anabrute_cl/build && ctest --output-on-failure`
Expected: All 4 tests pass

**Step 3: Commit all together if any uncommitted changes remain**

If any fixups were needed, commit them.
