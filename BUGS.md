# Known Bugs

## MEDIUM — Correctness / Robustness

### 5. `gpu_cruncher.c:156-157` — division by zero in `gpu_cruncher_get_stats`

```c
*busy_percentage = (float) micros_in_kernel / (max_time_ends-min_time_start) * 100.0f;
*anas_per_sec = (float)(calculated_anas) / ((max_time_ends-min_time_start)/1000000.0f);
```

If only one stats entry exists, `max_time_ends == min_time_start` and division is by zero. Called from the main thread's status loop which starts running immediately.

### 6. `permut_types.c:25` — `char_counts_subtract` corrupts `from` on partial failure

Modifies `from->length` and early `counts[]` entries before discovering a later entry would underflow. Returns `false` with `from` in an inconsistent state. Works by accident in current usage (caller passes a copy in `cpu_cruncher.c:200`), but the function's contract is broken.

### 7. `main.c:66-69` — `format_bignum` array overrun

`size_suffixes` has 6 entries (indices 0-5). Values near `UINT64_MAX` with `div=1000` produce `divs=6`, reading past the array.

### 9. `gpu_cruncher.c:24` — file descriptor leak in `read_file`

```c
if (filesize != read) {
    free(buf);
    return NULL;   // fd is never closed
}
```

Missing `fclose(fd)` before return.

## LOW — Latent / Cosmetic

### 12. `common.h:31,33` — unparenthesized macros

`#define PERMUT_TASKS_IN_KERNEL_TASK 256*1024` — using this in an expression like `sizeof(x) * PERMUT_TASKS_IN_KERNEL_TASK` would miscompute due to operator precedence.

### 13. `gpu_cruncher.c:167,179,221-235` — `ret_iferr` in `void*` functions

`ret_iferr` expands to `return val` where `val` is `int`. In `run_gpu_cruncher_thread` (returns `void*`), this implicitly converts int to pointer — UB. Harmless since `pthread_join` discards the return value.

### 14. `gpu_cruncher.c:306` — OpenCL event leak in `krnl_permut`

`krnl_permut_free` releases the kernel and mem object but never calls `clReleaseEvent(krnl->event)`. The event is created by `clEnqueueNDRangeKernel` and waited on, but never released.

### 15. `main.c:252` — fragile `hashes_reversed` non-empty check

```c
if (gpu_cruncher_ctxs[gi].hashes_reversed[hi*MAX_STR_LENGTH/4]) {
```

Checks if the first 4 bytes of the matched-string slot are non-zero via uint32 access. Works because matched strings start with printable ASCII, but fragile and confusing.

### 16. `gpu_cruncher.c:57` — pointer-to-integer without cast

```c
const cl_context_properties ctx_props [] = { CL_CONTEXT_PLATFORM, platform_id, 0, 0 };
```

`platform_id` is a pointer but `cl_context_properties` is `intptr_t`. Should cast: `(cl_context_properties)platform_id`.

---

## Fixed

- **#1** `cpu_cruncher.c:180` — `&` vs `&&` OOB read *(fixed: changed to `&&`)*
- **#2** `hashes.c` — off-by-one heap overflow in `read_hashes` *(fixed: `>` to `>=`)*
- **#3** `hashes.c` — invalid hash lines parsed without `continue` *(fixed: added `continue`)*
- **#4** `cpu_cruncher.c` — VLA overflows in `permut[]` and `all_strs[]` *(fixed: use `MAX_OFFSETS_LENGTH`/`MAX_STR_LENGTH`)*
- **#8** `dict.c` — line trimming OOB for short lines *(fixed: bounds checks + empty string skip)*
- **#10** `dict.c` — fd leak on error paths in `read_dict` *(fixed: added `fclose` calls)*
- **#11** `task_buffers.c` — partial memset of `c[]` array *(fixed: now zeros full `MAX_OFFSETS_LENGTH`)*
