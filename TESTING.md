# TESTING.md — Test Coverage Design

## Goal

Safety net for the optimization work in IDEAS.md. Lock down current behavior so refactoring doesn't break things.

## Framework

Plain C + `assert()`. No dependencies. Each test file is a standalone `main()`. CMake `add_test()` targets, run all with `ctest`.

## Test Files

```
tests/
  test_permut_types.c    — char_counts operations
  test_fact.c            — factorial lookup
  test_hashes.c          — hex<->binary conversions
  test_task_buffers.c    — single-buffer ops + basic queue
  test_format.c          — format_bignum
  test_cpu_pipeline.c    — integration: small dict → tasks_buffer verification
```

### `test_permut_types.c` — char_counts operations

Tests for `char_counts_create`, `_subtract`, `_equal`, `_contains`, `_copy`, `_strings_create`, `_strings_addstring`.

Key cases:
- `char_counts_create` with valid seed phrase chars → correct counts and length
- `char_counts_create` with invalid char → returns true (error)
- `char_counts_subtract` success: verify counts and length decremented
- `char_counts_subtract` failure on length: verify returns false
- `char_counts_subtract` failure on individual count underflow: verify returns false
- `char_counts_subtract` repeated (used in the `for(ccs_count=1; subtract(...); ccs_count++)` pattern) — verify multiple subtractions work until exhaustion
- `char_counts_equal` matching and non-matching pairs
- `char_counts_contains` with subset, superset, and disjoint counts
- `char_counts_copy` → verify deep copy (modify source, dest unchanged)
- `char_counts_strings_addstring` → verify string storage and count

### `test_fact.c` — factorial lookup

- `fact(0)` through `fact(20)` — verify against known values
- `fact(21)` and above → returns 0
- Focus on values actually used: `fact(2)` through `fact(5)` (since `MAX_WORD_LENGTH=5`)

### `test_hashes.c` — hex<->binary conversions

- Round-trip: `ascii_to_hash` → `hash_to_ascii` → compare with original
- Known MD5 values: `md5("") = d41d8cd98f00b204e9800998ecf8427e`
- Edge cases: all-zero hash, all-F hash

### `test_task_buffers.c` — buffer operations

Single-buffer tests (no threads):
- `tasks_buffer_allocate` → non-null, `num_tasks == 0`
- `tasks_buffer_add_task` → verify `permut_task` fields packed correctly: `all_strs`, `offsets`, `a[]`, `n`, `i=0`, `iters_done=0`
- `tasks_buffer_isfull` → false when empty, true after `PERMUT_TASKS_IN_KERNEL_TASK` adds
- Add a task with known offsets, verify the `a[]` array and permutable count `n` are computed correctly

Queue tests (threaded):
- Single producer, single consumer: add buffer, get buffer, verify same pointer
- Producer blocks when full (`TASKS_BUFFERS_SIZE` buffers), unblocks when consumer drains
- `tasks_buffers_close` → consumer gets NULL

### `test_format.c` — format_bignum

- `format_bignum(999, buf, 1000)` → `"999"`
- `format_bignum(1000, buf, 1000)` → `"1K"`
- `format_bignum(1500000, buf, 1000)` → `"1M"`
- `format_bignum(0, buf, 1000)` → `"0"`

**Requires:** Extract `format_bignum` from `main.c` (currently `static`) into a shared header or separate file.

### `test_cpu_pipeline.c` — integration test

End-to-end test of the CPU recursion pipeline without GPU:

1. Define a tiny seed phrase (e.g., `"abc"` with `CHARCOUNT` adjusted, or use a subset of the real seed phrase)
2. Create a small dictionary (3-4 words that form valid anagrams)
3. Set up a real `tasks_buffers` queue
4. Run `recurse_dict_words` on a single thread
5. Drain the queue, verify:
   - Expected number of `tasks_buffer`s produced
   - Each `permut_task` has valid `n`, `i=0`, non-empty `all_strs`, non-empty `offsets`
   - Total task count matches expected combinatorial count

This catches regressions in the recursion logic without needing to mock anything.

## CMake Setup

Add `tests/CMakeLists.txt`:

```cmake
enable_testing()

# Pure function tests — no OpenCL dependency
add_executable(test_permut_types test_permut_types.c ../permut_types.c ../seedphrase.c)
add_test(NAME permut_types COMMAND test_permut_types)

add_executable(test_fact test_fact.c ../fact.c)
add_test(NAME fact COMMAND test_fact)

add_executable(test_hashes test_hashes.c ../hashes.c)
add_test(NAME hashes COMMAND test_hashes)

add_executable(test_task_buffers test_task_buffers.c ../task_buffers.c)
target_link_libraries(test_task_buffers pthread)
add_test(NAME task_buffers COMMAND test_task_buffers)

add_executable(test_format test_format.c ../format.c)  # after extraction from main.c
add_test(NAME format COMMAND test_format)

# Integration test — links CPU-side code, no OpenCL
add_executable(test_cpu_pipeline test_cpu_pipeline.c
    ../cpu_cruncher.c ../permut_types.c ../seedphrase.c
    ../task_buffers.c ../fact.c)
target_link_libraries(test_cpu_pipeline pthread)
add_test(NAME cpu_pipeline COMMAND test_cpu_pipeline)
```

Include from root `CMakeLists.txt`:
```cmake
add_subdirectory(tests)
```

## Refactoring Required

Minimal — only one change needed before writing tests:

1. **Extract `format_bignum` from `main.c`** — move to a shared file (e.g., `format.c/h`) or just make it non-static and declare in a header. Currently `static` on line 63 of `main.c`.

Everything else is testable as-is.

## What's NOT Tested

- `gpu_cruncher.c` — requires OpenCL hardware. Covered by existing `kernel_debug.c`.
- `kernels/permut.cl` — covered by `kernel_debug.c`.
- `main()` orchestration — integration only, tested manually.
- `os.c` — platform wrappers, trivial.

## Implementation Order

1. Create `tests/` directory and `CMakeLists.txt`
2. `test_fact.c` — trivial, validates the test setup works
3. `test_permut_types.c` — highest value, covers the hottest CPU code path
4. `test_hashes.c` — pure conversions
5. `test_task_buffers.c` — validates task packing (critical for GPU correctness)
6. Extract `format_bignum`, write `test_format.c`
7. `test_cpu_pipeline.c` — integration test, most setup work
