# Bug Fix Testing Design

## Goal

Fix the 3 HIGH priority bugs from BUGS.md in a verifiable manner using integration/end-to-end tests that exercise real code paths with real inputs.

## Test Infrastructure

- `tests/` directory, one C file per test area
- Plain CTest + `assert.h`, no external framework
- Compiled with `-fsanitize=address -fsanitize=undefined` to catch memory issues
- Run via `cd build && ctest`

## Refactoring: Extract Functions from main.c

`read_hashes`, `read_dict`, and `format_bignum` are trapped in `main.c`. To make them linkable from test executables:

- `read_hashes` → `hashes.c/h`
- `read_dict` → new `dict.c/h`
- `format_bignum` → `os.c/h`

No behavioral changes, just moving code to make it testable.

## Test 1: CPU Anagram Enumeration (`tests/test_cpu_enumeration.c`)

Integration test for the full CPU pipeline: dict setup → recursive enumeration → task packing.

**Setup:**
- Use the real seed phrase "tyranousplutotwits"
- Build dict entries programmatically (avoid file I/O in this test):
  - Case A: `["tyranousplutotwits"]` → expect 1 task with n=1
  - Case B: `["tyranous", "plutotwits"]` → expect 1 combination with n=2
  - Case C: `["tyranous", "pluto", "twits"]` → expect 1 combination with n=3
- Create a `tasks_buffers` queue, single CPU cruncher context (1 thread, id=0)
- Run `run_cpu_cruncher_thread` synchronously

**Verification:**
- Drain the task buffer queue
- Decode `permut_task` structs back into candidate strings (using the same offsets/all_strs logic the kernel uses)
- Assert the expected anagram combinations are found (and no unexpected ones)

**Coverage:** Exercises `recurse_dict_words`, `recurse_string_combs`, `recurse_combs`, `submit_tasks`, `tasks_buffer_add_task`, and the code around bug #1. Doesn't catch the specific OOB read (which is behaviorally masked) but ensures enumeration correctness.

## Test 2: Hash File Parsing (`tests/test_hash_parsing.c`)

Integration test for `read_hashes` with real-format input files.

**Test cases:**
- **Valid file with 3 known hashes:** Write 3 known 32-char hex strings to temp file. Call `read_hashes`. Assert count=3 and parsed values match expected uint32 arrays.
- **File with invalid line (bug #3):** Include a line that's not 32 hex chars. Assert the returned count excludes the invalid line. Currently fails because the invalid line is parsed and counted.
- **Boundary file (bug #2):** Create a file with exactly `(file_size+1)/33` hashes (the estimated count). Call `read_hashes`. With ASan, the off-by-one heap overflow is caught. Assert correct count.

**Coverage:** Directly tests bugs #2 and #3. Also validates the hash parsing roundtrip (ascii_to_hash correctness).

## Test 3: Dict File Parsing (`tests/test_dict_parsing.c`)

Integration test for `read_dict` with real-format input files.

**Test cases:**
- **Basic loading:** Write a small dict with known words from the seed phrase alphabet. Verify correct `dict_length` and that entries contain expected char_counts.
- **Filtering:** Include words with characters not in the seed phrase (e.g., "hello" has 'h','e' not in mapping). Verify they're excluded.
- **Dedup and grouping:** Include anagram-equivalent words (e.g., "list"/"slit" if applicable). Verify they end up in the same `char_counts_strings` entry.
- **Line trimming edge cases:** Empty lines, single-char lines (covers bug #5 from BUGS.md at MEDIUM level).

## Bug Fixes

After tests are written and confirmed failing:

1. **Bug #2** (`main.c:99`): Change `hashes_num > hashes_num_est` to `hashes_num >= hashes_num_est`
2. **Bug #3** (`main.c:95`): Add `continue;` after the invalid-hash fprintf
3. **Bug #1** (`cpu_cruncher.c:180`): Change `&` to `&&` — test won't behaviorally fail but ASan will catch the OOB read. After fix, ASan clean.

## File Layout

```
tests/
  test_cpu_enumeration.c
  test_hash_parsing.c
  test_dict_parsing.c
```

CMakeLists.txt changes: `enable_testing()`, add_executable for each test, link against project source files, add ASan flags, `add_test()`.
