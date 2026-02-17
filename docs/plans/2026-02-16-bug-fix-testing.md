# Bug Fix Testing Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fix the 3 HIGH priority bugs from BUGS.md with integration tests that verify correctness.

**Architecture:** Extract `read_hashes` and `read_dict` from `main.c` into shared files so test executables can link them. Write integration tests using CTest + assert + ASan. TDD: write failing tests first, then fix bugs.

**Tech Stack:** C99, CMake, CTest, AddressSanitizer (`-fsanitize=address,undefined`)

---

### Task 1: Extract `read_hashes` from `main.c` to `hashes.c/h`

**Files:**
- Modify: `hashes.h` — add `read_hashes` declaration
- Modify: `hashes.c` — move `read_hashes` implementation from `main.c`
- Modify: `main.c` — remove `read_hashes` function, add `#include "hashes.h"` (already present)
- Modify: `kernel_debug.c` — remove local `read_hashes` and its local `format_bignum` duplicate; use shared `read_hashes` from `hashes.h`

**Step 1: Add declaration to `hashes.h`**

Add before the `#endif`:

```c
uint32_t read_hashes(const char *file_name, uint32_t **hashes);
```

Note: change parameter from `char *` to `const char *` and drop the meaningless `const` on the return type vs the original in `main.c`.

**Step 2: Move `read_hashes` from `main.c` to `hashes.c`**

Cut lines 73-109 from `main.c` (the entire `read_hashes` function) and paste into `hashes.c`. Update the signature to match the header:

```c
uint32_t read_hashes(const char *file_name, uint32_t **hashes) {
    FILE *const fd = fopen(file_name, "r");
    if (!fd) {
        return 0;
    }

    fseek(fd, 0L, SEEK_END);
    const uint32_t file_size = (const uint32_t) ftell(fd);
    rewind(fd);

    const uint32_t hashes_num_est = (file_size + 1) / 33;
    uint32_t hashes_num = 0;

    *hashes = malloc(hashes_num_est*16);

    char buf[128];
    while(fgets(buf, sizeof(buf), fd) != NULL) {
        for (int i=0; i<sizeof(buf); i++) {
            if (buf[i] == '\n' || buf[i] == '\r') {
                buf[i] = 0;
            }
        }
        if (strlen(buf) != 32) {
            fprintf(stderr, "not a hash! (%s)\n", buf);
        }

        if (hashes_num>hashes_num_est) {
            fprintf(stderr, "too many hashes? skipping tail...\n");
            break;
        }

        ascii_to_hash(buf, &((*hashes)[hashes_num*4]));
        hashes_num++;
    }

    fclose(fd);
    return hashes_num;
}
```

Note: also add the missing `fclose(fd)` before the return (currently leaked in the original). This fixes an fd leak but doesn't change behavior.

**Step 3: Remove `read_hashes` from `main.c`**

Delete the `read_hashes` function (lines 73-109). The `#include "hashes.h"` is already there via `gpu_cruncher.h` includes. Add an explicit `#include "hashes.h"` in `main.c` includes if not present.

**Step 4: Remove local `read_hashes` from `kernel_debug.c`**

Delete lines 16-52 (the local `read_hashes` copy) from `kernel_debug.c`. The shared version from `hashes.c` is already linked (same source set in CMakeLists.txt). Keep the local `format_bignum` for now (it's used by `kernel_debug` but we're not extracting it in this plan).

**Step 5: Build and verify**

```bash
cd /Users/alepar/AleCode/anabrute_cl/cmake-build-debug && cmake .. && make
```

Expected: builds cleanly. Both `anabrute` and `kernel_debug` targets link.

**Step 6: Commit**

```
refactor: move read_hashes from main.c to hashes.c/h
```

---

### Task 2: Extract `read_dict` from `main.c` to `dict.c/h`

**Files:**
- Create: `dict.h`
- Create: `dict.c`
- Modify: `main.c` — remove `read_dict`, add `#include "dict.h"`
- Modify: `CMakeLists.txt` — add `dict.c` to source lists

**Step 1: Create `dict.h`**

```c
#ifndef ANABRUTE_DICT_H
#define ANABRUTE_DICT_H

#include "permut_types.h"

int read_dict(const char *filename, char_counts_strings *dict, uint32_t *dict_length, char_counts *seed_phrase);

#endif //ANABRUTE_DICT_H
```

Note: add `filename` parameter (was hardcoded to `"input.dict"` in `main.c`).

**Step 2: Create `dict.c`**

```c
#include "dict.h"

int read_dict(const char *filename, char_counts_strings *dict, uint32_t *dict_length, char_counts *seed_phrase) {
    FILE *dictFile = fopen(filename, "r");
    if (!dictFile) {
        fprintf(stderr, "dict file not found!\n");
        return -1;
    }

    char buf1[100] = {0}, buf2[100] = {0};
    char *buflines[] = {buf1, buf2};
    int lineidx = 0;

    while(fgets(buflines[lineidx], 100, dictFile) != NULL) {
        char *const str = buflines[lineidx];
        const size_t len = strlen(str);
        if (str[len-1] == '\n' || str[len-1] == '\r') {
            str[len-1] = 0;
        }
        if (str[len-2] == '\n' || str[len-2] == '\r') {
            str[len-2] = 0;
        }

        if (strcmp(buflines[0], buflines[1])) {
            lineidx = 1-lineidx;
            if (char_counts_strings_create(str, &dict[*dict_length])) {
                continue;
            }

            if (char_counts_contains(seed_phrase, &dict[*dict_length].counts)) {
                int i;

                for (i=0; i<*dict_length; i++) {
                    if (char_counts_equal(&dict[i].counts, &dict[*dict_length].counts)) {
                        break;
                    }
                }
                if (i==*dict_length) {
                    (*dict_length)++;
                    if (*dict_length > MAX_DICT_SIZE) {
                        fprintf(stderr, "dict overflow! %d\n", *dict_length);
                        fclose(dictFile);
                        return -2;
                    }
                }

                if (char_counts_strings_addstring(&dict[i], str)) {
                    fprintf(stderr, "strings overflow! %d", dict[i].strings_len);
                    fclose(dictFile);
                    return -3;
                }
            }
        }
    }
    fclose(dictFile);
    return 0;
}
```

Note: this is a direct copy of `main.c` lines 9-61, with:
- `filename` parameter instead of hardcoded `"input.dict"`
- `fclose(dictFile)` added to error return paths (fixes fd leak — BUGS.md #10)
- Bugs preserved as-is (line trimming OOB — will be tested and fixed later)

**Step 3: Update `main.c`**

Remove the `read_dict` function (lines 9-61). Add `#include "dict.h"`. Update the call site:

```c
// was: read_dict(dict, &dict_length, &seed_phrase);
read_dict("input.dict", dict, &dict_length, &seed_phrase);
```

**Step 4: Update `CMakeLists.txt`**

Add `dict.c` to both executable source lists:

```cmake
add_executable (anabrute main.c gpu_cruncher.c hashes.c dict.c permut_types.c seedphrase.c fact.c cpu_cruncher.c os.c task_buffers.c)
...
add_executable (kernel_debug kernel_debug.c gpu_cruncher.c hashes.c dict.c permut_types.c seedphrase.c fact.c cpu_cruncher.c os.c task_buffers.c)
```

**Step 5: Build and verify**

```bash
cd /Users/alepar/AleCode/anabrute_cl/cmake-build-debug && cmake .. && make
```

**Step 6: Commit**

```
refactor: extract read_dict from main.c to dict.c/h
```

---

### Task 3: Set up test infrastructure in CMakeLists.txt

**Files:**
- Modify: `CMakeLists.txt` — add test targets, ASan, CTest

**Step 1: Add test infrastructure to `CMakeLists.txt`**

Append to end of file:

```cmake
# === Tests ===
enable_testing()

set(ASAN_FLAGS "-fsanitize=address,undefined -fno-omit-frame-pointer")

# Test: hash parsing (minimal dependencies)
add_executable(test_hash_parsing tests/test_hash_parsing.c hashes.c)
set_property(TARGET test_hash_parsing PROPERTY C_STANDARD 99)
target_include_directories(test_hash_parsing PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_options(test_hash_parsing PRIVATE -g ${ASAN_FLAGS})
target_link_options(test_hash_parsing PRIVATE ${ASAN_FLAGS})
add_test(NAME hash_parsing COMMAND test_hash_parsing)

# Test: dict parsing
add_executable(test_dict_parsing tests/test_dict_parsing.c dict.c permut_types.c seedphrase.c)
set_property(TARGET test_dict_parsing PROPERTY C_STANDARD 99)
target_include_directories(test_dict_parsing PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_options(test_dict_parsing PRIVATE -g ${ASAN_FLAGS})
target_link_options(test_dict_parsing PRIVATE ${ASAN_FLAGS})
add_test(NAME dict_parsing COMMAND test_dict_parsing)

# Test: CPU anagram enumeration
add_executable(test_cpu_enumeration tests/test_cpu_enumeration.c
    cpu_cruncher.c task_buffers.c dict.c permut_types.c seedphrase.c fact.c os.c hashes.c)
set_property(TARGET test_cpu_enumeration PROPERTY C_STANDARD 99)
target_include_directories(test_cpu_enumeration PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_options(test_cpu_enumeration PRIVATE -g ${ASAN_FLAGS})
target_link_options(test_cpu_enumeration PRIVATE ${ASAN_FLAGS})
target_link_libraries(test_cpu_enumeration pthread)
add_test(NAME cpu_enumeration COMMAND test_cpu_enumeration)
```

Note: `ASAN_FLAGS` is a string, so use `PRIVATE` with `target_compile_options`/`target_link_options`. On macOS with Apple Clang, ASan is built in. The `set(ASAN_FLAGS ...)` approach may need to be split — use `separate_arguments` or just pass the flags directly.

Actually, CMake doesn't expand strings in `target_compile_options` the same way. Use:

```cmake
add_compile_options(-g)

add_executable(test_hash_parsing tests/test_hash_parsing.c hashes.c)
set_property(TARGET test_hash_parsing PROPERTY C_STANDARD 99)
target_include_directories(test_hash_parsing PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_options(test_hash_parsing PRIVATE -fsanitize=address -fsanitize=undefined -fno-omit-frame-pointer)
target_link_options(test_hash_parsing PRIVATE -fsanitize=address -fsanitize=undefined)
add_test(NAME hash_parsing COMMAND test_hash_parsing)
```

Repeat the pattern for each test target.

**Step 2: Create empty test stubs**

Create `tests/` directory and placeholder files so CMake doesn't error:

```bash
mkdir -p tests
```

Create `tests/test_hash_parsing.c`:
```c
#include <stdio.h>
int main() { printf("test_hash_parsing: TODO\n"); return 0; }
```

Same for `tests/test_dict_parsing.c` and `tests/test_cpu_enumeration.c`.

**Step 3: Build and verify CTest works**

```bash
cd /Users/alepar/AleCode/anabrute_cl/cmake-build-debug && cmake .. && make
ctest --output-on-failure
```

Expected: 3 tests, all pass (stubs).

**Step 4: Commit**

```
chore: set up CTest infrastructure with ASan for integration tests
```

---

### Task 4: Hash parsing tests + fix bugs #2 and #3

**Files:**
- Modify: `tests/test_hash_parsing.c` — write 3 integration tests
- Modify: `hashes.c` — fix bugs #2 and #3

**Step 1: Write test file**

`tests/test_hash_parsing.c`:

```c
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include "hashes.h"

static void write_file(const char *path, const char *content) {
    FILE *f = fopen(path, "w");
    assert(f && "failed to create test file");
    fputs(content, f);
    fclose(f);
}

/*
 * Test 1: Valid file with 3 known MD5 hashes.
 * Verifies read_hashes returns correct count and parsed values roundtrip.
 */
void test_valid_hashes(void) {
    const char *path = "/tmp/anabrute_test_hashes_valid.txt";
    write_file(path,
        "d41d8cd98f00b204e9800998ecf8427e\n"   // MD5("")
        "0cc175b9c0f1b6a831c399e269772661\n"   // MD5("a")
        "92eb5ffee6ae2fec3ad71c777531578f\n");  // MD5("abc")

    uint32_t *hashes = NULL;
    uint32_t count = read_hashes(path, &hashes);

    assert(count == 3);
    assert(hashes != NULL);

    // Roundtrip: parse → format → compare against original
    char buf[33];
    hash_to_ascii(hashes + 0*4, buf);
    assert(strcmp(buf, "d41d8cd98f00b204e9800998ecf8427e") == 0);

    hash_to_ascii(hashes + 1*4, buf);
    assert(strcmp(buf, "0cc175b9c0f1b6a831c399e269772661") == 0);

    hash_to_ascii(hashes + 2*4, buf);
    assert(strcmp(buf, "92eb5ffee6ae2fec3ad71c777531578f") == 0);

    free(hashes);
    unlink(path);
    printf("  PASS: test_valid_hashes\n");
}

/*
 * Test 2: File with an invalid (non-32-char) line mixed in.
 * BUGS.md #3: invalid lines should be skipped, not parsed.
 * Before fix: count=4 (invalid line counted). After fix: count=3.
 */
void test_invalid_line_skipped(void) {
    const char *path = "/tmp/anabrute_test_hashes_invalid.txt";
    write_file(path,
        "d41d8cd98f00b204e9800998ecf8427e\n"
        "0cc175b9c0f1b6a831c399e269772661\n"
        "not_a_valid_hash\n"
        "92eb5ffee6ae2fec3ad71c777531578f\n");

    uint32_t *hashes = NULL;
    uint32_t count = read_hashes(path, &hashes);

    assert(count == 3 && "invalid line should not be counted");

    // Verify the 3 valid hashes were parsed correctly (invalid line skipped)
    char buf[33];
    hash_to_ascii(hashes + 0*4, buf);
    assert(strcmp(buf, "d41d8cd98f00b204e9800998ecf8427e") == 0);
    hash_to_ascii(hashes + 1*4, buf);
    assert(strcmp(buf, "0cc175b9c0f1b6a831c399e269772661") == 0);
    hash_to_ascii(hashes + 2*4, buf);
    assert(strcmp(buf, "92eb5ffee6ae2fec3ad71c777531578f") == 0);

    free(hashes);
    unlink(path);
    printf("  PASS: test_invalid_line_skipped\n");
}

/*
 * Test 3: Short invalid line makes file size estimate too low,
 * causing more lines than allocated slots.
 * BUGS.md #2 + #3: the short line throws off (file_size+1)/33 estimate,
 * and without the continue, it writes past the allocated buffer.
 * Under ASan: crashes with heap-buffer-overflow before the fix.
 */
void test_short_invalid_line_no_overflow(void) {
    const char *path = "/tmp/anabrute_test_hashes_overflow.txt";
    // File: 33 + 2 + 33 = 68 bytes. Estimate: (68+1)/33 = 2. Actual lines: 3.
    write_file(path,
        "d41d8cd98f00b204e9800998ecf8427e\n"
        "x\n"
        "0cc175b9c0f1b6a831c399e269772661\n");

    uint32_t *hashes = NULL;
    uint32_t count = read_hashes(path, &hashes);

    // After fix: invalid "x" skipped, count=2, no overflow
    assert(count == 2);

    free(hashes);
    unlink(path);
    printf("  PASS: test_short_invalid_line_no_overflow\n");
}

int main(void) {
    printf("test_hash_parsing:\n");
    test_valid_hashes();
    test_invalid_line_skipped();
    test_short_invalid_line_no_overflow();
    printf("All hash parsing tests passed!\n");
    return 0;
}
```

**Step 2: Build and run — expect failures**

```bash
cd /Users/alepar/AleCode/anabrute_cl/cmake-build-debug && cmake .. && make test_hash_parsing
./tests/../test_hash_parsing  # or: ctest -R hash_parsing --output-on-failure
```

Expected:
- `test_valid_hashes`: PASS (no bugs in the happy path)
- `test_invalid_line_skipped`: FAIL — assertion `count == 3` fails because count is 4
- `test_short_invalid_line_no_overflow`: CRASH under ASan (heap-buffer-overflow) or FAIL assertion

**Step 3: Fix bug #3 in `hashes.c` — skip invalid lines**

In `hashes.c`, in the `read_hashes` function, after the `strlen(buf) != 32` warning, add `continue`:

```c
        if (strlen(buf) != 32) {
            fprintf(stderr, "not a hash! (%s)\n", buf);
            continue;
        }
```

**Step 4: Fix bug #2 in `hashes.c` — off-by-one boundary**

In `hashes.c`, change `>` to `>=`:

```c
        if (hashes_num>=hashes_num_est) {
            fprintf(stderr, "too many hashes? skipping tail...\n");
            break;
        }
```

**Step 5: Build and run — all should pass**

```bash
cd /Users/alepar/AleCode/anabrute_cl/cmake-build-debug && make test_hash_parsing
ctest -R hash_parsing --output-on-failure
```

Expected: all 3 tests PASS, no ASan errors.

**Step 6: Commit**

```
fix: skip invalid hash lines and fix off-by-one in read_hashes (BUGS #2, #3)
```

---

### Task 5: Dict parsing tests + fix line trimming bug

**Files:**
- Modify: `tests/test_dict_parsing.c` — write integration tests
- Modify: `dict.c` — fix line trimming OOB (BUGS.md #8, MEDIUM)

**Step 1: Write test file**

`tests/test_dict_parsing.c`:

```c
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include "dict.h"
#include "seedphrase.h"

static void write_file(const char *path, const char *content) {
    FILE *f = fopen(path, "w");
    assert(f && "failed to create test file");
    fputs(content, f);
    fclose(f);
}

/*
 * Test 1: Load a small dict with words from the seed phrase alphabet.
 * Seed phrase is "tyranousplutotwits" (chars: s,t,a,o,i,r,n,p,l,u,y,w).
 */
void test_basic_loading(void) {
    const char *path = "/tmp/anabrute_test_dict_basic.txt";
    write_file(path, "out\nrun\npots\n");

    char_counts seed;
    char_counts_create(seed_phrase_str, &seed);

    char_counts_strings dict[MAX_DICT_SIZE];
    uint32_t dict_length = 0;

    int err = read_dict(path, dict, &dict_length, &seed);
    assert(err == 0);
    assert(dict_length > 0 && "should load at least one word");

    // All loaded words should be containable in the seed phrase
    for (uint32_t i = 0; i < dict_length; i++) {
        assert(char_counts_contains(&seed, &dict[i].counts));
    }

    unlink(path);
    printf("  PASS: test_basic_loading\n");
}

/*
 * Test 2: Words with chars not in seed phrase should be filtered out.
 * "hello" has 'h','e' which are not in the seed phrase mapping.
 */
void test_filtering_invalid_chars(void) {
    const char *path = "/tmp/anabrute_test_dict_filter.txt";
    write_file(path, "hello\nworld\nout\n");

    char_counts seed;
    char_counts_create(seed_phrase_str, &seed);

    char_counts_strings dict[MAX_DICT_SIZE];
    uint32_t dict_length = 0;

    int err = read_dict(path, dict, &dict_length, &seed);
    assert(err == 0);

    // "hello" and "world" have chars outside the seed phrase alphabet
    // Only "out" should survive (o, u, t are all in the seed phrase)
    assert(dict_length == 1);
    assert(dict[0].strings_len == 1);
    assert(strcmp(dict[0].strings[0], "out") == 0);

    unlink(path);
    printf("  PASS: test_filtering_invalid_chars\n");
}

/*
 * Test 3: Anagram-equivalent words should be grouped into one entry.
 * "opts" and "pots" and "stop" and "tops" are all anagrams (same char_counts).
 */
void test_anagram_grouping(void) {
    const char *path = "/tmp/anabrute_test_dict_group.txt";
    write_file(path, "opts\npots\nstop\ntops\nout\n");

    char_counts seed;
    char_counts_create(seed_phrase_str, &seed);

    char_counts_strings dict[MAX_DICT_SIZE];
    uint32_t dict_length = 0;

    int err = read_dict(path, dict, &dict_length, &seed);
    assert(err == 0);

    // "opts"/"pots"/"stop"/"tops" should be 1 entry with 4 strings
    // "out" should be a separate entry
    assert(dict_length == 2);

    // Find the entry with 4 strings (the anagram group)
    int group_idx = (dict[0].strings_len == 4) ? 0 : 1;
    assert(dict[group_idx].strings_len == 4);
    assert(dict[1 - group_idx].strings_len == 1);

    unlink(path);
    printf("  PASS: test_anagram_grouping\n");
}

/*
 * Test 4: Empty lines and single-char lines should not crash.
 * BUGS.md #8: str[len-2] is OOB when len < 2.
 * Under ASan: crashes before fix.
 */
void test_short_lines_no_crash(void) {
    const char *path = "/tmp/anabrute_test_dict_short.txt";
    // Single-char line "a\n" has len=2 after fgets, so str[0]='a',str[1]='\n'
    // After trimming: str[0]='a',str[1]=0. len-2=0, str[0] checked — OK.
    // But a truly empty line "\n" has len=1: str[0]='\n'. str[len-2]=str[-1] — OOB!
    write_file(path, "\nout\n\n");

    char_counts seed;
    char_counts_create(seed_phrase_str, &seed);

    char_counts_strings dict[MAX_DICT_SIZE];
    uint32_t dict_length = 0;

    int err = read_dict(path, dict, &dict_length, &seed);
    assert(err == 0);
    // "out" should still be loaded despite empty lines
    assert(dict_length == 1);

    unlink(path);
    printf("  PASS: test_short_lines_no_crash\n");
}

int main(void) {
    printf("test_dict_parsing:\n");
    test_basic_loading();
    test_filtering_invalid_chars();
    test_anagram_grouping();
    test_short_lines_no_crash();
    printf("All dict parsing tests passed!\n");
    return 0;
}
```

**Step 2: Build and run — expect test 4 to crash under ASan**

```bash
cd /Users/alepar/AleCode/anabrute_cl/cmake-build-debug && make test_dict_parsing
ctest -R dict_parsing --output-on-failure
```

Expected:
- Tests 1-3: PASS
- Test 4 (`test_short_lines_no_crash`): CRASH under ASan — stack-buffer-overflow reading `str[len-2]` when `len < 2`

**Step 3: Fix line trimming in `dict.c`**

Replace the line trimming code in `dict.c`:

```c
        // OLD (buggy):
        if (str[len-1] == '\n' || str[len-1] == '\r') {
            str[len-1] = 0;
        }
        if (str[len-2] == '\n' || str[len-2] == '\r') {
            str[len-2] = 0;
        }

        // NEW (safe):
        if (len > 0 && (str[len-1] == '\n' || str[len-1] == '\r')) {
            str[len-1] = 0;
        }
        if (len > 1 && (str[len-2] == '\n' || str[len-2] == '\r')) {
            str[len-2] = 0;
        }
```

**Step 4: Build and run — all should pass**

```bash
cd /Users/alepar/AleCode/anabrute_cl/cmake-build-debug && make test_dict_parsing
ctest -R dict_parsing --output-on-failure
```

Expected: all 4 tests PASS.

**Step 5: Commit**

```
fix: prevent OOB read in dict line trimming for short lines (BUGS #8)

Also adds integration tests for dict parsing covering filtering,
anagram grouping, and edge cases.
```

---

### Task 6: CPU enumeration test + fix bug #1

**Files:**
- Modify: `tests/test_cpu_enumeration.c` — write integration test
- Modify: `cpu_cruncher.c` — fix `&` to `&&` (BUGS.md #1)

**Step 1: Write test file**

`tests/test_cpu_enumeration.c`:

```c
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "common.h"
#include "cpu_cruncher.h"
#include "dict.h"
#include "fact.h"
#include "seedphrase.h"

/*
 * Helper: decode the initial permutation from a permut_task into a string.
 * Replicates the kernel's string construction logic (permut.cl lines 196-212).
 */
static void decode_task(permut_task *task, char *result) {
    int wcs = 0;
    for (int io = 0; task->offsets[io]; io++) {
        int8_t off = task->offsets[io];
        if (off < 0) {
            off = -off - 1;
        } else {
            off = task->a[off - 1] - 1;
        }
        while (task->all_strs[(uint8_t)off]) {
            result[wcs++] = task->all_strs[(uint8_t)off];
            off++;
        }
        result[wcs++] = ' ';
    }
    if (wcs > 0) wcs--;  // remove trailing space
    result[wcs] = '\0';
}

/*
 * Helper: check if a string (with spaces) is a valid anagram of the seed phrase.
 * Ignores spaces, checks that remaining chars have identical frequency to seed.
 */
static bool is_anagram_of_seed(const char *str) {
    char_counts seed;
    char_counts_create(seed_phrase_str, &seed);

    char_counts candidate;
    memset(&candidate, 0, sizeof(candidate));
    candidate.length = 0;

    for (int i = 0; str[i]; i++) {
        if (str[i] == ' ') continue;
        int idx = char_to_index(str[i]);
        if (idx < 0) return false;
        candidate.counts[idx]++;
        candidate.length++;
    }

    return char_counts_equal(&seed, &candidate);
}

/*
 * Helper: run CPU cruncher with given dict file, collect all tasks.
 * Returns array of tasks (caller must free) and sets *num_tasks.
 */
static permut_task* run_cruncher_with_dict(const char *dict_path, int *num_tasks) {
    char_counts seed;
    char_counts_create(seed_phrase_str, &seed);

    char_counts_strings dict[MAX_DICT_SIZE];
    uint32_t dict_length = 0;
    int err = read_dict(dict_path, dict, &dict_length, &seed);
    assert(err == 0);

    // Build dict_by_char index (same as main.c)
    char_counts_strings* dict_by_char[CHARCOUNT][MAX_DICT_SIZE];
    int dict_by_char_len[CHARCOUNT] = {0};
    for (uint32_t i = 0; i < dict_length; i++) {
        for (int ci = 0; ci < CHARCOUNT; ci++) {
            if (dict[i].counts.counts[ci]) {
                dict_by_char[ci][dict_by_char_len[ci]++] = &dict[i];
                break;
            }
        }
    }

    // Create pipeline
    tasks_buffers tasks_buffs;
    tasks_buffers_create(&tasks_buffs);

    cpu_cruncher_ctx ctx;
    cpu_cruncher_ctx_create(&ctx, 0, 1, &seed, &dict_by_char, dict_by_char_len, &tasks_buffs);

    // Run synchronously (single thread, id=0, step=1 so it covers all entries)
    run_cpu_cruncher_thread(&ctx);
    tasks_buffers_close(&tasks_buffs);

    // Drain all task buffers, collect tasks
    int capacity = 1024;
    permut_task *all_tasks = malloc(capacity * sizeof(permut_task));
    *num_tasks = 0;

    tasks_buffer *buf;
    while (1) {
        tasks_buffers_get_buffer(&tasks_buffs, &buf);
        if (!buf) break;
        for (uint32_t i = 0; i < buf->num_tasks; i++) {
            if (*num_tasks >= capacity) {
                capacity *= 2;
                all_tasks = realloc(all_tasks, capacity * sizeof(permut_task));
            }
            all_tasks[*num_tasks] = buf->permut_tasks[i];
            (*num_tasks)++;
        }
        tasks_buffer_free(buf);
    }

    tasks_buffers_free(&tasks_buffs);
    return all_tasks;
}

static void write_file(const char *path, const char *content) {
    FILE *f = fopen(path, "w");
    assert(f);
    fputs(content, f);
    fclose(f);
}

/*
 * Test 1: Single word that IS the entire seed phrase.
 * Dict = ["tyranousplutotwits"]. Expect 1 task with n=1.
 */
void test_single_word_anagram(void) {
    const char *path = "/tmp/anabrute_test_enum_single.txt";
    write_file(path, "tyranousplutotwits\n");

    int num_tasks;
    permut_task *tasks = run_cruncher_with_dict(path, &num_tasks);

    assert(num_tasks == 1);
    assert(tasks[0].n == 1);

    char decoded[MAX_STR_LENGTH];
    decode_task(&tasks[0], decoded);
    assert(strcmp(decoded, "tyranousplutotwits") == 0);
    assert(is_anagram_of_seed(decoded));

    free(tasks);
    unlink(path);
    printf("  PASS: test_single_word_anagram\n");
}

/*
 * Test 2: Two words that together form the seed phrase.
 * Dict = ["tyranous", "plutotwits"]. Expect 1 task with n=2.
 */
void test_two_word_anagram(void) {
    const char *path = "/tmp/anabrute_test_enum_two.txt";
    write_file(path, "tyranous\nplutotwits\n");

    int num_tasks;
    permut_task *tasks = run_cruncher_with_dict(path, &num_tasks);

    assert(num_tasks == 1);
    assert(tasks[0].n == 2);

    char decoded[MAX_STR_LENGTH];
    decode_task(&tasks[0], decoded);
    assert(is_anagram_of_seed(decoded));

    // Total permutations should be 2! = 2
    assert(fact(tasks[0].n) == 2);

    free(tasks);
    unlink(path);
    printf("  PASS: test_two_word_anagram\n");
}

/*
 * Test 3: Three words that together form the seed phrase.
 * Dict = ["tyranous", "pluto", "twits"]. Expect task(s) with n=3.
 */
void test_three_word_anagram(void) {
    const char *path = "/tmp/anabrute_test_enum_three.txt";
    write_file(path, "tyranous\npluto\ntwits\n");

    int num_tasks;
    permut_task *tasks = run_cruncher_with_dict(path, &num_tasks);

    assert(num_tasks >= 1 && "should find at least one combination");

    // Every task should decode to a valid anagram
    for (int i = 0; i < num_tasks; i++) {
        char decoded[MAX_STR_LENGTH];
        decode_task(&tasks[i], decoded);
        assert(is_anagram_of_seed(decoded));
    }

    // At least one task should have n=3 (all 3 words permutable)
    bool found_n3 = false;
    for (int i = 0; i < num_tasks; i++) {
        if (tasks[i].n == 3) {
            found_n3 = true;
            break;
        }
    }
    assert(found_n3 && "should have a task with n=3");

    free(tasks);
    unlink(path);
    printf("  PASS: test_three_word_anagram\n");
}

/*
 * Test 4: Dict with words that DON'T form any anagram of the seed phrase.
 * Dict = ["out", "run"]. Neither alone nor together exhaust all seed chars.
 * Expect 0 tasks.
 */
void test_no_valid_anagrams(void) {
    const char *path = "/tmp/anabrute_test_enum_none.txt";
    write_file(path, "out\nrun\n");

    int num_tasks;
    permut_task *tasks = run_cruncher_with_dict(path, &num_tasks);

    assert(num_tasks == 0 && "no combination should exhaust the seed phrase");

    free(tasks);
    unlink(path);
    printf("  PASS: test_no_valid_anagrams\n");
}

/*
 * Test 5: Dict with synonym words (same char_counts, different strings).
 * "opts"/"pots"/"stop"/"tops" are all anagrams of each other.
 * Combined with other words to complete the seed phrase, each synonym
 * should generate its own task(s).
 */
void test_synonym_words(void) {
    const char *path = "/tmp/anabrute_test_enum_synonyms.txt";
    // "tyranous" + "pluto" + "twits" = seed phrase
    // "wists" has same char_counts as... hmm, need actual synonyms from seed chars
    // Let's use: "tops" and "stop" and "pots" and "opts" all have {t,o,p,s}
    // We need the rest: seed - {t,o,p,s} = {t,y,r,a,n,o,u,l,u,w,i,s,t}
    // That doesn't simplify to known words easily. Let's use a different approach.
    //
    // Actually, just test that with "pluto"/"tulop" (if tulop were a word)...
    // This is hard to construct from the fixed seed phrase alphabet.
    // Skip complex synonym test; the anagram grouping is already tested in dict_parsing.
    printf("  SKIP: test_synonym_words (covered by dict_parsing anagram grouping test)\n");
}

int main(void) {
    printf("test_cpu_enumeration:\n");
    test_single_word_anagram();
    test_two_word_anagram();
    test_three_word_anagram();
    test_no_valid_anagrams();
    test_synonym_words();
    printf("All CPU enumeration tests passed!\n");
    return 0;
}
```

**Step 2: Build and run — expect all to pass (bug #1 is behaviorally masked)**

```bash
cd /Users/alepar/AleCode/anabrute_cl/cmake-build-debug && make test_cpu_enumeration
ctest -R cpu_enumeration --output-on-failure
```

Expected: all tests PASS. Bug #1 (`&` vs `&&`) doesn't affect behavior because the OOB read hits `remainder->length` which is always >0 at that point (guarded by early return on line 176). The tests establish correctness baselines.

**Step 3: Fix bug #1 in `cpu_cruncher.c` — `&` to `&&`**

In `cpu_cruncher.c` line 180, change:

```c
    // OLD (buggy — bitwise & doesn't short-circuit, reads counts[CHARCOUNT] OOB):
    for (;curchar<CHARCOUNT & !remainder->counts[curchar]; curchar++)

    // NEW (logical && short-circuits, no OOB read):
    for (;curchar<CHARCOUNT && !remainder->counts[curchar]; curchar++)
```

Also remove the now-unnecessary dead code guard on line 181-183:

```c
    // This was dead code (never reached because & still produced correct result)
    // With && fix, the for-loop itself stops at CHARCOUNT, making this unnecessary.
    // But keeping it as a safety check is fine. Remove or keep — your choice.
```

**Step 4: Build and run — all still pass**

```bash
cd /Users/alepar/AleCode/anabrute_cl/cmake-build-debug && make test_cpu_enumeration
ctest --output-on-failure
```

Expected: all tests pass. No behavioral change, but the OOB memory read is eliminated.

**Step 5: Commit**

```
fix: change & to && in cpu_cruncher char-skip loop (BUGS #1)

Bitwise & evaluated both operands even when curchar >= CHARCOUNT,
causing an out-of-bounds read of remainder->counts[]. The read
happened to hit the length field which was always >0, masking the
bug. Logical && short-circuits and avoids the OOB access entirely.
```

---

### Task 7: Final verification — all tests pass

**Step 1: Clean build + run all tests**

```bash
cd /Users/alepar/AleCode/anabrute_cl/cmake-build-debug && cmake .. && make
ctest --output-on-failure -V
```

Expected: 3 tests, all PASS, no ASan errors.

**Step 2: Verify main targets still build**

```bash
make anabrute kernel_debug
```

Expected: both build cleanly.

**Step 3: Final commit if any stragglers**

Verify `git status` is clean (all changes committed in earlier steps).

---

## Summary of Changes

| Bug | File | Fix | Test |
|-----|------|-----|------|
| #1 (HIGH) | `cpu_cruncher.c:180` | `&` → `&&` | `test_cpu_enumeration` (correctness baseline) |
| #2 (HIGH) | `hashes.c` (was `main.c:99`) | `>` → `>=` | `test_hash_parsing::test_short_invalid_line_no_overflow` |
| #3 (HIGH) | `hashes.c` (was `main.c:95`) | add `continue` | `test_hash_parsing::test_invalid_line_skipped` |
| #8 (MED) | `dict.c` (was `main.c:23`) | add `len >` guards | `test_dict_parsing::test_short_lines_no_crash` |
| #10 (MED) | `dict.c` (was `main.c:47`) | add `fclose` on error paths | (fixed during extraction) |
| #9 (MED) | `hashes.c` (was `gpu_cruncher.c:24`) | add `fclose` before return | (fixed during extraction) |
