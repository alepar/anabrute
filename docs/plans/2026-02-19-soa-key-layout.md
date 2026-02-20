# SoA Key Layout Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Convert AVX-512 and AVX2 MD5 cruncher paths from AoS to SoA key layout, add OR-based string construction to AVX-512, and replace gathers with aligned loads.

**Architecture:** Modify `process_task()` in avx_cruncher.c to store keys in SoA format `keys[word_pos][lanes]`. Write a new `construct_string_or_soa()` that writes word images directly into SoA layout at a given lane. Update `md5_check_avx512` and `md5_check_avx2` to load keys with aligned loads instead of gather/set instructions.

**Tech Stack:** C, AVX-512, AVX2, SSE2 intrinsics

---

### Task 1: AVX-512 SoA layout + aligned loads

**Files:**
- Modify: `avx_cruncher.c` (lines 432-607)

**Step 1: Change md5_check_avx512 signature and key loading**

Change `keys[16][16]` parameter from AoS to SoA. Replace gather loop with aligned loads:

```c
static void md5_check_avx512(cruncher_config *cfg,
                              uint32_t keys[16][16],  // keys[word][lane] now
                              int wcs_arr[16], int count) {
    __m512i k[16];
    for (int w = 0; w < 16; w++) {
        k[w] = _mm512_load_si512((__m512i *)keys[w]);
    }
    // ... rest of MD5 unchanged
}
```

**Step 2: Add construct_string_or_soa for AVX-512**

New function that writes OR-based word images into SoA layout at a specific lane:

```c
static int construct_string_or_soa(permut_task *task,
                                    uint32_t keys[][16],  // keys[word][lane]
                                    int lane,
                                    const uint32_t wimg[][11],
                                    const uint8_t *wlen_sp,
                                    int num_offsets) {
    int pos = 0;
    for (int io = 0; io < num_offsets; io++) {
        int8_t off = task->offsets[io];
        int byte_off = (off < 0) ? (-off - 1) : (task->a[off - 1] - 1);
        int idx = pos >> 2;
        int shift = (pos & 3) << 3;
        int len_sp = wlen_sp[byte_off];
        int nw = (len_sp + 3) >> 2;
        if (shift == 0) {
            for (int j = 0; j < nw; j++)
                keys[idx + j][lane] |= wimg[byte_off][j];
        } else {
            for (int j = 0; j < nw; j++) {
                keys[idx + j][lane] |= wimg[byte_off][j] << shift;
                keys[idx + j + 1][lane] |= wimg[byte_off][j] >> (32 - shift);
            }
        }
        pos += len_sp;
    }
    int wcs = pos - 1;
    ((char *)&keys[wcs >> 2][lane])[(wcs & 3)] = (char)0x80;
    keys[14][lane] = (uint32_t)(wcs << 3);
    return wcs;
}
```

**Step 3: Update process_task AVX-512 path**

Replace memcpy string construction with precompute + OR-based SoA construction:

```c
// AVX-512 path in process_task:
uint32_t wimg[MAX_STR_LENGTH][11];
uint8_t wlen_sp[MAX_STR_LENGTH];
int num_offsets;
precompute_word_images(task, wimg, wlen_sp, &num_offsets);

uint32_t keys[16][16] __attribute__((aligned(64)));  // keys[word][lane]
memset(keys, 0, sizeof(keys));
int wcs_arr[16];
int batch = 0;

do {
    wcs_arr[batch] = construct_string_or_soa(task, keys, batch, wimg, wlen_sp, num_offsets);
    batch++;
    if (batch == 16) {
        md5_check_avx512(cfg, keys, wcs_arr, 16);
        batch = 0;
        memset(keys, 0, sizeof(keys));
    }
} while (heap_next(task));

if (batch > 0) {
    // Duplicate last lane into remaining slots
    for (int w = 0; w < 16; w++)
        for (int i = batch; i < 16; i++)
            keys[w][i] = keys[w][batch - 1];
    md5_check_avx512(cfg, keys, wcs_arr, batch);
}
```

**Step 4: Build and test**

Run: `cd build && make -j8 && ctest --output-on-failure`
Expected: all tests pass

**Step 5: Benchmark**

Run: `./bench_avx 1 8 4`
Expected: significant improvement over 30 MAna/s baseline

---

### Task 2: AVX2 SoA layout + aligned loads

**Files:**
- Modify: `avx_cruncher.c` (lines 311-637)

**Step 1: Change md5_check_avx2 signature and key loading**

Replace `_mm256_set_epi32` manual gather with aligned loads. Change keys parameter to SoA layout:

```c
static void md5_check_avx2(cruncher_config *cfg,
                            uint32_t keys[16][8],  // keys[word][lane]
                            int wcs_arr[8], int count) {
    __m256i k[16];
    for (int w = 0; w < 16; w++) {
        k[w] = _mm256_load_si256((__m256i *)keys[w]);
    }
    // ... rest of MD5 unchanged
}
```

**Step 2: Update process_task AVX2 path**

Use precompute + `construct_string_or_soa` writing into `keys[16][8]`:

```c
// AVX2 path in process_task:
uint32_t keys[16][8] __attribute__((aligned(32)));  // keys[word][lane]
memset(keys, 0, sizeof(keys));
int wcs_arr[8];
int batch = 0;

do {
    wcs_arr[batch] = construct_string_or_soa(task, keys, batch, wimg, wlen_sp, num_offsets);
    batch++;
    if (batch == 8) {
        md5_check_avx2(cfg, keys, wcs_arr, 8);
        batch = 0;
        memset(keys, 0, sizeof(keys));
    }
} while (heap_next(task));
```

**Step 3: Build, test, benchmark**

Run: `cd build && make -j8 && ctest --output-on-failure && ./bench_avx 1 8 4`

---

### Task 3: Cleanup and final verification

**Step 1: Remove dead code**

Remove old `construct_string` function if no longer used (scalar path may still need it).

**Step 2: End-to-end test**

Run: `./anabrute` with real input files, verify hashes are found correctly.

**Step 3: Profile**

Run: `perf record ./bench_avx 1 8 4` and verify gathers are eliminated.
