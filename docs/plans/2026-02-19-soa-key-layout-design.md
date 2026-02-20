# SoA Key Layout + OR-Based Construction for AVX-512 and AVX2

## Problem

The AVX-512 MD5 path runs at ~30 MAna/s per thread on Ryzen 9700X. Profiling reveals two dominant bottlenecks:

1. **String construction (50% of total time)**: The AVX-512 path uses memcpy-based byte-by-byte string construction while the AVX2 path already uses the faster OR-based approach. Even the AVX2 OR-based path still writes to AoS layout and pays a manual gather cost.

2. **Key transposition (12% of total time)**: Both paths store keys as `keys[lane][word]` (AoS). SIMD MD5 needs `keys[word][lane]` (SoA). AVX-512 uses `vpgatherdd` (~12 cycles/gather on Zen 4, 16 gathers = ~192 cycles). AVX2 uses `_mm256_set_epi32` (8 scalar loads + shuffle per word, 16 words).

## Solution

### Change 1: SoA key layout

Change key arrays from `keys[lanes][16]` to `keys[16][lanes]` where first index is word position, second is lane.

**AVX-512 (16 lanes):**
```c
// Before: keys[16][16] = keys[lane][word], requires gather
k[w] = _mm512_i32gather_epi32(indices, base, 4);

// After: keys[16][16] = keys[word][lane], aligned load
k[w] = _mm512_load_si512(&keys[w][0]);
```

**AVX2 (8 lanes):**
```c
// Before: keys[8][16] = keys[lane][word], manual set
k[w] = _mm256_set_epi32(keys[7][w], ..., keys[0][w]);

// After: keys[16][8] = keys[word][lane], aligned load
k[w] = _mm256_load_si256((__m256i *)&keys[w][0]);
```

### Change 2: OR-based construction into SoA layout

Port `construct_string_or` to write directly into the SoA layout. The function signature changes from writing a single `key[16]` (one lane's 64-byte block) to writing into `keys[word][lane]` at a specific lane index.

```c
// For each lane in batch:
//   for each word in permutation order:
//     OR wimg[word][j] into keys[idx + j][lane]  (with bit shift)
```

### Change 3: Eliminate redundant zeroing

Zero the SoA keys array once per batch (before filling 16/8 lanes), not per lane. Only zero words 0 through `max_words_needed` (typically ~10 of 16).

## Expected Impact

- Gather elimination: ~12% total time saved
- OR construction for AVX-512: ~30-40% total time saved
- Combined estimate: **2-3x throughput** (30 → 60-90 MAna/s per thread)
- AVX2 also benefits from aligned loads vs `_mm256_set_epi32`

## Files Modified

| File | Change |
|------|--------|
| `avx_cruncher.c` | SoA layout for both paths, `construct_string_or_soa` function, aligned loads in `md5_check_avx512` and `md5_check_avx2` |

## Verification

1. `bench_avx` before → baseline
2. Apply changes
3. `ctest --output-on-failure` → all tests pass
4. `bench_avx` after → measure improvement
5. `perf record` → verify gathers gone
