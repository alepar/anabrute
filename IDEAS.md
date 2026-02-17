# IDEAS.md — Performance Optimization Design Notes

## Status Legend
- **DONE** — implemented and merged
- **TODO** — not yet implemented
- **REJECTED** — tested and found ineffective or worse

---

## GPU Kernel Optimizations (OpenCL + Metal)

### GPU-1. Precompute Word Lengths Once Per Task (DONE — Metal)

Both `permut.cl` and `permut.metal` recompute `while(all_strs[off])` strlen for every word on every permutation. For n=4 (24 permutations), that's 96 strlen calls reduced to 4. Saves ALU + reduces branch divergence.

### GPU-2. Replace PUTCHAR with Direct Byte Writes (DONE — Metal)

- **OpenCL**: Line 18 has `#pragma OPENCL EXTENSION cl_khr_byte_addressable_store : disable` — explicitly disabling byte writes. Enable it and use `((uchar*)key)[wcs] = val` to eliminate 5 ALU ops per byte.
- **Metal**: Supports byte writes natively. Use `((thread uint8_t*)key)[wcs] = val`.

### GPU-3. Hoist Key Zeroing (DONE — Metal)

Both kernels zero `key[16]` every permutation. All permutations have the same total string length, so zero once before the loop. Saves 16 register writes × (n!-1) permutations.

### GPU-4. Target Hashes in Local Memory (DONE — Metal)

Every work item reads `hashes[]` from global memory for every permutation. Copy to `__local` (OpenCL) / threadgroup (Metal) memory once per work group. Local memory bandwidth is ~10-50x higher.

### GPU-5. Persistent OpenCL Buffers (TODO, MEDIUM)

`krnl_permut_create` allocates a new `cl_mem` buffer every dispatch. Pre-allocate at max size and reuse with `clEnqueueWriteBuffer`. Already done for Metal (`buf_tasks`).

### GPU-6. Batch-by-N GPU Dispatch (TODO, HIGH)

CPU-side batch-by-N is done. GPU side still uses fixed `MAX_ITERS_IN_KERNEL_TASK=512`. Set `iters_per_task = fact(n)` per batch so every task completes in one dispatch. Eliminates re-dispatch overhead and task state readback.

### GPU-7. Early-Exit Hash Comparison (DONE — Metal)

Unroll the hash comparison loop to make the fast path (no match on hash[0]) explicit with `continue` instead of inner loop + break.

### GPU-8. Skip Last 3 MD5 Rounds (DONE — Metal)

In MD5's 64 rounds, variable `a` (hash[0]) is last modified in round 60. Compute only 61 rounds, check `a + H0` against target hash[0]. Only compute rounds 61-63 on a match. Since matches are ~1 in 2^32, this saves 3 rounds for essentially 100% of hashes (~4.7% MD5 savings). On GPU all threads in a warp skip together since no thread will match. **Result: 2.5→2.7 GAna/s (~8% improvement).** Source: penartur5 forum thread optimization.

### GPU-9. OR-Based String Construction (REJECTED — Metal, TODO — AVX)

Precompute each word's bytes as packed uint32s, OR them into the key buffer at the correct bit offset. Eliminates per-byte writes in the hot loop. penartur5 reported 2.5x speedup on CPU/AVX string assembly. **Result on Metal: 4x SLOWER** (600 MAna/s vs 2.5 GAna/s baseline). The precomputed wimg arrays cause massive register spilling on GPU — same root cause as GPU-11. Metal compiler generates superior code for direct byte writes. **May still benefit AVX cruncher** where L1 cache is plentiful and SIMD OR is native.

### GPU-10. Rotate Intrinsic in MD5 STEP (REJECTED — Metal)

Replace manual `(a << s) | (a >> (32-s))` with Metal's `rotate(a, s)`. **Result: ~1.4% improvement, within noise.** Metal compiler already recognizes and optimizes the shift+OR rotate pattern.

### GPU-11. uint32 Accumulator String Construction (REJECTED — Metal)

Shift+OR bytes into a uint32 accumulator, flush every 4 bytes. **Result: 2x SLOWER** (1.1 GAna/s vs 2.5 GAna/s baseline). Metal compiler generates better code for direct byte writes than manual shift+OR with per-byte branching.

### GPU-12. Double-Buffered Metal Dispatches (REJECTED — Metal)

Two pre-allocated MTLBuffers, overlap memcpy with GPU execution. **Result: no measurable improvement.** Apple Silicon unified memory makes memcpy nearly free, nothing to overlap.

---

## CPU Enumeration Optimizations

### CPU-1. SIMD `char_counts_subtract` (TODO, HIGH)

The hottest function in the search. Called millions of times per thread in `recurse_dict_words`. With `CHARCOUNT=12`, the entire `counts[12]` array fits in a single 128-bit SSE/NEON register. Replace the scalar loop with:

```c
__m128i from_v = _mm_loadu_si128((__m128i*)from->counts);
__m128i what_v = _mm_loadu_si128((__m128i*)what->counts);
__m128i saturated = _mm_subs_epu8(from_v, what_v);
__m128i real_sub  = _mm_sub_epi8(from_v, what_v);
if (_mm_movemask_epi8(_mm_xor_si128(saturated, real_sub)) & 0xFFF) return false;
_mm_storeu_si128((__m128i*)from->counts, saturated);
```

Prerequisite: pad `char_counts.counts` to 16 bytes (currently 12).

### CPU-2. Atomic Work Stealing (TODO, MEDIUM)

Strided parallelization at the top level causes imbalanced work. Use an atomic counter for dynamic work stealing:

```c
int i = __sync_fetch_and_add(&next_l0_index, 1);
if (i >= dict_by_char_len[curchar]) break;
```

### CPU-3. `char_counts_copy` as memcpy (TODO, LOW)

Element-by-element copy for 13 bytes → `memcpy(dst, src, sizeof(char_counts))`.

### CPU-4. Cache strlen in `recurse_string_combs` (TODO, LOW)

```c
for (int j=0; j<=strlen(scs[i].str); j++) { ... }
```
`strlen` recomputed every iteration. Use `memcpy` instead.

### CPU-5. Pre-sort Dictionary by Descending Length (TODO, LOW-MEDIUM)

Sort entries within each character group by descending `counts.length`. Longer words hit the `MAX_WORD_LENGTH` cutoff sooner, pruning the search tree earlier.

### CPU-6. Precompute Compatibility Matrix (TODO, SPECULATIVE)

For each pair of dictionary entries, check if `counts[i] + counts[j] <= seed_phrase`. Store as bitset. During recursion, intersect candidates with bitset.

### CPU-7. Bit-Packed char_counts Overflow Check (TODO, MEDIUM)

Pack all character counts into a single uint32/uint64 with sentinel bits per letter. Each letter gets ceil(log2(max_count+1))+1 bits; the extra bit detects overflow. Adding a word's packed counts to the current state and AND-ing against a mask checks all letters for overflow in one operation — no per-letter comparison loop. For this phrase (max count 4, 12 unique chars) only 36 bits needed — fits in a single uint64. Source: Nine17/Loks forum thread.

### CPU-8. Duplicate Permutation Elimination (TODO, HIGH)

When multiple dictionary words share the same character vector, we generate and MD5-hash all n! permutations including duplicates. For example, words with identical char_counts produce identical anagram strings when swapped. Generating only unique permutations avoids wasted MD5 cycles. Impact grows with word count — at 7 words with duplicate vectors, savings can be significant. Source: penartur5/DarkGray/alepar forum thread.

---

## Implemented Optimizations

### DONE: AVX-512 MD5 (16-lane) + memcpy String Construction
Replaced PUTCHAR_SCALAR with memcpy-based string construction. Added `md5_avx512.h` with ternarylogic, native rotate, hardware gather. 103→149 M/s.

### DONE: SIMD Early-Exit Hash Check + Precomputed Word Lengths + Inlined String Build
`check_hashes_avx512()` uses `_mm512_cmpeq_epi32_mask` for early exit. Precomputed `wlen[]` array. Inlined string construction in process_task. 149→224 M/s.

### DONE: Hoist memset Out of Permutation Loop
Zero `keys[16][16]` once upfront instead of per-permutation. Correctness-sound since all permutations have constant total string length. Performance-neutral but cleaner code.

### DONE: Batch-by-N Task Dispatch (CPU side)
Per-N `local_buffers[]` in `cpu_cruncher.c`. Each `tasks_buffer` contains only tasks with the same `n` value.

### DONE: Buffer Pool / Free-List
`tasks_buffers_obtain()` / `tasks_buffers_recycle()` in `task_buffers.c`. Eliminates 24MB calloc/free churn after warmup.

### DONE: Metal Cruncher Backend
Full Metal compute shader port with pre-allocated reusable task buffer.

### DONE: CPU/AVX Cruncher Backend
Scalar + AVX2 (8-wide) + AVX-512 (16-wide) MD5 paths with compile-time dispatch.

---

## Rejected Optimizations

### REJECTED: Incremental String Construction (Heap's Swap Delta)
Maintain a template buffer, update only the swapped region after each Heap's step, memcpy template to key lane. **Result: 35-40% regression** (140M/s vs 224M/s). With 16-wide SIMD batching, each lane needs its own complete key buffer. Template update (~15 bytes) + copy to lane (~26 bytes) = ~41 bytes per permutation vs ~30 bytes for direct construction.

### REJECTED: Factoradic kth_permutation (Index-Based Permutation)
Generate the k-th permutation directly via factoradic decomposition instead of sequential Heap's. **Result: 15% regression** (192M/s vs 224M/s). Division/modulo per level + element-shifting inner loop outweighs the benefit of eliminating sequential dependency.

### REJECTED: MD5-Only GPU (CPU Generates Full Strings)
Destroys bandwidth advantage. `permut_task` (96 bytes) encodes up to fact(n) candidates. Sending raw strings would be 12,600x more data.

### REJECTED: Apple Neural Engine for MD5
ANE is a matrix multiply accelerator with no native bitwise ops, rotations, or 32-bit integer arithmetic.

---

## Alternative Architectures

### Metal GPU (DONE)
Ported to Metal compute shaders. Unified memory eliminates PCIe overhead. See `metal_cruncher.m`.

### AVX2/AVX-512 CPU-Only (DONE)
Eliminates GPU pipeline entirely. Each CPU thread does recursive backtracking + SIMD MD5. See `avx_cruncher.c`.

See `BUGS.md` for known bugs.
