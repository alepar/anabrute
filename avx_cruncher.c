#include "avx_cruncher.h"
#include "os.h"
#include "task_buffers.h"
#include "md5_avx2.h"
#include "md5_avx512.h"

#include <string.h>
#include <stdio.h>

#if defined(__x86_64__) || defined(_M_AMD64)
static bool cpu_has_avx512f(void) {
    return __builtin_cpu_supports("avx512f");
}
#endif

/* Which SIMD path to use for MD5 hashing */
typedef enum { SIMD_AVX512, SIMD_AVX2, SIMD_SCALAR } simd_mode;

/* Context for one CPU cruncher thread */
typedef struct {
    cruncher_config *cfg;
    simd_mode mode;
    volatile bool is_running;
    volatile uint64_t consumed_bufs;
    volatile uint64_t consumed_anas;
    uint64_t task_time_start;
    uint64_t task_time_end;
} avx_cruncher_ctx;

/* ---------- PUTCHAR_SCALAR: same byte-packing as the OpenCL kernel's PUTCHAR ---------- */
#define PUTCHAR_SCALAR(buf, index, val) \
    (buf)[(index) >> 2] = ((buf)[(index) >> 2] & ~(0xffU << (((index) & 3) << 3))) + ((uint32_t)(val) << (((index) & 3) << 3))

/* ---------- Scalar MD5 — identical algorithm to kernels/permut.cl md5() ---------- */

#define F(x, y, z) ((z) ^ ((x) & ((y) ^ (z))))
#define G(x, y, z) ((y) ^ ((z) & ((x) ^ (y))))
#define H(x, y, z) ((x) ^ (y) ^ (z))
#define I(x, y, z) ((y) ^ ((x) | ~(z)))

#define STEP(f, a, b, c, d, x, t, s) \
    (a) += f((b), (c), (d)) + (x) + (t); \
    (a) = (((a) << (s)) | (((a) & 0xffffffff) >> (32 - (s)))); \
    (a) += (b);

#define GET(i) (key[(i)])

static void md5_scalar(const uint32_t *key, uint32_t *hash) {
    uint32_t a, b, c, d;

    a = 0x67452301;
    b = 0xefcdab89;
    c = 0x98badcfe;
    d = 0x10325476;

    /* Round 1 */
    STEP(F, a, b, c, d, GET(0), 0xd76aa478, 7)
    STEP(F, d, a, b, c, GET(1), 0xe8c7b756, 12)
    STEP(F, c, d, a, b, GET(2), 0x242070db, 17)
    STEP(F, b, c, d, a, GET(3), 0xc1bdceee, 22)
    STEP(F, a, b, c, d, GET(4), 0xf57c0faf, 7)
    STEP(F, d, a, b, c, GET(5), 0x4787c62a, 12)
    STEP(F, c, d, a, b, GET(6), 0xa8304613, 17)
    STEP(F, b, c, d, a, GET(7), 0xfd469501, 22)
    STEP(F, a, b, c, d, GET(8), 0x698098d8, 7)
    STEP(F, d, a, b, c, GET(9), 0x8b44f7af, 12)
    STEP(F, c, d, a, b, GET(10), 0xffff5bb1, 17)
    STEP(F, b, c, d, a, GET(11), 0x895cd7be, 22)
    STEP(F, a, b, c, d, GET(12), 0x6b901122, 7)
    STEP(F, d, a, b, c, GET(13), 0xfd987193, 12)
    STEP(F, c, d, a, b, GET(14), 0xa679438e, 17)
    STEP(F, b, c, d, a, GET(15), 0x49b40821, 22)

    /* Round 2 */
    STEP(G, a, b, c, d, GET(1), 0xf61e2562, 5)
    STEP(G, d, a, b, c, GET(6), 0xc040b340, 9)
    STEP(G, c, d, a, b, GET(11), 0x265e5a51, 14)
    STEP(G, b, c, d, a, GET(0), 0xe9b6c7aa, 20)
    STEP(G, a, b, c, d, GET(5), 0xd62f105d, 5)
    STEP(G, d, a, b, c, GET(10), 0x02441453, 9)
    STEP(G, c, d, a, b, GET(15), 0xd8a1e681, 14)
    STEP(G, b, c, d, a, GET(4), 0xe7d3fbc8, 20)
    STEP(G, a, b, c, d, GET(9), 0x21e1cde6, 5)
    STEP(G, d, a, b, c, GET(14), 0xc33707d6, 9)
    STEP(G, c, d, a, b, GET(3), 0xf4d50d87, 14)
    STEP(G, b, c, d, a, GET(8), 0x455a14ed, 20)
    STEP(G, a, b, c, d, GET(13), 0xa9e3e905, 5)
    STEP(G, d, a, b, c, GET(2), 0xfcefa3f8, 9)
    STEP(G, c, d, a, b, GET(7), 0x676f02d9, 14)
    STEP(G, b, c, d, a, GET(12), 0x8d2a4c8a, 20)

    /* Round 3 */
    STEP(H, a, b, c, d, GET(5), 0xfffa3942, 4)
    STEP(H, d, a, b, c, GET(8), 0x8771f681, 11)
    STEP(H, c, d, a, b, GET(11), 0x6d9d6122, 16)
    STEP(H, b, c, d, a, GET(14), 0xfde5380c, 23)
    STEP(H, a, b, c, d, GET(1), 0xa4beea44, 4)
    STEP(H, d, a, b, c, GET(4), 0x4bdecfa9, 11)
    STEP(H, c, d, a, b, GET(7), 0xf6bb4b60, 16)
    STEP(H, b, c, d, a, GET(10), 0xbebfbc70, 23)
    STEP(H, a, b, c, d, GET(13), 0x289b7ec6, 4)
    STEP(H, d, a, b, c, GET(0), 0xeaa127fa, 11)
    STEP(H, c, d, a, b, GET(3), 0xd4ef3085, 16)
    STEP(H, b, c, d, a, GET(6), 0x04881d05, 23)
    STEP(H, a, b, c, d, GET(9), 0xd9d4d039, 4)
    STEP(H, d, a, b, c, GET(12), 0xe6db99e5, 11)
    STEP(H, c, d, a, b, GET(15), 0x1fa27cf8, 16)
    STEP(H, b, c, d, a, GET(2), 0xc4ac5665, 23)

    /* Round 4 */
    STEP(I, a, b, c, d, GET(0), 0xf4292244, 6)
    STEP(I, d, a, b, c, GET(7), 0x432aff97, 10)
    STEP(I, c, d, a, b, GET(14), 0xab9423a7, 15)
    STEP(I, b, c, d, a, GET(5), 0xfc93a039, 21)
    STEP(I, a, b, c, d, GET(12), 0x655b59c3, 6)
    STEP(I, d, a, b, c, GET(3), 0x8f0ccc92, 10)
    STEP(I, c, d, a, b, GET(10), 0xffeff47d, 15)
    STEP(I, b, c, d, a, GET(1), 0x85845dd1, 21)
    STEP(I, a, b, c, d, GET(8), 0x6fa87e4f, 6)
    STEP(I, d, a, b, c, GET(15), 0xfe2ce6e0, 10)
    STEP(I, c, d, a, b, GET(6), 0xa3014314, 15)
    STEP(I, b, c, d, a, GET(13), 0x4e0811a1, 21)
    STEP(I, a, b, c, d, GET(4), 0xf7537e82, 6)
    STEP(I, d, a, b, c, GET(11), 0xbd3af235, 10)
    STEP(I, c, d, a, b, GET(2), 0x2ad7d2bb, 15)
    STEP(I, b, c, d, a, GET(9), 0xeb86d391, 21)

    hash[0] = a + 0x67452301;
    hash[1] = b + 0xefcdab89;
    hash[2] = c + 0x98badcfe;
    hash[3] = d + 0x10325476;
}

#undef F
#undef G
#undef H
#undef I
#undef STEP
#undef GET

/* ---------- String construction from permut_task ---------- */

/*
 * Constructs the candidate string from a permut_task into a uint32_t[16] key buffer
 * suitable for MD5. Applies MD5 padding. Returns the string length (excluding padding).
 *
 * Uses memcpy instead of byte-at-a-time PUTCHAR_SCALAR for ~4-8x faster string
 * construction. The key buffer is pre-zeroed so MD5 padding bytes beyond the string
 * are already correct.
 *
 * Offset resolution logic (matches OpenCL kernel exactly):
 *   - Iterate offsets[] until hitting 0
 *   - Positive offset: look up a[offset-1] to get 1-based byte offset in all_strs, subtract 1
 *   - Negative offset: use (-offset)-1 as byte offset
 *   - Copy characters, add space between words
 *   - Strip trailing space, add 0x80 padding byte, write bit length at offset 56-57
 */
static int construct_string(permut_task *task, uint32_t *key) {
    memset(key, 0, 64);
    char *dst = (char *)key;
    int wcs = 0;
    for (int io = 0; task->offsets[io]; io++) {
        int8_t off = task->offsets[io];
        if (off < 0) {
            off = -off - 1;
        } else {
            off = task->a[off - 1] - 1;
        }

        /* Add space before word (except first) */
        if (wcs > 0) {
            dst[wcs++] = ' ';
        }

        /* Find word length and copy with memcpy */
        const char *word = &task->all_strs[(uint8_t)off];
        int len = 0;
        while (word[len]) len++;
        memcpy(dst + wcs, word, len);
        wcs += len;
    }

    /* MD5 padding */
    dst[wcs] = (char)0x80;
    dst[56] = (char)(wcs << 3);
    dst[57] = (char)(wcs >> 5);
    return wcs;
}

/* ---------- Heap's permutation algorithm ---------- */

/*
 * Advance Heap's permutation algorithm to the next permutation.
 * Returns true if a new permutation was generated, false if exhausted.
 *
 * Uses a tmp-variable swap (functionally equivalent to the OpenCL kernel's XOR swap).
 */
static bool heap_next(permut_task *task) {
    while (task->i < task->n) {
        if (task->c[task->i] < task->i) {
            if (task->i % 2 == 0) {
                uint8_t tmp = task->a[0];
                task->a[0] = task->a[task->i];
                task->a[task->i] = tmp;
            } else {
                uint8_t tmp = task->a[task->c[task->i]];
                task->a[task->c[task->i]] = task->a[task->i];
                task->a[task->i] = tmp;
            }
            task->c[task->i]++;
            task->i = 0;
            return true;
        } else {
            task->c[task->i] = 0;
            task->i++;
        }
    }
    return false;
}

/* ---------- GPU-9: OR-based string construction ---------- */

/*
 * Precompute each word's bytes as packed uint32s with trailing space.
 * Indexed by byte offset in all_strs. Only computes for offsets actually used.
 */
static void precompute_word_images(permut_task *task,
                                    uint32_t wimg[][11],
                                    uint8_t *wlen_sp,
                                    int *num_offsets_out) {
    memset(wlen_sp, 0, MAX_STR_LENGTH);
    int num_offsets = 0;
    for (int io = 0; task->offsets[io]; io++) {
        int8_t off = task->offsets[io];
        int byte_off = (off < 0) ? (-off - 1) : (task->a[off - 1] - 1);
        if (wlen_sp[byte_off] == 0) {
            memset(wimg[byte_off], 0, 11 * sizeof(uint32_t));
            const char *word = &task->all_strs[byte_off];
            int len = 0;
            while (word[len]) {
                wimg[byte_off][len >> 2] |=
                    ((uint32_t)(uint8_t)word[len]) << ((len & 3) << 3);
                len++;
            }
            /* trailing space */
            wimg[byte_off][len >> 2] |= ((uint32_t)' ') << ((len & 3) << 3);
            wlen_sp[byte_off] = len + 1;
        }
        num_offsets = io + 1;
    }
    *num_offsets_out = num_offsets;
}

/*
 * OR-based string construction: places precomputed word images into the key
 * buffer using uint32 OR operations at the correct bit offset. Eliminates
 * per-permutation strlen and separate space writes.
 */
static int construct_string_or(permut_task *task, uint32_t *key,
                                const uint32_t wimg[][11],
                                const uint8_t *wlen_sp,
                                int num_offsets) {
    memset(key, 0, 64);
    int pos = 0;
    for (int io = 0; io < num_offsets; io++) {
        int8_t off = task->offsets[io];
        int byte_off;
        if (off < 0) byte_off = -off - 1;
        else byte_off = task->a[off - 1] - 1;

        int idx = pos >> 2;
        int shift = (pos & 3) << 3;
        int len_sp = wlen_sp[byte_off];
        int nw = (len_sp + 3) >> 2;

        if (shift == 0) {
            for (int j = 0; j < nw; j++)
                key[idx + j] |= wimg[byte_off][j];
        } else {
            for (int j = 0; j < nw; j++) {
                key[idx + j] |= wimg[byte_off][j] << shift;
                key[idx + j + 1] |= wimg[byte_off][j] >> (32 - shift);
            }
        }
        pos += len_sp;
    }
    int wcs = pos - 1; /* remove trailing space */
    char *dst = (char *)key;
    dst[wcs] = (char)0x80;  /* overwrite trailing space with MD5 padding */
    dst[56] = (char)(wcs << 3);
    dst[57] = (char)(wcs >> 5);
    return wcs;
}

/* ---------- Process one task (all permutations) ---------- */

static void check_hashes(cruncher_config *cfg, uint32_t *hash, uint32_t *key, int wcs) {
    for (uint32_t ih = 0; ih < cfg->hashes_num; ih++) {
        if (hash[0] == cfg->hashes[4 * ih] &&
            hash[1] == cfg->hashes[4 * ih + 1] &&
            hash[2] == cfg->hashes[4 * ih + 2] &&
            hash[3] == cfg->hashes[4 * ih + 3]) {
            PUTCHAR_SCALAR(key, wcs, 0);
            memcpy(cfg->hashes_reversed + ih * MAX_STR_LENGTH / 4,
                   key, MAX_STR_LENGTH);
        }
    }
}

#if defined(__x86_64__) || defined(_M_AMD64)
/*
 * Combined AVX2 MD5 (8-lane) + early-exit hash check.
 * Computes only 61 of 64 MD5 rounds. hash[0] (a) is finalized after round 60.
 * SIMD-checks a against all targets; skips last 3 rounds (~5% MD5 savings)
 * for the ~100% case where no hash[0] matches.
 */
static void md5_check_avx2(cruncher_config *cfg,
                            uint32_t keys[16][8], int wcs_arr[8], int count) {
    __m256i k[16];
    for (int w = 0; w < 16; w++) {
        k[w] = _mm256_load_si256((__m256i *)keys[w]);
    }

    __m256i a = _mm256_set1_epi32(0x67452301);
    __m256i b = _mm256_set1_epi32((int32_t)0xefcdab89);
    __m256i c = _mm256_set1_epi32((int32_t)0x98badcfe);
    __m256i d = _mm256_set1_epi32(0x10325476);

    /* Round 1 */
    MD5_AVX2_STEP(MD5_AVX2_F, a, b, c, d, k[ 0], 0xd76aa478,  7);
    MD5_AVX2_STEP(MD5_AVX2_F, d, a, b, c, k[ 1], 0xe8c7b756, 12);
    MD5_AVX2_STEP(MD5_AVX2_F, c, d, a, b, k[ 2], 0x242070db, 17);
    MD5_AVX2_STEP(MD5_AVX2_F, b, c, d, a, k[ 3], 0xc1bdceee, 22);
    MD5_AVX2_STEP(MD5_AVX2_F, a, b, c, d, k[ 4], 0xf57c0faf,  7);
    MD5_AVX2_STEP(MD5_AVX2_F, d, a, b, c, k[ 5], 0x4787c62a, 12);
    MD5_AVX2_STEP(MD5_AVX2_F, c, d, a, b, k[ 6], 0xa8304613, 17);
    MD5_AVX2_STEP(MD5_AVX2_F, b, c, d, a, k[ 7], 0xfd469501, 22);
    MD5_AVX2_STEP(MD5_AVX2_F, a, b, c, d, k[ 8], 0x698098d8,  7);
    MD5_AVX2_STEP(MD5_AVX2_F, d, a, b, c, k[ 9], 0x8b44f7af, 12);
    MD5_AVX2_STEP(MD5_AVX2_F, c, d, a, b, k[10], 0xffff5bb1, 17);
    MD5_AVX2_STEP(MD5_AVX2_F, b, c, d, a, k[11], 0x895cd7be, 22);
    MD5_AVX2_STEP(MD5_AVX2_F, a, b, c, d, k[12], 0x6b901122,  7);
    MD5_AVX2_STEP(MD5_AVX2_F, d, a, b, c, k[13], 0xfd987193, 12);
    MD5_AVX2_STEP(MD5_AVX2_F, c, d, a, b, k[14], 0xa679438e, 17);
    MD5_AVX2_STEP(MD5_AVX2_F, b, c, d, a, k[15], 0x49b40821, 22);
    /* Round 2 */
    MD5_AVX2_STEP(MD5_AVX2_G, a, b, c, d, k[ 1], 0xf61e2562,  5);
    MD5_AVX2_STEP(MD5_AVX2_G, d, a, b, c, k[ 6], 0xc040b340,  9);
    MD5_AVX2_STEP(MD5_AVX2_G, c, d, a, b, k[11], 0x265e5a51, 14);
    MD5_AVX2_STEP(MD5_AVX2_G, b, c, d, a, k[ 0], 0xe9b6c7aa, 20);
    MD5_AVX2_STEP(MD5_AVX2_G, a, b, c, d, k[ 5], 0xd62f105d,  5);
    MD5_AVX2_STEP(MD5_AVX2_G, d, a, b, c, k[10], 0x02441453,  9);
    MD5_AVX2_STEP(MD5_AVX2_G, c, d, a, b, k[15], 0xd8a1e681, 14);
    MD5_AVX2_STEP(MD5_AVX2_G, b, c, d, a, k[ 4], 0xe7d3fbc8, 20);
    MD5_AVX2_STEP(MD5_AVX2_G, a, b, c, d, k[ 9], 0x21e1cde6,  5);
    MD5_AVX2_STEP(MD5_AVX2_G, d, a, b, c, k[14], 0xc33707d6,  9);
    MD5_AVX2_STEP(MD5_AVX2_G, c, d, a, b, k[ 3], 0xf4d50d87, 14);
    MD5_AVX2_STEP(MD5_AVX2_G, b, c, d, a, k[ 8], 0x455a14ed, 20);
    MD5_AVX2_STEP(MD5_AVX2_G, a, b, c, d, k[13], 0xa9e3e905,  5);
    MD5_AVX2_STEP(MD5_AVX2_G, d, a, b, c, k[ 2], 0xfcefa3f8,  9);
    MD5_AVX2_STEP(MD5_AVX2_G, c, d, a, b, k[ 7], 0x676f02d9, 14);
    MD5_AVX2_STEP(MD5_AVX2_G, b, c, d, a, k[12], 0x8d2a4c8a, 20);
    /* Round 3 */
    MD5_AVX2_STEP(MD5_AVX2_H, a, b, c, d, k[ 5], 0xfffa3942,  4);
    MD5_AVX2_STEP(MD5_AVX2_H, d, a, b, c, k[ 8], 0x8771f681, 11);
    MD5_AVX2_STEP(MD5_AVX2_H, c, d, a, b, k[11], 0x6d9d6122, 16);
    MD5_AVX2_STEP(MD5_AVX2_H, b, c, d, a, k[14], 0xfde5380c, 23);
    MD5_AVX2_STEP(MD5_AVX2_H, a, b, c, d, k[ 1], 0xa4beea44,  4);
    MD5_AVX2_STEP(MD5_AVX2_H, d, a, b, c, k[ 4], 0x4bdecfa9, 11);
    MD5_AVX2_STEP(MD5_AVX2_H, c, d, a, b, k[ 7], 0xf6bb4b60, 16);
    MD5_AVX2_STEP(MD5_AVX2_H, b, c, d, a, k[10], 0xbebfbc70, 23);
    MD5_AVX2_STEP(MD5_AVX2_H, a, b, c, d, k[13], 0x289b7ec6,  4);
    MD5_AVX2_STEP(MD5_AVX2_H, d, a, b, c, k[ 0], 0xeaa127fa, 11);
    MD5_AVX2_STEP(MD5_AVX2_H, c, d, a, b, k[ 3], 0xd4ef3085, 16);
    MD5_AVX2_STEP(MD5_AVX2_H, b, c, d, a, k[ 6], 0x04881d05, 23);
    MD5_AVX2_STEP(MD5_AVX2_H, a, b, c, d, k[ 9], 0xd9d4d039,  4);
    MD5_AVX2_STEP(MD5_AVX2_H, d, a, b, c, k[12], 0xe6db99e5, 11);
    MD5_AVX2_STEP(MD5_AVX2_H, c, d, a, b, k[15], 0x1fa27cf8, 16);
    MD5_AVX2_STEP(MD5_AVX2_H, b, c, d, a, k[ 2], 0xc4ac5665, 23);
    /* Round 4 — steps 48..60 (a is finalized after step 60) */
    MD5_AVX2_STEP(MD5_AVX2_I, a, b, c, d, k[ 0], 0xf4292244,  6);
    MD5_AVX2_STEP(MD5_AVX2_I, d, a, b, c, k[ 7], 0x432aff97, 10);
    MD5_AVX2_STEP(MD5_AVX2_I, c, d, a, b, k[14], 0xab9423a7, 15);
    MD5_AVX2_STEP(MD5_AVX2_I, b, c, d, a, k[ 5], 0xfc93a039, 21);
    MD5_AVX2_STEP(MD5_AVX2_I, a, b, c, d, k[12], 0x655b59c3,  6);
    MD5_AVX2_STEP(MD5_AVX2_I, d, a, b, c, k[ 3], 0x8f0ccc92, 10);
    MD5_AVX2_STEP(MD5_AVX2_I, c, d, a, b, k[10], 0xffeff47d, 15);
    MD5_AVX2_STEP(MD5_AVX2_I, b, c, d, a, k[ 1], 0x85845dd1, 21);
    MD5_AVX2_STEP(MD5_AVX2_I, a, b, c, d, k[ 8], 0x6fa87e4f,  6);
    MD5_AVX2_STEP(MD5_AVX2_I, d, a, b, c, k[15], 0xfe2ce6e0, 10);
    MD5_AVX2_STEP(MD5_AVX2_I, c, d, a, b, k[ 6], 0xa3014314, 15);
    MD5_AVX2_STEP(MD5_AVX2_I, b, c, d, a, k[13], 0x4e0811a1, 21);
    MD5_AVX2_STEP(MD5_AVX2_I, a, b, c, d, k[ 4], 0xf7537e82,  6);  /* step 60: a final */

    /* --- Early exit: check hash[0] before computing last 3 rounds --- */
    __m256i ha = _mm256_add_epi32(a, _mm256_set1_epi32(0x67452301));
    int any_match = 0;
    for (uint32_t ih = 0; ih < cfg->hashes_num; ih++) {
        __m256i cmp = _mm256_cmpeq_epi32(ha,
                          _mm256_set1_epi32((int32_t)cfg->hashes[4 * ih]));
        any_match |= _mm256_movemask_epi8(cmp);
    }
    if (!any_match) return;  /* fast path: skip last 3 rounds */

    /* Rare path: finish rounds 61-63 */
    MD5_AVX2_STEP(MD5_AVX2_I, d, a, b, c, k[11], 0xbd3af235, 10);
    MD5_AVX2_STEP(MD5_AVX2_I, c, d, a, b, k[ 2], 0x2ad7d2bb, 15);
    MD5_AVX2_STEP(MD5_AVX2_I, b, c, d, a, k[ 9], 0xeb86d391, 21);

    __m256i hb = _mm256_add_epi32(b, _mm256_set1_epi32((int32_t)0xefcdab89));
    __m256i hc = _mm256_add_epi32(c, _mm256_set1_epi32((int32_t)0x98badcfe));
    __m256i hd = _mm256_add_epi32(d, _mm256_set1_epi32(0x10325476));

    uint32_t a_vals[8], b_vals[8], c_vals[8], d_vals[8];
    _mm256_storeu_si256((__m256i *)a_vals, ha);
    _mm256_storeu_si256((__m256i *)b_vals, hb);
    _mm256_storeu_si256((__m256i *)c_vals, hc);
    _mm256_storeu_si256((__m256i *)d_vals, hd);

    for (int lane = 0; lane < count; lane++) {
        uint32_t hash[4] = {a_vals[lane], b_vals[lane], c_vals[lane], d_vals[lane]};
        /* Reconstruct this lane's key from SoA layout (rare path) */
        uint32_t lane_key[16];
        for (int w = 0; w < 16; w++) lane_key[w] = keys[w][lane];
        check_hashes(cfg, hash, lane_key, wcs_arr[lane]);
    }
}
#endif

/*
 * OR-based string construction into SoA key layout.
 * Writes precomputed word images directly into keys[word_pos][lane],
 * where the first index is the uint32 position in the MD5 block (0-15)
 * and the second index is the SIMD lane.
 * keys must be zeroed before calling (once per batch, not per lane).
 */
static int construct_string_or_soa_16(permut_task *task,
                                       uint32_t keys[][16],
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
    /* MD5 padding: 0x80 byte after message, length in bits at offset 56 */
    ((char *)&keys[wcs >> 2][lane])[(wcs & 3)] = (char)0x80;
    keys[14][lane] = (uint32_t)(wcs << 3);
    return wcs;
}

/*
 * OR-based string construction into SoA key layout (8-lane variant for AVX2).
 */
static int construct_string_or_soa_8(permut_task *task,
                                      uint32_t keys[][8],
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

/*
 * Combined AVX-512 MD5 (16-lane) + early-exit hash check (GPU-8).
 * Computes only 61 of 64 MD5 rounds. hash[0] (a) is finalized after step 60.
 * SIMD-checks a against all targets using _mm512_cmpeq_epi32_mask;
 * skips last 3 rounds (~5% MD5 savings) for the ~100% case where no
 * hash[0] matches.
 * Keys are in SoA layout: keys[word_pos][lane] — enables aligned loads.
 */
#if defined(__x86_64__) || defined(_M_AMD64)
static void md5_check_avx512(cruncher_config *cfg,
                              uint32_t keys[16][16], int wcs_arr[16], int count) {
    __m512i k[16];
    for (int w = 0; w < 16; w++) {
        k[w] = _mm512_load_si512((__m512i *)keys[w]);
    }

    __m512i a = _mm512_set1_epi32(0x67452301);
    __m512i b = _mm512_set1_epi32((int32_t)0xefcdab89);
    __m512i c = _mm512_set1_epi32((int32_t)0x98badcfe);
    __m512i d = _mm512_set1_epi32(0x10325476);

    /* Round 1 */
    MD5_512_STEP(MD5_512_F, a, b, c, d, k[ 0], 0xd76aa478,  7);
    MD5_512_STEP(MD5_512_F, d, a, b, c, k[ 1], 0xe8c7b756, 12);
    MD5_512_STEP(MD5_512_F, c, d, a, b, k[ 2], 0x242070db, 17);
    MD5_512_STEP(MD5_512_F, b, c, d, a, k[ 3], 0xc1bdceee, 22);
    MD5_512_STEP(MD5_512_F, a, b, c, d, k[ 4], 0xf57c0faf,  7);
    MD5_512_STEP(MD5_512_F, d, a, b, c, k[ 5], 0x4787c62a, 12);
    MD5_512_STEP(MD5_512_F, c, d, a, b, k[ 6], 0xa8304613, 17);
    MD5_512_STEP(MD5_512_F, b, c, d, a, k[ 7], 0xfd469501, 22);
    MD5_512_STEP(MD5_512_F, a, b, c, d, k[ 8], 0x698098d8,  7);
    MD5_512_STEP(MD5_512_F, d, a, b, c, k[ 9], 0x8b44f7af, 12);
    MD5_512_STEP(MD5_512_F, c, d, a, b, k[10], 0xffff5bb1, 17);
    MD5_512_STEP(MD5_512_F, b, c, d, a, k[11], 0x895cd7be, 22);
    MD5_512_STEP(MD5_512_F, a, b, c, d, k[12], 0x6b901122,  7);
    MD5_512_STEP(MD5_512_F, d, a, b, c, k[13], 0xfd987193, 12);
    MD5_512_STEP(MD5_512_F, c, d, a, b, k[14], 0xa679438e, 17);
    MD5_512_STEP(MD5_512_F, b, c, d, a, k[15], 0x49b40821, 22);
    /* Round 2 */
    MD5_512_STEP(MD5_512_G, a, b, c, d, k[ 1], 0xf61e2562,  5);
    MD5_512_STEP(MD5_512_G, d, a, b, c, k[ 6], 0xc040b340,  9);
    MD5_512_STEP(MD5_512_G, c, d, a, b, k[11], 0x265e5a51, 14);
    MD5_512_STEP(MD5_512_G, b, c, d, a, k[ 0], 0xe9b6c7aa, 20);
    MD5_512_STEP(MD5_512_G, a, b, c, d, k[ 5], 0xd62f105d,  5);
    MD5_512_STEP(MD5_512_G, d, a, b, c, k[10], 0x02441453,  9);
    MD5_512_STEP(MD5_512_G, c, d, a, b, k[15], 0xd8a1e681, 14);
    MD5_512_STEP(MD5_512_G, b, c, d, a, k[ 4], 0xe7d3fbc8, 20);
    MD5_512_STEP(MD5_512_G, a, b, c, d, k[ 9], 0x21e1cde6,  5);
    MD5_512_STEP(MD5_512_G, d, a, b, c, k[14], 0xc33707d6,  9);
    MD5_512_STEP(MD5_512_G, c, d, a, b, k[ 3], 0xf4d50d87, 14);
    MD5_512_STEP(MD5_512_G, b, c, d, a, k[ 8], 0x455a14ed, 20);
    MD5_512_STEP(MD5_512_G, a, b, c, d, k[13], 0xa9e3e905,  5);
    MD5_512_STEP(MD5_512_G, d, a, b, c, k[ 2], 0xfcefa3f8,  9);
    MD5_512_STEP(MD5_512_G, c, d, a, b, k[ 7], 0x676f02d9, 14);
    MD5_512_STEP(MD5_512_G, b, c, d, a, k[12], 0x8d2a4c8a, 20);
    /* Round 3 */
    MD5_512_STEP(MD5_512_H, a, b, c, d, k[ 5], 0xfffa3942,  4);
    MD5_512_STEP(MD5_512_H, d, a, b, c, k[ 8], 0x8771f681, 11);
    MD5_512_STEP(MD5_512_H, c, d, a, b, k[11], 0x6d9d6122, 16);
    MD5_512_STEP(MD5_512_H, b, c, d, a, k[14], 0xfde5380c, 23);
    MD5_512_STEP(MD5_512_H, a, b, c, d, k[ 1], 0xa4beea44,  4);
    MD5_512_STEP(MD5_512_H, d, a, b, c, k[ 4], 0x4bdecfa9, 11);
    MD5_512_STEP(MD5_512_H, c, d, a, b, k[ 7], 0xf6bb4b60, 16);
    MD5_512_STEP(MD5_512_H, b, c, d, a, k[10], 0xbebfbc70, 23);
    MD5_512_STEP(MD5_512_H, a, b, c, d, k[13], 0x289b7ec6,  4);
    MD5_512_STEP(MD5_512_H, d, a, b, c, k[ 0], 0xeaa127fa, 11);
    MD5_512_STEP(MD5_512_H, c, d, a, b, k[ 3], 0xd4ef3085, 16);
    MD5_512_STEP(MD5_512_H, b, c, d, a, k[ 6], 0x04881d05, 23);
    MD5_512_STEP(MD5_512_H, a, b, c, d, k[ 9], 0xd9d4d039,  4);
    MD5_512_STEP(MD5_512_H, d, a, b, c, k[12], 0xe6db99e5, 11);
    MD5_512_STEP(MD5_512_H, c, d, a, b, k[15], 0x1fa27cf8, 16);
    MD5_512_STEP(MD5_512_H, b, c, d, a, k[ 2], 0xc4ac5665, 23);
    /* Round 4 — steps 48..60 (a is finalized after step 60) */
    MD5_512_STEP(MD5_512_I, a, b, c, d, k[ 0], 0xf4292244,  6);
    MD5_512_STEP(MD5_512_I, d, a, b, c, k[ 7], 0x432aff97, 10);
    MD5_512_STEP(MD5_512_I, c, d, a, b, k[14], 0xab9423a7, 15);
    MD5_512_STEP(MD5_512_I, b, c, d, a, k[ 5], 0xfc93a039, 21);
    MD5_512_STEP(MD5_512_I, a, b, c, d, k[12], 0x655b59c3,  6);
    MD5_512_STEP(MD5_512_I, d, a, b, c, k[ 3], 0x8f0ccc92, 10);
    MD5_512_STEP(MD5_512_I, c, d, a, b, k[10], 0xffeff47d, 15);
    MD5_512_STEP(MD5_512_I, b, c, d, a, k[ 1], 0x85845dd1, 21);
    MD5_512_STEP(MD5_512_I, a, b, c, d, k[ 8], 0x6fa87e4f,  6);
    MD5_512_STEP(MD5_512_I, d, a, b, c, k[15], 0xfe2ce6e0, 10);
    MD5_512_STEP(MD5_512_I, c, d, a, b, k[ 6], 0xa3014314, 15);
    MD5_512_STEP(MD5_512_I, b, c, d, a, k[13], 0x4e0811a1, 21);
    MD5_512_STEP(MD5_512_I, a, b, c, d, k[ 4], 0xf7537e82,  6);  /* step 60: a final */

    /* --- Early exit: check hash[0] before computing last 3 rounds --- */
    __m512i ha = _mm512_add_epi32(a, _mm512_set1_epi32(0x67452301));
    __mmask16 any_match = 0;
    for (uint32_t ih = 0; ih < cfg->hashes_num; ih++) {
        any_match |= _mm512_cmpeq_epi32_mask(ha,
                         _mm512_set1_epi32((int32_t)cfg->hashes[4 * ih]));
    }
    if (!any_match) return;  /* fast path: skip last 3 rounds */

    /* Rare path: finish rounds 61-63 */
    MD5_512_STEP(MD5_512_I, d, a, b, c, k[11], 0xbd3af235, 10);
    MD5_512_STEP(MD5_512_I, c, d, a, b, k[ 2], 0x2ad7d2bb, 15);
    MD5_512_STEP(MD5_512_I, b, c, d, a, k[ 9], 0xeb86d391, 21);

    __m512i hb = _mm512_add_epi32(b, _mm512_set1_epi32((int32_t)0xefcdab89));
    __m512i hc = _mm512_add_epi32(c, _mm512_set1_epi32((int32_t)0x98badcfe));
    __m512i hd = _mm512_add_epi32(d, _mm512_set1_epi32(0x10325476));

    /* Extract and do full 4-word comparison for matching lanes */
    uint32_t a_vals[16], b_vals[16], c_vals[16], d_vals[16];
    _mm512_storeu_si512(a_vals, ha);
    _mm512_storeu_si512(b_vals, hb);
    _mm512_storeu_si512(c_vals, hc);
    _mm512_storeu_si512(d_vals, hd);

    __mmask16 remaining = any_match & (((uint32_t)1 << count) - 1);
    while (remaining) {
        int lane = __builtin_ctz(remaining);
        remaining &= remaining - 1;
        uint32_t hash[4] = {a_vals[lane], b_vals[lane], c_vals[lane], d_vals[lane]};
        /* Reconstruct this lane's key from SoA layout (rare path) */
        uint32_t lane_key[16];
        for (int w = 0; w < 16; w++) lane_key[w] = keys[w][lane];
        check_hashes(cfg, hash, lane_key, wcs_arr[lane]);
    }
}
#endif

static void process_task(avx_cruncher_ctx *actx, permut_task *task) {
    if (task->i >= task->n) return;
    cruncher_config *cfg = actx->cfg;

#if defined(__x86_64__) || defined(_M_AMD64)
    if (actx->mode == SIMD_AVX512) {
        /* --- AVX-512 path (16-lane, SoA key layout) --- */
        uint32_t wimg[MAX_STR_LENGTH][11];
        uint8_t wlen_sp[MAX_STR_LENGTH];
        int num_offsets;
        precompute_word_images(task, wimg, wlen_sp, &num_offsets);

        uint32_t keys[16][16] __attribute__((aligned(64)));  /* keys[word][lane] */
        memset(keys, 0, sizeof(keys));
        int wcs_arr[16];
        int batch = 0;

        do {
            wcs_arr[batch] = construct_string_or_soa_16(task, keys, batch,
                                                         wimg, wlen_sp, num_offsets);
            batch++;

            if (batch == 16) {
                md5_check_avx512(cfg, keys, wcs_arr, 16);
                batch = 0;
                memset(keys, 0, sizeof(keys));
            }
        } while (heap_next(task));

        if (batch > 0) {
            /* Duplicate last lane into remaining slots */
            for (int w = 0; w < 16; w++)
                for (int i = batch; i < 16; i++)
                    keys[w][i] = keys[w][batch - 1];
            md5_check_avx512(cfg, keys, wcs_arr, batch);
        }
        return;
    }
    /* --- AVX2 path (8-lane, SoA key layout) --- */
    if (actx->mode == SIMD_AVX2) {
        uint32_t wimg[MAX_STR_LENGTH][11];
        uint8_t wlen_sp[MAX_STR_LENGTH];
        int num_offsets;
        precompute_word_images(task, wimg, wlen_sp, &num_offsets);

        uint32_t keys[16][8] __attribute__((aligned(32)));  /* keys[word][lane] */
        memset(keys, 0, sizeof(keys));
        int wcs_arr[8];
        int batch = 0;

        do {
            wcs_arr[batch] = construct_string_or_soa_8(task, keys, batch,
                                                        wimg, wlen_sp, num_offsets);
            batch++;

            if (batch == 8) {
                md5_check_avx2(cfg, keys, wcs_arr, 8);
                batch = 0;
                memset(keys, 0, sizeof(keys));
            }
        } while (heap_next(task));

        if (batch > 0) {
            /* Duplicate last lane into remaining slots */
            for (int w = 0; w < 16; w++)
                for (int i = batch; i < 8; i++)
                    keys[w][i] = keys[w][batch - 1];
            md5_check_avx2(cfg, keys, wcs_arr, batch);
        }
        return;
    }
#endif
    /* --- Scalar fallback --- */
    do {
        uint32_t key[16];
        int wcs = construct_string(task, key);
        uint32_t hash[4];
        md5_scalar(key, hash);
        check_hashes(cfg, hash, key, wcs);
    } while (heap_next(task));
}

/* ---------- Vtable functions ---------- */

static uint32_t avx512_probe(void) {
#if defined(__x86_64__) || defined(_M_AMD64)
    if (!cpu_has_avx512f()) return 0;
    uint32_t cores = num_cpu_cores();
    printf("  avx512: %d cores, avx-512 available\n", cores);
    return cores;
#else
    return 0;
#endif
}

static uint32_t avx2_probe(void) {
#if defined(__x86_64__) || defined(_M_AMD64)
    if (!__builtin_cpu_supports("avx2")) return 0;
    uint32_t cores = num_cpu_cores();
    printf("  avx2: %d cores, avx2 available\n", cores);
    return cores;
#else
    return 0;
#endif
}

static uint32_t scalar_probe(void) {
    uint32_t cores = num_cpu_cores();
    printf("  cpu: %d cores available, suggesting %d thread(s)\n", cores, cores);
    return cores;
}

static int avx_create_with_mode(void *ctx, cruncher_config *cfg, uint32_t instance_id, simd_mode mode) {
    avx_cruncher_ctx *actx = ctx;
    actx->cfg = cfg;
    actx->mode = mode;
    actx->is_running = false;
    actx->consumed_bufs = 0;
    actx->consumed_anas = 0;
    actx->task_time_start = 0;
    actx->task_time_end = 0;
    return 0;
}

static int avx512_create(void *ctx, cruncher_config *cfg, uint32_t instance_id) {
    return avx_create_with_mode(ctx, cfg, instance_id, SIMD_AVX512);
}

static int avx2_create(void *ctx, cruncher_config *cfg, uint32_t instance_id) {
    return avx_create_with_mode(ctx, cfg, instance_id, SIMD_AVX2);
}

static int scalar_create(void *ctx, cruncher_config *cfg, uint32_t instance_id) {
    return avx_create_with_mode(ctx, cfg, instance_id, SIMD_SCALAR);
}

static void *avx_run(void *ctx) {
    avx_cruncher_ctx *actx = ctx;
    actx->is_running = true;
    actx->task_time_start = current_micros();

    tasks_buffer *buf;
    while (1) {
        tasks_buffers_get_buffer(actx->cfg->tasks_buffs, &buf);
        if (buf == NULL) break;

        for (uint32_t i = 0; i < buf->num_tasks; i++) {
            process_task(actx, &buf->permut_tasks[i]);
        }

        actx->consumed_anas += buf->num_anas;
        actx->consumed_bufs++;
        tasks_buffers_recycle(actx->cfg->tasks_buffs, buf);
    }

    actx->task_time_end = current_micros();
    actx->is_running = false;
    return NULL;
}

static void avx_get_stats(void *ctx, float *busy_pct, float *anas_per_sec) {
    avx_cruncher_ctx *actx = ctx;
    uint64_t now = current_micros();
    uint64_t start = actx->task_time_start;
    uint64_t end = actx->task_time_end;

    if (!start) {
        *busy_pct = 0;
        *anas_per_sec = 0;
        return;
    }

    uint64_t elapsed = (end ? end : now) - start;
    if (elapsed == 0) elapsed = 1;

    /* CPU thread is always 100% busy while running (no idle gaps between
     * kernel dispatches like the GPU backend), so this binary metric is
     * accurate — unlike the GPU's windowed kernel-time measurement. */
    *busy_pct = actx->is_running ? 100.0f : 0.0f;
    *anas_per_sec = (float)actx->consumed_anas / ((float)elapsed / 1000000.0f);
}

static uint64_t avx_get_total_anas(void *ctx) {
    return ((avx_cruncher_ctx *)ctx)->consumed_anas;
}

static bool avx_is_running(void *ctx) {
    return ((avx_cruncher_ctx *)ctx)->is_running;
}

static int avx_destroy(void *ctx) {
    return 0;
}

cruncher_ops avx512_cruncher_ops = {
    .name = "avx512",
    .probe = avx512_probe,
    .create = avx512_create,
    .run = avx_run,
    .get_stats = avx_get_stats,
    .get_total_anas = avx_get_total_anas,
    .is_running = avx_is_running,
    .destroy = avx_destroy,
    .ctx_size = sizeof(avx_cruncher_ctx),
};

cruncher_ops avx2_cruncher_ops = {
    .name = "avx2",
    .probe = avx2_probe,
    .create = avx2_create,
    .run = avx_run,
    .get_stats = avx_get_stats,
    .get_total_anas = avx_get_total_anas,
    .is_running = avx_is_running,
    .destroy = avx_destroy,
    .ctx_size = sizeof(avx_cruncher_ctx),
};

cruncher_ops scalar_cruncher_ops = {
    .name = "cpu",
    .probe = scalar_probe,
    .create = scalar_create,
    .run = avx_run,
    .get_stats = avx_get_stats,
    .get_total_anas = avx_get_total_anas,
    .is_running = avx_is_running,
    .destroy = avx_destroy,
    .ctx_size = sizeof(avx_cruncher_ctx),
};
