#ifndef ANABRUTE_MD5_AVX2_H
#define ANABRUTE_MD5_AVX2_H

/*
 * AVX2 vectorized MD5: computes 8 independent MD5 hashes in parallel.
 * Only available on x86_64 with AVX2 support.
 *
 * NOT YET WIRED INTO avx_cruncher.c — currently the CPU backend uses
 * md5_scalar() for all platforms. This file is scaffolding for a future
 * optimization that batches 8 permutations and computes their MD5 hashes
 * simultaneously using AVX2 intrinsics.
 *
 * Each input is a 16-element uint32_t array (64 bytes, already padded).
 * Each output is a 4-element uint32_t array (16 bytes, raw MD5 digest).
 */

#ifdef __x86_64__

#include <immintrin.h>
#include <stdint.h>

/* Vectorized MD5 round functions operating on 8 lanes */
#define MD5_AVX2_F(x, y, z) _mm256_or_si256(_mm256_and_si256((x), (y)), _mm256_andnot_si256((x), (z)))
#define MD5_AVX2_G(x, y, z) _mm256_or_si256(_mm256_and_si256((z), (x)), _mm256_andnot_si256((z), (y)))
#define MD5_AVX2_H(x, y, z) _mm256_xor_si256(_mm256_xor_si256((x), (y)), (z))
#define MD5_AVX2_I(x, y, z) _mm256_xor_si256((y), _mm256_or_si256((x), _mm256_xor_si256(_mm256_set1_epi32(-1), (z))))

/* Rotate left by compile-time constant */
#define MD5_AVX2_ROTL(val, s) \
    _mm256_or_si256(_mm256_slli_epi32((val), (s)), _mm256_srli_epi32((val), 32 - (s)))

/* MD5 step macro for AVX2 */
#define MD5_AVX2_STEP(f, a, b, c, d, x, t, s) do { \
    (a) = _mm256_add_epi32((a), f((b), (c), (d))); \
    (a) = _mm256_add_epi32((a), (x)); \
    (a) = _mm256_add_epi32((a), _mm256_set1_epi32((int32_t)(t))); \
    (a) = MD5_AVX2_ROTL((a), (s)); \
    (a) = _mm256_add_epi32((a), (b)); \
} while (0)

/*
 * Compute 8 MD5 hashes in parallel using AVX2.
 *
 * keys:   array of 8 pointers to uint32_t[16] input blocks (already MD5-padded)
 * hashes: array of 8 pointers to uint32_t[4] output digests
 */
static inline void md5_avx2_x8(const uint32_t *keys[8], uint32_t *hashes[8]) {
    /* Gather input words: key[lane][word] -> transposed as k[word] with 8 lanes */
    __m256i k[16];
    for (int w = 0; w < 16; w++) {
        k[w] = _mm256_set_epi32(
            keys[7][w], keys[6][w], keys[5][w], keys[4][w],
            keys[3][w], keys[2][w], keys[1][w], keys[0][w]
        );
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

    /* Round 4 */
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
    MD5_AVX2_STEP(MD5_AVX2_I, a, b, c, d, k[ 4], 0xf7537e82,  6);
    MD5_AVX2_STEP(MD5_AVX2_I, d, a, b, c, k[11], 0xbd3af235, 10);
    MD5_AVX2_STEP(MD5_AVX2_I, c, d, a, b, k[ 2], 0x2ad7d2bb, 15);
    MD5_AVX2_STEP(MD5_AVX2_I, b, c, d, a, k[ 9], 0xeb86d391, 21);

    /* Add initial values */
    a = _mm256_add_epi32(a, _mm256_set1_epi32(0x67452301));
    b = _mm256_add_epi32(b, _mm256_set1_epi32((int32_t)0xefcdab89));
    c = _mm256_add_epi32(c, _mm256_set1_epi32((int32_t)0x98badcfe));
    d = _mm256_add_epi32(d, _mm256_set1_epi32(0x10325476));

    /* Extract results to each lane's output buffer */
    /* a/b/c/d each have 8 lanes; we need hash[lane] = {a[lane], b[lane], c[lane], d[lane]} */
    uint32_t a_vals[8], b_vals[8], c_vals[8], d_vals[8];
    _mm256_storeu_si256((__m256i *)a_vals, a);
    _mm256_storeu_si256((__m256i *)b_vals, b);
    _mm256_storeu_si256((__m256i *)c_vals, c);
    _mm256_storeu_si256((__m256i *)d_vals, d);

    for (int i = 0; i < 8; i++) {
        hashes[i][0] = a_vals[i];
        hashes[i][1] = b_vals[i];
        hashes[i][2] = c_vals[i];
        hashes[i][3] = d_vals[i];
    }
}

#endif /* __x86_64__ */

#endif /* ANABRUTE_MD5_AVX2_H */
