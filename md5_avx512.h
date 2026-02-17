#ifndef ANABRUTE_MD5_AVX512_H
#define ANABRUTE_MD5_AVX512_H

/*
 * AVX-512 vectorized MD5: computes 16 independent MD5 hashes in parallel.
 * Only available on x86_64 with AVX-512F support.
 *
 * Uses _mm512_ternarylogic_epi32 for single-instruction boolean functions,
 * _mm512_rol_epi32 for single-instruction rotates, and
 * _mm512_i32gather_epi32 for hardware scatter/gather.
 *
 * Each input is a 16-element uint32_t array (64 bytes, already padded).
 * Each output is a 4-element uint32_t array (16 bytes, raw MD5 digest).
 */

#if defined(__x86_64__) && defined(__AVX512F__)

#include <immintrin.h>
#include <stdint.h>

/* Vectorized MD5 round functions using ternarylogic (1 instruction each)
 *
 * F(x,y,z) = (x & y) | (~x & z)  = ternarylogic(z, y, x, 0xCA)
 * G(x,y,z) = (z & x) | (~z & y)  = ternarylogic(y, x, z, 0xCA)
 * H(x,y,z) = x ^ y ^ z           = ternarylogic(x, y, z, 0x96)
 * I(x,y,z) = y ^ (x | ~z)        = ternarylogic(x, y, z, 0x39)
 *
 * Truth table derivation for I:
 *   x y z | ~z | x|~z | y^(x|~z)
 *   0 0 0 |  1 |   1  |    1     -> bit 0 = 1
 *   0 0 1 |  0 |   0  |    0     -> bit 1 = 0
 *   0 1 0 |  1 |   1  |    0     -> bit 2 = 0
 *   0 1 1 |  0 |   0  |    1     -> bit 3 = 1
 *   1 0 0 |  1 |   1  |    1     -> bit 4 = 1
 *   1 0 1 |  0 |   1  |    1     -> bit 5 = 1
 *   1 1 0 |  1 |   1  |    0     -> bit 6 = 0
 *   1 1 1 |  0 |   1  |    0     -> bit 7 = 0
 *   Result: 0b00110001 = 0x39   (reading bits 7..0)
 *   Wait — let me recheck with ternarylogic arg order (a=x, b=y, c=z):
 *   bit index = 4*a + 2*b + c
 *   idx 0 (a=0,b=0,c=0): y^(x|~z) = 0^(0|1) = 1
 *   idx 1 (a=0,b=0,c=1): 0^(0|0) = 0
 *   idx 2 (a=0,b=1,c=0): 1^(0|1) = 0
 *   idx 3 (a=0,b=1,c=1): 1^(0|0) = 1
 *   idx 4 (a=1,b=0,c=0): 0^(1|1) = 1
 *   idx 5 (a=1,b=0,c=1): 0^(1|0) = 1
 *   idx 6 (a=1,b=1,c=0): 1^(1|1) = 0
 *   idx 7 (a=1,b=1,c=1): 1^(1|0) = 0
 *   = 0b00110001 = reversed to 0b10001100... no wait:
 *   bits[7:0] = {bit7, bit6, ..., bit0} = {0,0,1,1,1,0,0,1} = 0x39
 */
#define MD5_512_F(x, y, z) _mm512_ternarylogic_epi32((z), (y), (x), 0xCA)
#define MD5_512_G(x, y, z) _mm512_ternarylogic_epi32((y), (x), (z), 0xCA)
#define MD5_512_H(x, y, z) _mm512_ternarylogic_epi32((x), (y), (z), 0x96)
#define MD5_512_I(x, y, z) _mm512_ternarylogic_epi32((x), (y), (z), 0x39)

/* Rotate left using native AVX-512 instruction */
#define MD5_512_ROTL(val, s) _mm512_rol_epi32((val), (s))

/* MD5 step macro for AVX-512 */
#define MD5_512_STEP(f, a, b, c, d, x, t, s) do { \
    (a) = _mm512_add_epi32((a), f((b), (c), (d))); \
    (a) = _mm512_add_epi32((a), (x)); \
    (a) = _mm512_add_epi32((a), _mm512_set1_epi32((int32_t)(t))); \
    (a) = MD5_512_ROTL((a), (s)); \
    (a) = _mm512_add_epi32((a), (b)); \
} while (0)

/*
 * Compute 16 MD5 hashes in parallel using AVX-512.
 *
 * keys:   array of 16 pointers to uint32_t[16] input blocks (already MD5-padded)
 * hashes: array of 16 pointers to uint32_t[4] output digests
 */
static inline void md5_avx512_x16(const uint32_t *keys[16], uint32_t *hashes[16]) {
    /* Gather input words using hardware gather.
     * For each MD5 word w, we need keys[lane][w] across all 16 lanes.
     * With contiguous keys[16][16] layout, keys[lane][w] is at base + lane*16 + w.
     * Use _mm512_i32gather_epi32 with indices = {0*16+w, 1*16+w, ..., 15*16+w}.
     */
    __m512i k[16];
    const __m512i lane_offsets = _mm512_setr_epi32(
        0*16, 1*16, 2*16, 3*16, 4*16, 5*16, 6*16, 7*16,
        8*16, 9*16, 10*16, 11*16, 12*16, 13*16, 14*16, 15*16
    );

    /* If keys are contiguous (keys[i] = keys[0] + i*16), use hardware gather */
    const uint32_t *base = keys[0];
    for (int w = 0; w < 16; w++) {
        __m512i indices = _mm512_add_epi32(lane_offsets, _mm512_set1_epi32(w));
        k[w] = _mm512_i32gather_epi32(indices, (const int *)base, 4);
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

    /* Round 4 */
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
    MD5_512_STEP(MD5_512_I, a, b, c, d, k[ 4], 0xf7537e82,  6);
    MD5_512_STEP(MD5_512_I, d, a, b, c, k[11], 0xbd3af235, 10);
    MD5_512_STEP(MD5_512_I, c, d, a, b, k[ 2], 0x2ad7d2bb, 15);
    MD5_512_STEP(MD5_512_I, b, c, d, a, k[ 9], 0xeb86d391, 21);

    /* Add initial values */
    a = _mm512_add_epi32(a, _mm512_set1_epi32(0x67452301));
    b = _mm512_add_epi32(b, _mm512_set1_epi32((int32_t)0xefcdab89));
    c = _mm512_add_epi32(c, _mm512_set1_epi32((int32_t)0x98badcfe));
    d = _mm512_add_epi32(d, _mm512_set1_epi32(0x10325476));

    /* Scatter results: extract each lane to its output buffer */
    uint32_t a_vals[16], b_vals[16], c_vals[16], d_vals[16];
    _mm512_storeu_si512(a_vals, a);
    _mm512_storeu_si512(b_vals, b);
    _mm512_storeu_si512(c_vals, c);
    _mm512_storeu_si512(d_vals, d);

    for (int i = 0; i < 16; i++) {
        hashes[i][0] = a_vals[i];
        hashes[i][1] = b_vals[i];
        hashes[i][2] = c_vals[i];
        hashes[i][3] = d_vals[i];
    }
}

/*
 * Compute 16 MD5 hashes, returning raw __m512i vectors (no scatter).
 * Takes a contiguous uint32_t[16][16] array directly.
 * Caller can do SIMD comparison on the vectors before extracting scalars.
 */
static inline void md5_avx512_x16_vec(
    const uint32_t *base,  /* pointer to contiguous keys[16][16] */
    __m512i *out_a, __m512i *out_b, __m512i *out_c, __m512i *out_d
) {
    __m512i k[16];
    const __m512i lane_offsets = _mm512_setr_epi32(
        0*16, 1*16, 2*16, 3*16, 4*16, 5*16, 6*16, 7*16,
        8*16, 9*16, 10*16, 11*16, 12*16, 13*16, 14*16, 15*16
    );
    for (int w = 0; w < 16; w++) {
        __m512i indices = _mm512_add_epi32(lane_offsets, _mm512_set1_epi32(w));
        k[w] = _mm512_i32gather_epi32(indices, (const int *)base, 4);
    }

    __m512i a = _mm512_set1_epi32(0x67452301);
    __m512i b = _mm512_set1_epi32((int32_t)0xefcdab89);
    __m512i c = _mm512_set1_epi32((int32_t)0x98badcfe);
    __m512i d = _mm512_set1_epi32(0x10325476);

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
    MD5_512_STEP(MD5_512_I, a, b, c, d, k[ 4], 0xf7537e82,  6);
    MD5_512_STEP(MD5_512_I, d, a, b, c, k[11], 0xbd3af235, 10);
    MD5_512_STEP(MD5_512_I, c, d, a, b, k[ 2], 0x2ad7d2bb, 15);
    MD5_512_STEP(MD5_512_I, b, c, d, a, k[ 9], 0xeb86d391, 21);

    *out_a = _mm512_add_epi32(a, _mm512_set1_epi32(0x67452301));
    *out_b = _mm512_add_epi32(b, _mm512_set1_epi32((int32_t)0xefcdab89));
    *out_c = _mm512_add_epi32(c, _mm512_set1_epi32((int32_t)0x98badcfe));
    *out_d = _mm512_add_epi32(d, _mm512_set1_epi32(0x10325476));
}

#endif /* __x86_64__ && __AVX512F__ */

#endif /* ANABRUTE_MD5_AVX512_H */
