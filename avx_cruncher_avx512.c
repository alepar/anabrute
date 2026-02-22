/*
 * AVX-512 MD5 hashing — compiled separately with -mavx512f so that
 * the rest of avx_cruncher.c stays free of AVX-512 instructions.
 * This file is only linked; the AVX-512 code path is only called
 * at runtime after cpu_has_avx512f() confirms hardware support.
 */
#include "avx_cruncher.h"
#include "md5_avx512.h"
#include "task_buffers.h"

#include <string.h>
#include <stdint.h>

/*
 * Combined AVX-512 MD5 (16-lane) + early-exit hash check.
 * Computes only 61 of 64 MD5 rounds. hash[0] (a) is finalized after step 60.
 * SIMD-checks a against all targets using _mm512_cmpeq_epi32_mask;
 * skips last 3 rounds (~5% MD5 savings) for the ~100% case where no
 * hash[0] matches.
 * Keys are in SoA layout: keys[word_pos][lane] — enables aligned loads.
 */
void avx512_md5_check(cruncher_config *cfg,
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
        avx_check_hashes(cfg, hash, lane_key, wcs_arr[lane]);
    }
}
