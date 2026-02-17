#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include "os.h"
#include "md5_avx2.h"
#include "md5_avx512.h"

/* Copy PUTCHAR_SCALAR and md5_scalar from avx_cruncher.c for isolated testing */
#define PUTCHAR_SCALAR(buf, index, val) \
    (buf)[(index) >> 2] = ((buf)[(index) >> 2] & ~(0xffU << (((index) & 3) << 3))) + ((uint32_t)(val) << (((index) & 3) << 3))

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
    uint32_t a = 0x67452301, b = 0xefcdab89, c = 0x98badcfe, d = 0x10325476;
    STEP(F, a, b, c, d, GET(0), 0xd76aa478, 7) STEP(F, d, a, b, c, GET(1), 0xe8c7b756, 12)
    STEP(F, c, d, a, b, GET(2), 0x242070db, 17) STEP(F, b, c, d, a, GET(3), 0xc1bdceee, 22)
    STEP(F, a, b, c, d, GET(4), 0xf57c0faf, 7)  STEP(F, d, a, b, c, GET(5), 0x4787c62a, 12)
    STEP(F, c, d, a, b, GET(6), 0xa8304613, 17) STEP(F, b, c, d, a, GET(7), 0xfd469501, 22)
    STEP(F, a, b, c, d, GET(8), 0x698098d8, 7)  STEP(F, d, a, b, c, GET(9), 0x8b44f7af, 12)
    STEP(F, c, d, a, b, GET(10), 0xffff5bb1, 17) STEP(F, b, c, d, a, GET(11), 0x895cd7be, 22)
    STEP(F, a, b, c, d, GET(12), 0x6b901122, 7) STEP(F, d, a, b, c, GET(13), 0xfd987193, 12)
    STEP(F, c, d, a, b, GET(14), 0xa679438e, 17) STEP(F, b, c, d, a, GET(15), 0x49b40821, 22)
    STEP(G, a, b, c, d, GET(1), 0xf61e2562, 5)  STEP(G, d, a, b, c, GET(6), 0xc040b340, 9)
    STEP(G, c, d, a, b, GET(11), 0x265e5a51, 14) STEP(G, b, c, d, a, GET(0), 0xe9b6c7aa, 20)
    STEP(G, a, b, c, d, GET(5), 0xd62f105d, 5)  STEP(G, d, a, b, c, GET(10), 0x02441453, 9)
    STEP(G, c, d, a, b, GET(15), 0xd8a1e681, 14) STEP(G, b, c, d, a, GET(4), 0xe7d3fbc8, 20)
    STEP(G, a, b, c, d, GET(9), 0x21e1cde6, 5)  STEP(G, d, a, b, c, GET(14), 0xc33707d6, 9)
    STEP(G, c, d, a, b, GET(3), 0xf4d50d87, 14) STEP(G, b, c, d, a, GET(8), 0x455a14ed, 20)
    STEP(G, a, b, c, d, GET(13), 0xa9e3e905, 5) STEP(G, d, a, b, c, GET(2), 0xfcefa3f8, 9)
    STEP(G, c, d, a, b, GET(7), 0x676f02d9, 14) STEP(G, b, c, d, a, GET(12), 0x8d2a4c8a, 20)
    STEP(H, a, b, c, d, GET(5), 0xfffa3942, 4)  STEP(H, d, a, b, c, GET(8), 0x8771f681, 11)
    STEP(H, c, d, a, b, GET(11), 0x6d9d6122, 16) STEP(H, b, c, d, a, GET(14), 0xfde5380c, 23)
    STEP(H, a, b, c, d, GET(1), 0xa4beea44, 4)  STEP(H, d, a, b, c, GET(4), 0x4bdecfa9, 11)
    STEP(H, c, d, a, b, GET(7), 0xf6bb4b60, 16) STEP(H, b, c, d, a, GET(10), 0xbebfbc70, 23)
    STEP(H, a, b, c, d, GET(13), 0x289b7ec6, 4) STEP(H, d, a, b, c, GET(0), 0xeaa127fa, 11)
    STEP(H, c, d, a, b, GET(3), 0xd4ef3085, 16) STEP(H, b, c, d, a, GET(6), 0x04881d05, 23)
    STEP(H, a, b, c, d, GET(9), 0xd9d4d039, 4)  STEP(H, d, a, b, c, GET(12), 0xe6db99e5, 11)
    STEP(H, c, d, a, b, GET(15), 0x1fa27cf8, 16) STEP(H, b, c, d, a, GET(2), 0xc4ac5665, 23)
    STEP(I, a, b, c, d, GET(0), 0xf4292244, 6)  STEP(I, d, a, b, c, GET(7), 0x432aff97, 10)
    STEP(I, c, d, a, b, GET(14), 0xab9423a7, 15) STEP(I, b, c, d, a, GET(5), 0xfc93a039, 21)
    STEP(I, a, b, c, d, GET(12), 0x655b59c3, 6) STEP(I, d, a, b, c, GET(3), 0x8f0ccc92, 10)
    STEP(I, c, d, a, b, GET(10), 0xffeff47d, 15) STEP(I, b, c, d, a, GET(1), 0x85845dd1, 21)
    STEP(I, a, b, c, d, GET(8), 0x6fa87e4f, 6)  STEP(I, d, a, b, c, GET(15), 0xfe2ce6e0, 10)
    STEP(I, c, d, a, b, GET(6), 0xa3014314, 15) STEP(I, b, c, d, a, GET(13), 0x4e0811a1, 21)
    STEP(I, a, b, c, d, GET(4), 0xf7537e82, 6)  STEP(I, d, a, b, c, GET(11), 0xbd3af235, 10)
    STEP(I, c, d, a, b, GET(2), 0x2ad7d2bb, 15) STEP(I, b, c, d, a, GET(9), 0xeb86d391, 21)
    hash[0] = a + 0x67452301; hash[1] = b + 0xefcdab89;
    hash[2] = c + 0x98badcfe; hash[3] = d + 0x10325476;
}

int main(void) {
    const int N = 100000000;
    uint32_t key[16];
    uint32_t hash[4];
    volatile uint32_t sink = 0;

    /* === Test 1: Pure scalar MD5 throughput === */
    memset(key, 0x41, 64);
    key[14] = 20 << 3; /* fake length */

    uint64_t t0 = current_micros();
    for (int i = 0; i < N; i++) {
        key[0] ^= i; /* prevent optimization */
        md5_scalar(key, hash);
        sink += hash[0];
    }
    uint64_t t1 = current_micros();
    double scalar_sec = (double)(t1 - t0) / 1e6;
    printf("Scalar MD5:  %d hashes in %.3fs = %.1f M/s\n", N, scalar_sec, N / scalar_sec / 1e6);

    /* === Test 2: Pure AVX2 MD5 throughput (8-wide) === */
    uint32_t keys8[8][16];
    uint32_t hashes8[8][4];
    for (int i = 0; i < 8; i++) memset(keys8[i], 0x41, 64);

    const uint32_t *kp[8];
    uint32_t *hp[8];
    for (int i = 0; i < 8; i++) { kp[i] = keys8[i]; hp[i] = hashes8[i]; }

    t0 = current_micros();
    for (int i = 0; i < N / 8; i++) {
        keys8[0][0] ^= i;
        md5_avx2_x8(kp, hp);
        sink += hashes8[0][0];
    }
    t1 = current_micros();
    double avx2_sec = (double)(t1 - t0) / 1e6;
    int avx2_total = (N / 8) * 8;
    printf("AVX2 MD5:    %d hashes in %.3fs = %.1f M/s (%.1fx scalar)\n",
           avx2_total, avx2_sec, avx2_total / avx2_sec / 1e6, (avx2_total / avx2_sec) / (N / scalar_sec));

    /* === Test 3: Pure AVX-512 MD5 throughput (16-wide) === */
#if defined(__x86_64__) && defined(__AVX512F__)
    uint32_t keys16[16][16];
    uint32_t hashes16[16][4];
    for (int i = 0; i < 16; i++) memset(keys16[i], 0x41, 64);

    const uint32_t *kp16[16];
    uint32_t *hp16[16];
    for (int i = 0; i < 16; i++) { kp16[i] = keys16[i]; hp16[i] = hashes16[i]; }

    t0 = current_micros();
    for (int i = 0; i < N / 16; i++) {
        keys16[0][0] ^= i;
        md5_avx512_x16(kp16, hp16);
        sink += hashes16[0][0];
    }
    t1 = current_micros();
    double avx512_sec = (double)(t1 - t0) / 1e6;
    int avx512_total = (N / 16) * 16;
    printf("AVX512 MD5:  %d hashes in %.3fs = %.1f M/s (%.1fx scalar, %.1fx AVX2)\n",
           avx512_total, avx512_sec, avx512_total / avx512_sec / 1e6,
           (avx512_total / avx512_sec) / (N / scalar_sec),
           (avx512_total / avx512_sec) / (avx2_total / avx2_sec));
#else
    printf("AVX512 MD5:  (not available on this platform)\n");
#endif

    /* === Test 4: construct_string cost (PUTCHAR_SCALAR) === */
    /* Simulate building a 4-word ~20 char string via PUTCHAR_SCALAR */
    const char *words = "tyranous\0pluto\0twits\0lot";
    int word_offsets[] = {0, 9, 15, 21};
    int word_lens[] = {8, 5, 5, 3};

    t0 = current_micros();
    for (int i = 0; i < N; i++) {
        memset(key, 0, 64);
        int wcs = 0;
        for (int w = 0; w < 4; w++) {
            int off = word_offsets[w];
            for (int c = 0; c < word_lens[w]; c++) {
                PUTCHAR_SCALAR(key, wcs, (uint8_t)words[off + c]);
                wcs++;
            }
            PUTCHAR_SCALAR(key, wcs, ' ');
            wcs++;
        }
        wcs--;
        PUTCHAR_SCALAR(key, wcs, 0x80);
        PUTCHAR_SCALAR(key, 56, wcs << 3);
        PUTCHAR_SCALAR(key, 57, wcs >> 5);
        sink += key[0];
    }
    t1 = current_micros();
    double str_sec = (double)(t1 - t0) / 1e6;
    printf("String PUTCHAR: %d strings in %.3fs = %.1f M/s\n", N, str_sec, N / str_sec / 1e6);

    /* === Test 5: construct_string cost (memcpy) === */
    t0 = current_micros();
    for (int i = 0; i < N; i++) {
        memset(key, 0, 64);
        char *dst = (char *)key;
        int wcs = 0;
        for (int w = 0; w < 4; w++) {
            memcpy(dst + wcs, words + word_offsets[w], word_lens[w]);
            wcs += word_lens[w];
            dst[wcs] = ' ';
            wcs++;
        }
        wcs--;
        dst[wcs] = (char)0x80;
        dst[56] = (char)(wcs << 3);
        dst[57] = (char)(wcs >> 5);
        sink += key[0];
    }
    t1 = current_micros();
    double str_fast_sec = (double)(t1 - t0) / 1e6;
    printf("String memcpy:  %d strings in %.3fs = %.1f M/s (%.1fx PUTCHAR)\n",
           N, str_fast_sec, N / str_fast_sec / 1e6, str_sec / str_fast_sec);

    printf("\nBottleneck: PUTCHAR string is %.1fx slower than scalar MD5\n", str_sec / scalar_sec);
    printf("            memcpy  string is %.1fx slower than scalar MD5\n", str_fast_sec / scalar_sec);
    printf("(sink=%u)\n", sink);
    return 0;
}
