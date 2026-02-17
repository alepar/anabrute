/* MD5 Metal kernel based on Solar Designer's MD5 algorithm implementation at:
 * http://openwall.info/wiki/people/solar/software/public-domain-source-code/md5
 *
 * This software is Copyright (c) 2010, Dhiru Kholia <dhiru.kholia at gmail.com>,
 * and it is hereby released to the general public under the following terms:
 * Redistribution and use in source and binary forms, with or without modification,
 * are permitted.
 *
 * Useful References:
 * 1. CUDA MD5 Hashing Experiments, http://majuric.org/software/cudamd5/
 * 2. oclcrack, http://sghctoma.extra.hu/index.php?p=entry&id=11
 * 3. http://people.eku.edu/styere/Encrypt/JS-MD5.html
 * 4. http://en.wikipedia.org/wiki/MD5#Algorithm */

#include <metal_stdlib>
using namespace metal;

/* Macros for reading/writing chars from int32's (from rar_kernel.cl) */
#define GETCHAR(buf, index) (((device uint8_t*)(buf))[(index)])
#define GETCHAR_L(buf, index) (((thread uint8_t*)(buf))[(index)])
#define PUTCHAR(buf, index, val) (buf)[(index)>>2] = ((buf)[(index)>>2] & ~(0xffU << (((index) & 3) << 3))) + ((val) << (((index) & 3) << 3))

/* The basic MD5 functions */
#define F(x, y, z)			((z) ^ ((x) & ((y) ^ (z))))
#define G(x, y, z)			((y) ^ ((z) & ((x) ^ (y))))
#define H(x, y, z)			((x) ^ (y) ^ (z))
#define I(x, y, z)			((y) ^ ((x) | ~(z)))

/* The MD5 transformation for all four rounds. */
#define STEP(f, a, b, c, d, x, t, s) \
    (a) += f((b), (c), (d)) + (x) + (t); \
    (a) = (((a) << (s)) | (((a) & 0xffffffff) >> (32 - (s)))); \
    (a) += (b);

#define GET(i) (key[(i)])

/*
 * @param key - char string grouped into 16 uint's (little endian)
 * @param hash - output for MD5 hash of a key (4 uint's).
 */
static void md5(const thread uint32_t *key, thread uint32_t *hash)
{
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

// ====================
// === permutations ===
// ====================

#define MAX_STR_LENGTH 40
#define MAX_OFFSETS_LENGTH 16

typedef struct permut_task_s {
    char all_strs[MAX_STR_LENGTH];
    char offsets[MAX_OFFSETS_LENGTH];
    uint8_t a[MAX_OFFSETS_LENGTH];
    uint8_t c[MAX_OFFSETS_LENGTH];
    uint16_t i;
    uint16_t n;
    uint32_t iters_done;
} permut_task;

static uint64_t fact(uint8_t x) {
    switch(x) {
        case 0: 	return 1UL;
        case 1: 	return 1UL;
        case 2: 	return 2UL;
        case 3: 	return 6UL;
        case 4: 	return 24UL;
        case 5: 	return 120UL;
        case 6: 	return 720UL;
        case 7: 	return 5040UL;
        case 8: 	return 40320UL;
        case 9: 	return 362880UL;
        case 10: 	return 3628800UL;
        case 11: 	return 39916800UL;
        case 12: 	return 479001600UL;
        case 13: 	return 6227020800UL;
        case 14: 	return 87178291200UL;
        case 15: 	return 1307674368000UL;
        case 16: 	return 20922789888000UL;
        case 17: 	return 355687428096000UL;
        case 18: 	return 6402373705728000UL;
        case 19: 	return 121645100408832000UL;
        case 20: 	return 2432902008176640000UL;
        default:    return 0UL;
    }
}

#define MAX_HASHES 32

kernel void permut(
    device permut_task *tasks [[buffer(0)]],
    constant uint32_t &iters_per_task [[buffer(1)]],
    constant uint32_t *hashes [[buffer(2)]],
    constant uint32_t &hashes_num [[buffer(3)]],
    device uint32_t *hashes_reversed [[buffer(4)]],
    uint id [[thread_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]])
{
    // GPU-4: cooperative load of target hashes into threadgroup memory
    threadgroup uint32_t local_hashes[MAX_HASHES * 4];
    for (uint i = tid; i < hashes_num * 4; i += tg_size) {
        local_hashes[i] = hashes[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    permut_task task = tasks[id];

    if (task.i >= task.n) {
        return;
    }

    // GPU-1: precompute word lengths (constant across all permutations)
    uint8_t wlen[MAX_STR_LENGTH];
    // lengths for fixed words (negative offsets)
    for (uint8_t io = 0; task.offsets[io]; io++) {
        if (task.offsets[io] < 0) {
            uint8_t byte_off = -task.offsets[io] - 1;
            uint8_t l = 0;
            while (task.all_strs[byte_off + l]) l++;
            wlen[byte_off] = l;
        }
    }
    // lengths for permutable words (a[] entries)
    for (uint8_t ai = 0; ai < task.n; ai++) {
        uint8_t byte_off = task.a[ai] - 1;
        uint8_t l = 0;
        while (task.all_strs[byte_off + l]) l++;
        wlen[byte_off] = l;
    }

    // Precompute total string length (constant across permutations)
    uint8_t str_len = 0;
    uint8_t num_words = 0;
    for (uint8_t io = 0; task.offsets[io]; io++) {
        char soff = task.offsets[io];
        uint8_t off = (soff < 0) ? (-soff - 1) : (task.a[soff - 1] - 1);
        str_len += wlen[off];
        num_words++;
    }
    str_len += num_words - 1; // spaces between words

    // GPU-3: zero key once and set constant MD5 padding
    uint32_t key[16];
    for (uint8_t ik = 0; ik < 16; ik++) key[ik] = 0;
    thread uint8_t *key_bytes = (thread uint8_t*)key;
    key_bytes[str_len] = 0x80;
    key_bytes[56] = str_len << 3;
    key_bytes[57] = str_len >> 5;

    uint32_t iter_counter = 0;
    uint32_t computed_hash[4];
    while (true) {
        if (iter_counter >= iters_per_task) break;

        // GPU-1 + GPU-2: construct key with precomputed lengths + native byte writes
        uint8_t wcs = 0;
        for (uint8_t io = 0; task.offsets[io]; io++) {
            if (wcs > 0) key_bytes[wcs++] = ' ';
            char soff = task.offsets[io];
            uint8_t off;
            if (soff < 0) {
                off = -soff - 1;
            } else {
                off = task.a[soff - 1] - 1;
            }
            uint8_t len = wlen[off];
            for (uint8_t j = 0; j < len; j++) {
                key_bytes[wcs++] = task.all_strs[off + j];
            }
        }

        md5(key, computed_hash);

        // GPU-7 + GPU-4: early-exit hash comparison against threadgroup-cached hashes
        for (uint8_t ih = 0; ih < hashes_num; ih++) {
            if (local_hashes[4 * ih] != computed_hash[0]) continue;
            if (local_hashes[4 * ih + 1] != computed_hash[1]) continue;
            if (local_hashes[4 * ih + 2] != computed_hash[2]) continue;
            if (local_hashes[4 * ih + 3] != computed_hash[3]) continue;

            // match — write result
            key_bytes[str_len] = 0; // null-terminate for output
            for (uint8_t ihr = 0; ihr < MAX_STR_LENGTH / 4; ihr++) {
                hashes_reversed[ih * (MAX_STR_LENGTH / 4) + ihr] = key[ihr];
            }
            key_bytes[str_len] = 0x80; // restore padding
            break;
        }

        // Heap's algorithm: find next permutation
        bool found_next = false;
        while (task.i < task.n) {
            if (task.c[task.i] < task.i) {
                if (task.i % 2 == 0) {
                    task.a[0] ^= task.a[task.i];
                    task.a[task.i] ^= task.a[0];
                    task.a[0] ^= task.a[task.i];
                } else {
                    task.a[task.c[task.i]] ^= task.a[task.i];
                    task.a[task.i] ^= task.a[task.c[task.i]];
                    task.a[task.c[task.i]] ^= task.a[task.i];
                }

                task.c[task.i]++;
                task.i = 0;
                iter_counter++;
                found_next = true;
                break;
            } else {
                task.c[task.i] = 0;
                task.i++;
            }
        }

        if (!found_next) break;
    }

    task.iters_done += iter_counter;

    // Write back mutable state
    for (uint8_t idx = 0; idx < MAX_OFFSETS_LENGTH; idx++) {
        tasks[id].a[idx] = task.a[idx];
        tasks[id].c[idx] = task.c[idx];
    }
    tasks[id].i = task.i;
    tasks[id].n = task.n;
    tasks[id].iters_done = task.iters_done;
}
