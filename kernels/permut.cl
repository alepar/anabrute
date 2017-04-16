//#define __global
//#define __kernel

/* MD5 OpenCL kernel based on Solar Designer's MD5 algorithm implementation at:
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

#pragma OPENCL EXTENSION cl_khr_byte_addressable_store : disable

/* Macros for reading/writing chars from int32's (from rar_kernel.cl) */
#define GETCHAR(buf, index) (((uchar*)(buf))[(index)])
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
void md5(const uint *key, uint *hash)
{
    uint a, b, c, d;

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
#define MAX_OFFSETS_LENGTH 20

typedef struct {
    uint all_strs[MAX_STR_LENGTH/4];
    uint offsets[MAX_OFFSETS_LENGTH/4];  // positives - permutable, negatives - fixed, zeroes - empty; abs(offset)-1 to get offset in all_strs
    uint start_from;
} permut_template;

ulong fact(uchar x) {
    switch(x) {
        case 0: 	return 1L;
        case 1: 	return 1L;
        case 2: 	return 2L;
        case 3: 	return 6L;
        case 4: 	return 24L;
        case 5: 	return 120L;
        case 6: 	return 720L;
        case 7: 	return 5040L;
        case 8: 	return 40320L;
        case 9: 	return 362880L;
        case 10: 	return 3628800L;
        case 11: 	return 39916800L;
        case 12: 	return 479001600L;
        case 13: 	return 6227020800L;
        case 14: 	return 87178291200L;
        case 15: 	return 1307674368000L;
        case 16: 	return 20922789888000L;
        case 17: 	return 355687428096000L;
        case 18: 	return 6402373705728000L;
        case 19: 	return 121645100408832000L;
        case 20: 	return 2432902008176640000L;
        default:    return 0L;
    }
}

__kernel void permut(__global const permut_template *permut_templates, const uint iters_per_item, __global uint *hashes, uint hashes_num, __global uint *hashes_reversed) {
    ulong id = get_global_id(0);
    __global const permut_template *tmpl = &permut_templates[id];

    uint key[16];
    ulong a = tmpl->start_from;

    char all_strs[MAX_STR_LENGTH];
    uint *all_strs_uint = all_strs; // read as uints
    for (uchar i=0; i<MAX_STR_LENGTH/4; i++) {
        all_strs_uint[i] = tmpl->all_strs[i];
    }

    uchar offsets_len;
    char offsets[MAX_OFFSETS_LENGTH];
    uint *offsets_uint = offsets; // read as uints
    for (offsets_len=0; offsets_len<MAX_OFFSETS_LENGTH/4; offsets_len++) {
        offsets_uint[offsets_len] = tmpl->offsets[offsets_len];
    }
    for (offsets_len=0; offsets_len<MAX_OFFSETS_LENGTH && offsets[offsets_len]; offsets_len++);

    if (a>1) {
        uchar unpicked_len = 0;
        char unpicked[MAX_OFFSETS_LENGTH];
        uchar idx_unp=0;

        for (uchar i=0; i<offsets_len; i++) {
            if (offsets[i] > 0) {
                unpicked_len++;
                unpicked[idx_unp++] = offsets[i];
            }
        }
        unpicked_len = idx_unp;

        uchar wo=0;
        for (char d=unpicked_len-1; d>=0; d--) {
            ulong factd = fact(d);
            ulong ord = (a-1) / factd;
            a -= ord * factd;

            for (idx_unp=0; idx_unp<unpicked_len; idx_unp++) {
                if (unpicked[idx_unp]>0) {
                    if (ord == 0) {
                        for (; wo<offsets_len; wo++) {
                            if (offsets[wo] > 0) {
                                offsets[wo++] = unpicked[idx_unp];
                                break;
                            }
                        }

                        unpicked[idx_unp] = 0;
                        break;
                    } else {
                        ord--;
                    }
                }
            }
        }
    }

    uint counter=0;
    uint computed_hash[4];
    do {
        for (uchar ik=0; ik<16; ik++) {
            key[ik] = 0;
        }
        // construct key
        uchar wcs=0;
        for (uchar io=0; io<offsets_len; io++) {
            char off = offsets[io];
            if (off < 0) {
                off = -off;
            }
            off--;

            while (all_strs[off]) {
                PUTCHAR(key, wcs, all_strs[off]);
                wcs++; off++;
            }
            PUTCHAR(key, wcs, ' ');
            wcs++;
        }
        wcs--;
        // padding code (borrowed from MD5_eq.c)
        PUTCHAR(key, wcs, 0x80);
        PUTCHAR(key, 56, wcs << 3);
        PUTCHAR(key, 57, wcs >> 5);

        // calculate hash
        md5(key, computed_hash);

        // is hash a match?
        //TODO copy hashes to local mem? or screw it?
        for(uchar ih=0; ih<hashes_num; ih++) {
            uchar match = 1;
            for(uchar ihj=0; ihj<4; ihj++) {
                if(hashes[4*ih+ihj] != computed_hash[ihj]) {
                    match = 0;
                    break;
                }
            }

            if (match) {
                PUTCHAR(key, wcs, 0);
                for (uchar ihr=0; ihr<MAX_STR_LENGTH/4; ihr++) {
                    hashes_reversed[ih*(MAX_STR_LENGTH/4)+ihr]=key[ihr];
                }
                break;
            }
        }

        // find next permut if possible
        char k = -1;
        char k1 = -1;
        char found = 0;

        for (char io=offsets_len-1; io>=0; io--) {
            if (offsets[io]>0) {
                k1 =k;
                k = io;

                if (k1 != -1 && k!=-1 && offsets[k]<offsets[k1]) {
                    found = 1;
                    break;
                }
            }
        }

        if (!found) {
            break;
        }

        char l;
        for (l=offsets_len-1; l>k; l--) {
            if (offsets[l]>offsets[k]) {
                break;
            }
        }

        offsets[l] ^= offsets[k];
        offsets[k] ^= offsets[l];
        offsets[l] ^= offsets[k];

        char li=k1, ri=offsets_len-1;
        while(li<ri) {
            while(offsets[li]<=0 && li<ri) li++;
            while(offsets[ri]<=0 && li<ri) ri--;

            if (li < ri) {
                offsets[li] ^= offsets[ri];
                offsets[ri] ^= offsets[li];
                offsets[li] ^= offsets[ri];

                li++; ri--;
            }
        }

        counter++;
    } while (counter < iters_per_item);
}
