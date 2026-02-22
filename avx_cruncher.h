#ifndef ANABRUTE_AVX_CRUNCHER_H
#define ANABRUTE_AVX_CRUNCHER_H

#include "cruncher.h"

extern cruncher_ops avx512_cruncher_ops;
extern cruncher_ops avx2_cruncher_ops;
extern cruncher_ops scalar_cruncher_ops;

/* Shared between avx_cruncher.c and avx_cruncher_avx512.c */
#define PUTCHAR_SCALAR(buf, index, val) \
    (buf)[(index) >> 2] = ((buf)[(index) >> 2] & ~(0xffU << (((index) & 3) << 3))) + ((uint32_t)(val) << (((index) & 3) << 3))

void avx_check_hashes(cruncher_config *cfg, uint32_t *hash, uint32_t *key, int wcs);
void avx512_md5_check(cruncher_config *cfg,
                      uint32_t keys[16][16], int wcs_arr[16], int count);

#endif //ANABRUTE_AVX_CRUNCHER_H
