#include "hashes.h"

void hash_to_ascii(const uint32_t *hash, char *buf) {
    int di = 0;
    for(int si=0; si<4; si++) {
        buf[di++] = (hash[si] & 0x000000f0) >>  4;
        buf[di++] = (hash[si] & 0x0000000f)      ;

        buf[di++] = (hash[si] & 0x0000f000) >> 12;
        buf[di++] = (hash[si] & 0x00000f00) >>  8;

        buf[di++] = (hash[si] & 0x00f00000) >> 20;
        buf[di++] = (hash[si] & 0x000f0000) >> 16;

        buf[di++] = (hash[si] & 0xf0000000) >> 28;
        buf[di++] = (hash[si] & 0x0f000000) >> 24;
    }

    for(int i=0; i<32; i++) {
        if (buf[i] > 9) {
            buf[i] += 'a' - 10;
        } else {
            buf[i] += '0';
        }
    }

    buf[di] = 0;
}

void ascii_to_hash(const char *buf, uint32_t *hash) {
    char *hash_bytes = (char *)hash;
    for (int i=0; i<16; i++) {
        char l = buf[i*2], r = buf[i*2+1];
        if (l > '9') l-= 'a' - 10;
        else l-='0';
        if (r > '9') r-= 'a' - 10;
        else r-='0';
        hash_bytes[i] = l<<4 | r;
    }
}
