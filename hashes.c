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

uint32_t read_hashes(const char *file_name, uint32_t **hashes) {
    FILE *const fd = fopen(file_name, "r");
    if (!fd) {
        return 0;
    }

    fseek(fd, 0L, SEEK_END);
    const uint32_t file_size = (const uint32_t) ftell(fd);
    rewind(fd);

    const uint32_t hashes_num_est = (file_size + 1) / 33;
    uint32_t hashes_num = 0;

    *hashes = malloc(hashes_num_est*16);

    char buf[128];
    while(fgets(buf, sizeof(buf), fd) != NULL) {
        for (int i=0; i<sizeof(buf); i++) {
            if (buf[i] == '\n' || buf[i] == '\r') {
                buf[i] = 0;
            }
        }
        if (strlen(buf) != 32) {
            fprintf(stderr, "not a hash! (%s)\n", buf);
            continue;
        }

        if (hashes_num>=hashes_num_est) {
            fprintf(stderr, "too many hashes? skipping tail...\n");
            break;
        }

        ascii_to_hash(buf, &((*hashes)[hashes_num*4]));
        hashes_num++;
    }

    fclose(fd);
    return hashes_num;
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
