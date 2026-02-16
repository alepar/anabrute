#ifndef ANABRUTE_HASHES_H
#define ANABRUTE_HASHES_H

#include "common.h"

void hash_to_ascii(const uint32_t *hash, char *buf);

void ascii_to_hash(const char *buf, uint32_t *hash);

uint32_t read_hashes(const char *file_name, uint32_t **hashes);

#endif //ANABRUTE_HASHES_H
