#ifndef ANABRUTE_HASHES_H
#define ANABRUTE_HASHES_H

#include "common.h"

void hash_to_ascii(const uint32_t *hash, char *buf);

void ascii_to_hash(const char *buf, uint32_t *hash);

#endif //ANABRUTE_HASHES_H
