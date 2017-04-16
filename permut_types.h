#ifndef __PERMUT_TYPES_H__
#define __PERMUT_TYPES_H__

#include <stdbool.h>
#include <stdint.h>

#ifdef __APPLE__
#include <unitypes.h>
#endif

#include "seedphrase.h"

#define MAX_STRINGS_SIZE 1024
#define MAX_DICT_SIZE 2048

typedef struct {
    uint8_t counts[CHARCOUNT];
    uint8_t length;
} char_counts;

typedef struct {
    char_counts counts;
    char **strings;
    int strings_len;
} char_counts_strings;

typedef struct {
    char_counts_strings* ccs;
    uint8_t count;
} stack_item;

typedef struct {
    char* str;
    int8_t count;
} string_and_count;

typedef struct {
    int8_t offset;
    int8_t count;
} string_idx_and_count;

bool char_counts_create(const char *s, char_counts *cc);
uint8_t char_counts_equal(char_counts *l, char_counts *r);
bool char_counts_contains(char_counts* cc, char_counts* subcc);
bool char_counts_subtract(char_counts *from, char_counts *what);
void char_counts_copy(char_counts *src, char_counts *dst);

bool char_counts_strings_create(const char *s, char_counts_strings *ccs);
bool char_counts_strings_addstring(char_counts_strings *ccs, const char *s);

#endif