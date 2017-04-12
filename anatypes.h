#ifndef __ANATYPES_H__
#define __ANATYPES_H__

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

bool char_counts_create(const char *s, char_counts *cc);
uint8_t char_counts_equal(char_counts *l, char_counts *r);
bool char_counts_contains(char_counts* cc, char_counts* subcc);

bool char_counts_strings_create(const char *s, char_counts_strings *ccs);
bool char_counts_strings_addstring(char_counts_strings *ccs, const char *s);

#endif