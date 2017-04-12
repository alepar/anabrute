#include <memory.h>
#include <stdlib.h>

#include "anatypes.h"

bool char_counts_create(const char *s, char_counts *cc) {
    memset(cc->counts, 0, CHARCOUNT);
    cc->length = 0;

    for (int i=0; s[i]; i++) {
        cc->length++;
        int idx = char_to_index(s[i]);
        if (idx >= 0) {
            cc->counts[idx]++;
        } else {
            return true;
        }
    }

    return false;
}

bool char_counts_subtract(char_counts *from, char_counts *what) {
    if (from->length < what->length) {
        return false;
    }
    from->length = from->length - what->length;

    for(int i=0; i<CHARCOUNT; i++) {
        if (from->counts[i] < what->counts[i]) {
            return false;
        }
        from->counts[i] = from->counts[i] - what->counts[i];
    }

    return true;
}

void char_counts_copy(char_counts *src, char_counts *dst) {
    dst->length = src->length;

    for(int i=0; i<CHARCOUNT; i++) {
        dst->counts[i] = src->counts[i];
    }
}

uint8_t char_counts_equal(char_counts *l, char_counts *r) {
    if (l->length != r->length) {
        return 0;
    }

    for(int i=0; i<CHARCOUNT; i++) {
        if (l->counts[i] != r->counts[i]) {
            return 0;
        }
    }

    return 1;
}

bool char_counts_strings_create(const char *s, char_counts_strings *ccs) {
    ccs->strings_len = 0;
    ccs->strings = malloc(sizeof(char*)*MAX_STRINGS_SIZE);
    return char_counts_create(s, &ccs->counts);
}

bool char_counts_strings_addstring(char_counts_strings *ccs, const char *s) {
    const size_t len = strlen(s) + 1;
    char* sc = malloc(len);
    memcpy(sc, s, len);
    ccs->strings[ccs->strings_len++] = sc;
    return ccs->strings_len > MAX_STRINGS_SIZE;
}

bool char_counts_contains(char_counts* cc, char_counts* subcc) {
    if (cc->length < subcc->length) {
        return false;
    }

    for (int i=0; i<CHARCOUNT; i++) {
        if (cc->counts[i] < subcc->counts[i]) {
            return false;
        }
    }

    return true;
}