#include <errno.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <stdbool.h>
#include <stdlib.h>

#include "anatypes.h"
#include "fact.h"

stack_item stack[20];
char_counts_strings* dict_by_char[CHARCOUNT][MAX_DICT_SIZE];
int dict_by_char_len[CHARCOUNT] = {0};

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

uint64_t recurse(char_counts remainder, int curchar, int curdictidx, int stack_len) {
    printf("\t%d\t%d\t%d\t%d\t||\t", remainder.length, curchar, curdictidx, stack_len);
    for (int i=0; i<stack_len; i++) {
        printf("%s", stack[i].ccs->strings[0]);
        if (stack[i].count > 1) {
            printf("*%d ", stack[i].count);
        } else {
            printf(" ");
        }
    }
    printf("\n");

    if (remainder.length == 0) {
        int word_count = 0;
        for (int i=0; i<stack_len; i++) {
            word_count += stack[i].count;
        }

        uint64_t total = fact(word_count);
        for (int i=0; i<stack_len; i++) {
            total /= fact(stack[i].count);

            for(int ii=0; ii<stack[i].count; ii++) {
                total *= stack[i].ccs->strings_len;
            }
        }

        return total;
    }

    if(curchar >= CHARCOUNT) {
        return 0;
    }

    uint64_t total = 0;

    for (int i=curdictidx; i<dict_by_char_len[curchar]; i++) {
        stack[stack_len].ccs = dict_by_char[curchar][i];

        char_counts next_remainder;
        char_counts_copy(&remainder, &next_remainder);
        for (uint8_t ccs_count=1; char_counts_subtract(&next_remainder, &dict_by_char[curchar][i]->counts); ccs_count++) {
            stack[stack_len].count = ccs_count;

            int next_char = curchar;
            int next_idx = i+1;

            if(next_remainder.counts[next_char] == 0) {
                next_char++;
                next_idx = 0;
            }

            const uint64_t local = recurse(next_remainder, next_char, next_idx, stack_len + 1);
            total += local;

            if (total < local) {
                fprintf(stderr, "counter overflow :(\n");
            }
        }
    }

    return total;
}

int main(int argc, char *argv[]) {
    char_counts seed_phrase;
    char_counts_create(seed_phrase_str, &seed_phrase);

    FILE *dictFile = fopen("wordlist.txt", "r");
    if (!dictFile) {
        fprintf(stderr, "dict file not found!\n");
        return -1;
    }

    char_counts_strings dict[MAX_DICT_SIZE];
    uint32_t dict_length = 0;

    char buf1[100] = {0}, buf2[100] = {0};
    char *buflines[] = {buf1, buf2};
    uint8_t lineidx = 0;

    int max_strings_len = 0;

    while(fgets(buflines[lineidx], 100, dictFile) != NULL) {
        char *const str = buflines[lineidx];
        const size_t len = strlen(str);
        if (str[len-1] == '\n' || str[len-1] == '\r') {
            str[len-1] = 0;
        }
        if (str[len-2] == '\n' || str[len-2] == '\r') {
            str[len-2] = 0;
        }

        if (strcmp(buflines[0], buflines[1])) {
            lineidx = 1-lineidx;
            if (char_counts_strings_create(str, &dict[dict_length])) {
                continue;
            }

            if (char_counts_contains(&seed_phrase, &dict[dict_length].counts)) {
                int i;

                for (i=0; i<dict_length; i++) {
                    if (char_counts_equal(&dict[i].counts, &dict[dict_length].counts)) {
                        break;
                    }
                }
                if (i==dict_length) {
                    dict_length++;
                    if (dict_length > MAX_DICT_SIZE) {
                        fprintf(stderr, "dict overflow! %d\n", dict_length);
                        return -2;
                    }
                }

                if (char_counts_strings_addstring(&dict[i], str)) {
                    fprintf(stderr, "strings overflow! %d", dict[i].strings_len);
                    return -3;
                }

                if (dict[i].strings_len > max_strings_len) {
                    max_strings_len = dict[i].strings_len;
                }
            }
        }
    }

    printf("max_strings_len %d\n", max_strings_len);

    for (int i=0; i<dict_length; i++) {
        for (int ci=0; ci<CHARCOUNT; ci++) {
            if (dict[i].counts.counts[ci]) {
                dict_by_char[ci][dict_by_char_len[ci]++] = &dict[i];
                break;
            }
        }
    }

    // todo resort dict_by_char?
    //   by length or char occurs?

    uint64_t total = recurse(seed_phrase, 0, 0, 0);

    printf("done: %lu\n", total);
}