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

stack_item stack[20];
string_and_count scs[120];
char_counts_strings* dict_by_char[CHARCOUNT][MAX_DICT_SIZE];
int dict_by_char_len[CHARCOUNT] = {0};

FILE *permuts;

void print_scs(const string_and_count *scs, const int scs_len) {
    fprintf(permuts, "\t");
    for (int i=0; i < scs_len; i++) {
        if (scs[i].count) {
            fprintf(permuts, "%s", scs[i].str);
            if (scs[i].count>1) {
                fprintf(permuts, "*%d", scs[i].count);
            }
            fprintf(permuts, " ");
        }
    }
    fprintf(permuts, "\n");
}

void print_permut(int8_t permut[], int len, char *strs) {
    fprintf(permuts, "\t\t");
    for (int i=0; i<len; i++) {
        if (permut[i] < 0) {
            fprintf(permuts, "%s ", &strs[-permut[i]-1]);
        } else if (permut[i] > 0) {
            fprintf(permuts, "*%s ", &strs[permut[i]-1]);
        } else {
            fprintf(permuts, "* ");
        }
    }
    fprintf(permuts, "\n");
}

void recurse_combs(char *all_strs, string_idx_and_count sics[], int sics_len, int sics_idx, int8_t permut[], int permut_len, int start_idx) {
    if (sics_idx >= sics_len) {
        int si, di=0;
        for (si = 0; si < sics_len; si++) {
            if (sics[si].count) {
                for (;permut[di];di++);
                permut[di] = sics[si].offset+1;
            }
        }
        print_permut(permut, permut_len, all_strs);
// TODO       iter_permuts(permut, permut_len, all_strs);
        for (di=0; di<permut_len; di++) {
            if (permut[di] > 0) {
                permut[di] = 0;
            }
        }
    } else if (start_idx > permut_len && sics[sics_idx].count > 0) {
        fprintf(permuts, "\t\tbailout\n");
    } else if (sics[sics_idx].count > 1 || start_idx > 0) {
        for (int i=start_idx; i<permut_len; i++) {
            if (permut[i] == 0) {
                permut[i] = -sics[sics_idx].offset-1;
                sics[sics_idx].count--;

                if (sics[sics_idx].count == 0) {
                    recurse_combs(all_strs, sics, sics_len, sics_idx+1, permut, permut_len, 0);
                } else if (sics[sics_idx].count <= permut_len-i-1) {
                    recurse_combs(all_strs, sics, sics_len, sics_idx, permut, permut_len, i+1);
                }

                sics[sics_idx].count++;
                permut[i] = 0;
            }
        }
    } else {
        recurse_combs(all_strs, sics, sics_len, sics_idx+1, permut, permut_len, 0);
    }
}

void recurse_string_combs(stack_item *stack, int stack_len, int stack_idx, int string_idx, string_and_count *scs, int scs_idx) {
    if (stack_idx >= stack_len) {
        print_scs(scs, scs_idx);

        string_idx_and_count sics[scs_idx];

        uint8_t strs_count = 0;
        for (int i=0; i<scs_idx; i++) {
            strs_count += strlen(scs[i].str)+1;
        }

        uint8_t word_count = 0;
        char all_strs[strs_count];
        int8_t all_offs=0;
        for (int i=0; i<scs_idx; i++) {
            word_count += scs[i].count;
            sics[i].count = scs[i].count;
            sics[i].offset = all_offs;
            for (int j=0; j<=strlen(scs[i].str); j++) {
                all_strs[all_offs++] = scs[i].str[j];
            }
        }

        int8_t permut[word_count];
        memset(permut, 0, word_count);
        recurse_combs(all_strs, sics, scs_idx, 0, permut, word_count, 0);
    } else if (stack[stack_idx].ccs->strings_len > string_idx+1) {
        const uint8_t orig_count = stack[stack_idx].count;
        for (uint8_t i=0; i <= orig_count; i++) {
            stack[stack_idx].count = orig_count-i;

            scs[scs_idx].str = stack[stack_idx].ccs->strings[string_idx];
            scs[scs_idx].count = i;
            recurse_string_combs(stack, stack_len, stack_idx, string_idx + 1, scs, scs_idx + 1);
        }
        stack[stack_idx].count = orig_count;
    } else {
        scs[scs_idx].str = stack[stack_idx].ccs->strings[string_idx];
        scs[scs_idx].count = stack[stack_idx].count;
        recurse_string_combs(stack, stack_len, stack_idx + 1, 0, scs, scs_idx + 1);
    }
}

void recurse_dict_words(char_counts remainder, int curchar, int curdictidx, int stack_len) {
/*    printf("\t%d\t%d\t%d\t%d\t||\t", remainder.length, curchar, curdictidx, stack_len);
    for (int i=0; i<stack_len; i++) {
        printf("%s", stack[i].ccs->strings[0]);
        if (stack[i].count > 1) {
            printf("*%d ", stack[i].count);
        } else {
            printf(" ");
        }
    }
    printf("\n");*/

    if (remainder.length == 0) {
        for (int i=0; i<stack_len; i++) {
            fprintf(permuts, "%s", stack[i].ccs->strings[0]);
            if (stack[i].count > 1) {
                fprintf(permuts, "*%d ", stack[i].count);
            } else {
                fprintf(permuts, " ");
            }
        }
        fprintf(permuts, "\n");
        recurse_string_combs(stack, stack_len, 0, 0, scs, 0);

        return;
    }

    if(curchar >= CHARCOUNT) {
        return;
    }

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

            recurse_dict_words(next_remainder, next_char, next_idx, stack_len + 1);
        }
    }

    return;
}

int read_dict(char_counts_strings *dict, uint32_t *dict_length, char_counts *seed_phrase) {
    FILE *dictFile = fopen("input.dict", "r");
    if (!dictFile) {
        fprintf(stderr, "dict file not found!\n");
        return -1;
    }

    char buf1[100] = {0}, buf2[100] = {0};
    char *buflines[] = {buf1, buf2};
    int lineidx = 0;

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
            if (char_counts_strings_create(str, &dict[*dict_length])) {
                continue;
            }

            if (char_counts_contains(seed_phrase, &dict[*dict_length].counts)) {
                int i;

                for (i=0; i<*dict_length; i++) {
                    if (char_counts_equal(&dict[i].counts, &dict[*dict_length].counts)) {
                        break;
                    }
                }
                if (i==*dict_length) {
                    (*dict_length)++;
                    if (*dict_length > MAX_DICT_SIZE) {
                        fprintf(stderr, "dict overflow! %d\n", *dict_length);
                        return -2;
                    }
                }

                if (char_counts_strings_addstring(&dict[i], str)) {
                    fprintf(stderr, "strings overflow! %d", dict[i].strings_len);
                    return -3;
                }
            }
        }
    }
    fclose(dictFile);
}

int main(int argc, char *argv[]) {

    permuts = fopen("output.permuts", "w");

    char_counts seed_phrase;
    char_counts_create(seed_phrase_str, &seed_phrase);

    char_counts_strings dict[MAX_DICT_SIZE];
    uint32_t dict_length = 0;

    read_dict(dict, &dict_length, &seed_phrase);

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

    recurse_dict_words(seed_phrase, 0, 0, 0);

    printf("done\n");

    fclose(permuts);
}