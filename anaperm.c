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

typedef struct {
    char* str;
    int8_t count;
} string_and_count;

typedef struct {
    int8_t offset;
    int8_t count;
} string_idx_and_count;

void print_scs(const string_and_count *scs, const int scs_len) {
    for (int i=0; i < scs_len; i++) {
        if (scs[i].count) {
            printf("%s", scs[i].str);
            if (scs[i].count>1) {
                printf("*%d", scs[i].count);
            }
            printf(" ");
        }
    }
    printf("\n");
}

void print_permut(int8_t permut[], int len, char *strs) {
    printf("\t");
    for (int i=0; i<len; i++) {
        if (permut[i] < 0) {
            printf("%s ", &strs[-permut[i]-1]);
        } else if (permut[i] > 0) {
            printf("*%s ", &strs[permut[i]-1]);
        } else {
            printf("* ");
        }
    }
    printf("\n");
}

void print_permut_permut(int8_t *permut, int len, char *strs) {
    printf("\t");
    print_permut(permut, len, strs);
}

void iter_permuts(int8_t permut[], int len, char *strs) {
    print_permut_permut(permut, len, strs);
    while(1) {
        int k = -1;
        int k1 = -1;
        bool found = false;

        for (int i=len-1; i>=0; i--) {
            if (permut[i]>0) {
                k1 = k;
                k = i;
                if (k1!=-1 && k!=-1 && permut[k]<permut[k1]) {
                    found = true;
                    break;
                }
            }
        }

        if (!found) {
            return;
        }

        int l;
        for (l=len-1; l>k; l--) {
            if (permut[l]>permut[k]) {
                break;
            }
        }

        permut[l] ^= permut[k];
        permut[k] ^= permut[l];
        permut[l] ^= permut[k];

        int li=k1, ri=len-1;
        while (1) {
            while(permut[li]<=0) li++;
            while(permut[ri]<=0) ri--;

            if (li < ri) {
                permut[li] ^= permut[ri];
                permut[ri] ^= permut[li];
                permut[li] ^= permut[ri];

                li++; ri--;
            } else {
                break;
            }
        }
        print_permut_permut(permut, len, strs);
    }
}

void recurse_combs(char *all_strs, string_idx_and_count sics[], int sics_len, int sics_idx, int8_t permut[], int permut_len, int start_idx) {
    if (sics_idx > sics_len) {
        int si, di=0;
        for (si = 0; si < sics_len; si++) {
            if (sics[si].count) {
                for (;permut[di];di++);
                permut[di] = sics[si].offset+1;
            }
        }
        print_permut(permut, permut_len, all_strs);
        iter_permuts(permut, permut_len, all_strs);
        for (di=0; di<permut_len; di++) {
            if (permut[di] > 0) {
                permut[di] = 0;
            }
        }
    } else if (start_idx > permut_len && sics[sics_idx].count > 0) {
        // bail out
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

            scs[scs_idx].str = stack->ccs->strings[string_idx];
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

int main(int argc, char *argv[]) {
    char_counts_strings ccs[5];

    char *strs0[] = {"up", "pu"};
    ccs[0].strings = strs0;
    ccs[0].strings_len = sizeof(strs0) / sizeof(char*);

    char *strs1[] = {"s"};
    ccs[1].strings = strs1;
    ccs[1].strings_len = sizeof(strs1) / sizeof(char*);

    char *strs2[] = {"t"};
    ccs[2].strings = strs2;
    ccs[2].strings_len = sizeof(strs2) / sizeof(char*);

    char *strs3[] = {"x"};
    ccs[3].strings = strs3;
    ccs[3].strings_len = sizeof(strs3) / sizeof(char*);

    char *strs4[] = {"y"};
    ccs[4].strings = strs4;
    ccs[4].strings_len = sizeof(strs4) / sizeof(char*);


    stack_item stack[5];

    stack[0].count = 3;
    stack[0].ccs = &ccs[0];

    stack[1].count = 2;
    stack[1].ccs = &ccs[1];

    stack[2].count = 1;
    stack[2].ccs = &ccs[2];

    stack[3].count = 1;
    stack[3].ccs = &ccs[3];

    stack[4].count = 1;
    stack[4].ccs = &ccs[4];

    string_and_count scs[6];

    recurse_string_combs(stack, 5, 0, 0, scs, 0);
}