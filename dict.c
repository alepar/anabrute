#include "dict.h"

int read_dict(const char *filename, char_counts_strings *dict, uint32_t *dict_length, char_counts *seed_phrase) {
    FILE *dictFile = fopen(filename, "r");
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
        if (len > 0 && (str[len-1] == '\n' || str[len-1] == '\r')) {
            str[len-1] = 0;
        }
        if (len > 1 && (str[len-2] == '\n' || str[len-2] == '\r')) {
            str[len-2] = 0;
        }

        if (strlen(str) == 0) {
            continue;
        }

        if (strcmp(buflines[0], buflines[1])) {
            lineidx = 1-lineidx;
            if (char_counts_strings_create(str, &dict[*dict_length])) {
                char_counts_strings_free(&dict[*dict_length]);
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
                        fclose(dictFile);
                        return -2;
                    }
                } else {
                    /* Anagram duplicate: free the abandoned entry */
                    char_counts_strings_free(&dict[*dict_length]);
                }

                if (char_counts_strings_addstring(&dict[i], str)) {
                    fprintf(stderr, "strings overflow! %d", dict[i].strings_len);
                    fclose(dictFile);
                    return -3;
                }
            } else {
                /* Word not contained in seed phrase: free the abandoned entry */
                char_counts_strings_free(&dict[*dict_length]);
            }
        }
    }
    fclose(dictFile);
    return 0;
}
