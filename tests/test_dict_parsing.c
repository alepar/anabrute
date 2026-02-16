#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include "dict.h"
#include "seedphrase.h"

static void write_file(const char *path, const char *content) {
    FILE *f = fopen(path, "w");
    assert(f && "failed to create test file");
    fputs(content, f);
    fclose(f);
}

/*
 * Test 1: Load a small dict with words from the seed phrase alphabet.
 * Seed phrase is "tyranousplutotwits" (chars: s,t,a,o,i,r,n,p,l,u,y,w).
 */
void test_basic_loading(void) {
    const char *path = "/tmp/anabrute_test_dict_basic.txt";
    write_file(path, "out\nrun\npots\n");

    char_counts seed;
    char_counts_create(seed_phrase_str, &seed);

    char_counts_strings dict[MAX_DICT_SIZE];
    uint32_t dict_length = 0;

    int err = read_dict(path, dict, &dict_length, &seed);
    assert(err == 0);
    assert(dict_length > 0 && "should load at least one word");

    /* All loaded words should be containable in the seed phrase */
    for (uint32_t i = 0; i < dict_length; i++) {
        assert(char_counts_contains(&seed, &dict[i].counts));
    }

    unlink(path);
    printf("  PASS: test_basic_loading\n");
}

/*
 * Test 2: Words with chars not in seed phrase should be filtered out.
 * "hello" has 'h','e' which are not in the seed phrase mapping.
 */
void test_filtering_invalid_chars(void) {
    const char *path = "/tmp/anabrute_test_dict_filter.txt";
    write_file(path, "hello\nworld\nout\n");

    char_counts seed;
    char_counts_create(seed_phrase_str, &seed);

    char_counts_strings dict[MAX_DICT_SIZE];
    uint32_t dict_length = 0;

    int err = read_dict(path, dict, &dict_length, &seed);
    assert(err == 0);

    /* "hello" and "world" have chars outside the seed phrase alphabet */
    /* Only "out" should survive (o, u, t are all in the seed phrase) */
    assert(dict_length == 1);
    assert(dict[0].strings_len == 1);
    assert(strcmp(dict[0].strings[0], "out") == 0);

    unlink(path);
    printf("  PASS: test_filtering_invalid_chars\n");
}

/*
 * Test 3: Anagram-equivalent words should be grouped into one entry.
 * "opts" and "pots" and "stop" and "tops" are all anagrams (same char_counts).
 */
void test_anagram_grouping(void) {
    const char *path = "/tmp/anabrute_test_dict_group.txt";
    write_file(path, "opts\npots\nstop\ntops\nout\n");

    char_counts seed;
    char_counts_create(seed_phrase_str, &seed);

    char_counts_strings dict[MAX_DICT_SIZE];
    uint32_t dict_length = 0;

    int err = read_dict(path, dict, &dict_length, &seed);
    assert(err == 0);

    /* "opts"/"pots"/"stop"/"tops" should be 1 entry with 4 strings */
    /* "out" should be a separate entry */
    assert(dict_length == 2);

    /* Find the entry with 4 strings (the anagram group) */
    int group_idx = (dict[0].strings_len == 4) ? 0 : 1;
    assert(dict[group_idx].strings_len == 4);
    assert(dict[1 - group_idx].strings_len == 1);

    unlink(path);
    printf("  PASS: test_anagram_grouping\n");
}

/*
 * Test 4: Empty lines and single-char lines should not crash.
 * BUGS.md #8: str[len-2] is OOB when len < 2.
 * Under ASan: crashes before fix due to stack-buffer-underflow.
 */
void test_short_lines_no_crash(void) {
    const char *path = "/tmp/anabrute_test_dict_short.txt";
    /* A bare "\n" has len=1 after fgets. str[len-2] = str[-1] = OOB */
    write_file(path, "\nout\n\n");

    char_counts seed;
    char_counts_create(seed_phrase_str, &seed);

    char_counts_strings dict[MAX_DICT_SIZE];
    uint32_t dict_length = 0;

    int err = read_dict(path, dict, &dict_length, &seed);
    assert(err == 0);
    /* "out" should still be loaded despite empty lines */
    assert(dict_length == 1);

    unlink(path);
    printf("  PASS: test_short_lines_no_crash\n");
}

int main(void) {
    printf("test_dict_parsing:\n");
    test_basic_loading();
    test_filtering_invalid_chars();
    test_anagram_grouping();
    test_short_lines_no_crash();
    printf("All dict parsing tests passed!\n");
    return 0;
}
