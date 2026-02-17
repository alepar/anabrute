#include <assert.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include "cpu_cruncher.h"
#include "dict.h"
#include "seedphrase.h"

static int cmp_ccs_length_desc(const void *a, const void *b) {
    const char_counts_strings *ca = *(const char_counts_strings *const *)a;
    const char_counts_strings *cb = *(const char_counts_strings *const *)b;
    return (int)cb->counts.length - (int)ca->counts.length;
}

static void write_file(const char *path, const char *content) {
    FILE *f = fopen(path, "w");
    assert(f && "failed to create test file");
    fputs(content, f);
    fclose(f);
}

/*
 * Helper: loads dict, organizes into dict_by_char, runs single-threaded
 * CPU cruncher, collects all produced tasks.
 * Returns total number of tasks. Caller must free(*out_tasks) if non-NULL.
 */
static uint32_t run_cruncher_with_dict(const char *dict_path, permut_task **out_tasks) {
    char_counts seed;
    char_counts_create(seed_phrase_str, &seed);

    char_counts_strings dict[MAX_DICT_SIZE];
    uint32_t dict_length = 0;
    int err = read_dict(dict_path, dict, &dict_length, &seed);
    assert(err == 0);

    /* Organize dict_by_char (same as main.c) */
    char_counts_strings* dict_by_char[CHARCOUNT][MAX_DICT_SIZE];
    int dict_by_char_len[CHARCOUNT] = {0};
    for (uint32_t i = 0; i < dict_length; i++) {
        for (int ci = 0; ci < CHARCOUNT; ci++) {
            if (dict[i].counts.counts[ci]) {
                dict_by_char[ci][dict_by_char_len[ci]++] = &dict[i];
                break;
            }
        }
    }

    /* Sort by descending length (same as main.c) */
    for (int ci = 0; ci < CHARCOUNT; ci++) {
        if (dict_by_char_len[ci] > 1) {
            qsort(dict_by_char[ci], dict_by_char_len[ci], sizeof(char_counts_strings*), cmp_ccs_length_desc);
        }
    }

    /* Create tasks_buffers */
    tasks_buffers tasks_buffs;
    tasks_buffers_create(&tasks_buffs);

    /* Run CPU cruncher single-threaded */
    volatile uint32_t shared_l0_counter = 0;
    cpu_cruncher_ctx ctx;
    cpu_cruncher_ctx_create(&ctx, 0, 1, &seed, &dict_by_char, dict_by_char_len, &tasks_buffs, &shared_l0_counter);
    run_cpu_cruncher_thread(&ctx);

    /* Close buffers so get_buffer returns NULL when empty */
    tasks_buffers_close(&tasks_buffs);

    /* Collect all tasks */
    uint32_t total_tasks = 0;
    *out_tasks = NULL;
    uint32_t tasks_capacity = 0;

    tasks_buffer *buf;
    while (1) {
        tasks_buffers_get_buffer(&tasks_buffs, &buf);
        if (buf == NULL) break;

        if (total_tasks + buf->num_tasks > tasks_capacity) {
            tasks_capacity = total_tasks + buf->num_tasks + 64;
            *out_tasks = realloc(*out_tasks, tasks_capacity * sizeof(permut_task));
        }
        memcpy(*out_tasks + total_tasks, buf->permut_tasks,
               buf->num_tasks * sizeof(permut_task));
        total_tasks += buf->num_tasks;
        tasks_buffers_recycle(&tasks_buffs, buf);
    }

    tasks_buffers_free(&tasks_buffs);

    for (uint32_t i = 0; i < dict_length; i++) {
        char_counts_strings_free(&dict[i]);
    }

    return total_tasks;
}

/*
 * Test 1: Single word that IS the seed phrase.
 * "tyranousplutotwits" is an exact anagram → 1 task with n=1.
 */
void test_single_word_anagram(void) {
    const char *path = "/tmp/anabrute_test_cpu_single.txt";
    write_file(path, "tyranousplutotwits\n");

    permut_task *tasks = NULL;
    uint32_t count = run_cruncher_with_dict(path, &tasks);

    assert(count == 1);
    assert(tasks[0].n == 1);
    /* Verify the word is stored in all_strs */
    assert(strcmp(tasks[0].all_strs, "tyranousplutotwits") == 0);

    free(tasks);
    unlink(path);
    printf("  PASS: test_single_word_anagram\n");
}

/*
 * Test 2: Two words that together form the full anagram.
 * "tyranous" + "plutotwits" → 1 task with n=2.
 */
void test_two_word_anagram(void) {
    const char *path = "/tmp/anabrute_test_cpu_two.txt";
    write_file(path, "tyranous\nplutotwits\n");

    permut_task *tasks = NULL;
    uint32_t count = run_cruncher_with_dict(path, &tasks);

    assert(count == 1);
    assert(tasks[0].n == 2);

    free(tasks);
    unlink(path);
    printf("  PASS: test_two_word_anagram\n");
}

/*
 * Test 3: Three words that together form the full anagram.
 * "tyranous" + "pluto" + "twits" → task(s) with n=3.
 */
void test_three_word_anagram(void) {
    const char *path = "/tmp/anabrute_test_cpu_three.txt";
    write_file(path, "tyranous\npluto\ntwits\n");

    permut_task *tasks = NULL;
    uint32_t count = run_cruncher_with_dict(path, &tasks);

    assert(count > 0 && "should find at least one 3-word anagram");

    for (uint32_t i = 0; i < count; i++) {
        assert(tasks[i].n == 3);
    }

    free(tasks);
    unlink(path);
    printf("  PASS: test_three_word_anagram\n");
}

/*
 * Test 4: Words that cannot combine to form the full anagram.
 * "out" + "run" don't cover all seed phrase characters → 0 tasks.
 */
void test_no_valid_anagrams(void) {
    const char *path = "/tmp/anabrute_test_cpu_none.txt";
    write_file(path, "out\nrun\n");

    permut_task *tasks = NULL;
    uint32_t count = run_cruncher_with_dict(path, &tasks);

    assert(count == 0);

    free(tasks);
    unlink(path);
    printf("  PASS: test_no_valid_anagrams\n");
}

int main(void) {
    printf("test_cpu_enumeration:\n");
    test_single_word_anagram();
    test_two_word_anagram();
    test_three_word_anagram();
    test_no_valid_anagrams();
    printf("All CPU enumeration tests passed!\n");
    return 0;
}
