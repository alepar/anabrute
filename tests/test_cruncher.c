#include <assert.h>
#include <stdio.h>
#include <string.h>
#include "avx_cruncher.h"
#include "cruncher.h"
#include "opencl_cruncher.h"
#ifdef __APPLE__
#include "metal_cruncher.h"
#endif
#include "hashes.h"
#include "task_buffers.h"
#include "fact.h"
#include "os.h"

/*
 * Helper: construct a tasks_buffer with a single manually-built task.
 * words[] is an array of num_words strings. All positions are permutable.
 */
static tasks_buffer *make_task_buffer(const char *words[], int num_words) {
    tasks_buffer *buf = tasks_buffer_allocate();
    permut_task *task = buf->permut_tasks;
    memset(task, 0, sizeof(permut_task));

    /* Pack words into all_strs and set up a[] with 1-based byte offsets */
    uint8_t off = 0;
    for (int i = 0; i < num_words; i++) {
        task->a[i] = off + 1;  /* 1-based offset into all_strs */
        int len = strlen(words[i]);
        memcpy(task->all_strs + off, words[i], len + 1);  /* include null terminator */
        off += len + 1;
    }

    /* All positions are permutable: positive values are 1-based indices into a[] */
    for (int i = 0; i < num_words; i++) {
        task->offsets[i] = i + 1;
    }
    task->offsets[num_words] = 0;  /* terminator */

    task->n = num_words;
    task->i = 0;
    task->iters_done = 0;
    memset(task->c, 0, MAX_OFFSETS_LENGTH);

    buf->num_tasks = 1;
    buf->num_anas = fact(num_words);

    return buf;
}

/*
 * Helper: run a cruncher backend on given tasks, check hashes_reversed for matches.
 * The caller is responsible for allocating hashes_reversed (hashes_num * MAX_STR_LENGTH bytes).
 */
static void run_cruncher_on_tasks(cruncher_ops *ops, tasks_buffer *buf,
                                   uint32_t *hashes, uint32_t hashes_num,
                                   uint32_t *hashes_reversed) {
    tasks_buffers tasks_buffs;
    tasks_buffers_create(&tasks_buffs);

    // cfg must outlive ops->run() because the backend stores a pointer to it
    cruncher_config cfg = {
        .tasks_buffs = &tasks_buffs,
        .hashes = hashes,
        .hashes_num = hashes_num,
        .hashes_reversed = hashes_reversed,
    };

    void *ctx = calloc(1, ops->ctx_size);
    assert(ctx && "failed to allocate cruncher context");
    int err = ops->create(ctx, &cfg, 0);
    assert(err == 0 && "failed to create cruncher");

    /* Add the buffer and close the queue before running, so the cruncher
     * thread will consume the buffer and then exit when it finds the queue
     * closed with nothing remaining. */
    tasks_buffers_add_buffer(&tasks_buffs, buf);
    tasks_buffers_close(&tasks_buffs);

    ops->run(ctx);

    ops->destroy(ctx);
    free(ctx);
    tasks_buffers_free(&tasks_buffs);
}

/*
 * Test 1: single word "tyranousplutotwits" with n=1.
 * MD5("tyranousplutotwits") = 896304cdb1add2652c6445f245cfd3b2
 */
static void test_single_word_match(cruncher_ops *ops) {
    const char *hash_hex = "896304cdb1add2652c6445f245cfd3b2";
    uint32_t hashes[4];
    ascii_to_hash(hash_hex, hashes);

    uint32_t hashes_reversed[MAX_STR_LENGTH / 4];
    memset(hashes_reversed, 0, MAX_STR_LENGTH);

    const char *words[] = {"tyranousplutotwits"};
    tasks_buffer *buf = make_task_buffer(words, 1);

    run_cruncher_on_tasks(ops, buf, hashes, 1, hashes_reversed);

    assert(hashes_reversed[0] != 0 && "should find single-word match");
    printf("    PASS: single word match\n");
}

/*
 * Test 2: two words "tyranous" + "plutotwits" with n=2.
 * MD5("tyranous plutotwits") = 04b386be280077bbb71bf72ebc17b92d
 * MD5("plutotwits tyranous") = 8c4232547ac7fdf9e3f130784147815a
 */
static void test_two_word_match(cruncher_ops *ops) {
    const char *hash_hexes[] = {
        "04b386be280077bbb71bf72ebc17b92d",
        "8c4232547ac7fdf9e3f130784147815a",
    };
    uint32_t hashes[8];
    ascii_to_hash(hash_hexes[0], hashes);
    ascii_to_hash(hash_hexes[1], hashes + 4);

    uint32_t hashes_reversed[2 * MAX_STR_LENGTH / 4];
    memset(hashes_reversed, 0, 2 * MAX_STR_LENGTH);

    const char *words[] = {"tyranous", "plutotwits"};
    tasks_buffer *buf = make_task_buffer(words, 2);

    run_cruncher_on_tasks(ops, buf, hashes, 2, hashes_reversed);

    assert(hashes_reversed[0] != 0 && "should find first ordering");
    assert(hashes_reversed[MAX_STR_LENGTH / 4] != 0 && "should find second ordering");
    printf("    PASS: two word match\n");
}

/*
 * Test 3: no matching hash. Should produce zero matches.
 */
static void test_no_match(cruncher_ops *ops) {
    const char *hash_hex = "00000000000000000000000000000000";
    uint32_t hashes[4];
    ascii_to_hash(hash_hex, hashes);

    uint32_t hashes_reversed[MAX_STR_LENGTH / 4];
    memset(hashes_reversed, 0, MAX_STR_LENGTH);

    const char *words[] = {"tyranousplutotwits"};
    tasks_buffer *buf = make_task_buffer(words, 1);

    run_cruncher_on_tasks(ops, buf, hashes, 1, hashes_reversed);

    assert(hashes_reversed[0] == 0 && "should NOT find match for zero hash");
    printf("    PASS: no false matches\n");
}

/*
 * Test 4: multiple hashes, only one should match.
 * Hash 0: MD5("tyranousplutotwits") = 896304cdb1add2652c6445f245cfd3b2  -> should match
 * Hash 1: all zeros -> should NOT match
 */
static void test_multiple_hashes_selective(cruncher_ops *ops) {
    const char *hash_hexes[] = {
        "896304cdb1add2652c6445f245cfd3b2",  /* MD5("tyranousplutotwits") */
        "00000000000000000000000000000000",   /* should NOT match */
    };
    uint32_t hashes[8];
    ascii_to_hash(hash_hexes[0], hashes);
    ascii_to_hash(hash_hexes[1], hashes + 4);

    uint32_t hashes_reversed[2 * MAX_STR_LENGTH / 4];
    memset(hashes_reversed, 0, 2 * MAX_STR_LENGTH);

    const char *words[] = {"tyranousplutotwits"};
    tasks_buffer *buf = make_task_buffer(words, 1);

    run_cruncher_on_tasks(ops, buf, hashes, 2, hashes_reversed);

    assert(hashes_reversed[0] != 0 && "should find match for real hash");
    assert(hashes_reversed[MAX_STR_LENGTH / 4] == 0 && "should NOT match zero hash");
    printf("    PASS: multiple hashes, only correct matches\n");
}

static void run_backend_tests(cruncher_ops *ops) {
    printf("  Testing %s backend:\n", ops->name);
    test_single_word_match(ops);
    test_two_word_match(ops);
    test_no_match(ops);
    test_multiple_hashes_selective(ops);
}

int main(void) {
    printf("test_cruncher:\n");

    cruncher_ops *backends[] = {
#ifdef __APPLE__
        &metal_cruncher_ops,
#endif
        &opencl_cruncher_ops,
        &avx_cruncher_ops,
        NULL
    };

    int tested = 0;
    for (int i = 0; backends[i]; i++) {
        uint32_t count = backends[i]->probe();
        if (count == 0) {
            printf("  %s: not available, skipping\n", backends[i]->name);
            continue;
        }
        run_backend_tests(backends[i]);
        tested++;
    }

    if (tested == 0) {
        printf("  WARNING: no backends available, all tests skipped\n");
    }

    printf("All cruncher tests passed!\n");
    return 0;
}
