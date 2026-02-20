#include <stdio.h>
#include <string.h>
#include <pthread.h>
#include "avx_cruncher.h"
#include "cruncher.h"
#include "hashes.h"
#include "task_buffers.h"
#include "fact.h"
#include "os.h"

/*
 * Benchmark: measures AVX/scalar cruncher throughput.
 * Creates N buffers of tasks with varying n values (2-5 words),
 * runs them through the cruncher with configurable thread count.
 */

static void fill_buffer_with_tasks(tasks_buffer *buf, int n_words) {
    buf->num_tasks = 0;
    buf->num_anas = 0;

    /* Create tasks that look like real anagram candidates */
    const char *sample_words[] = {"tyranous", "pluto", "twits", "put", "lot"};

    for (uint32_t t = 0; t < PERMUT_TASKS_IN_KERNEL_TASK && t < 256*1024; t++) {
        permut_task *task = &buf->permut_tasks[t];
        memset(task, 0, sizeof(permut_task));

        uint8_t off = 0;
        int words_to_use = n_words > 5 ? 5 : n_words;
        for (int i = 0; i < words_to_use; i++) {
            task->a[i] = off + 1;
            int len = strlen(sample_words[i % 5]);
            memcpy(task->all_strs + off, sample_words[i % 5], len + 1);
            off += len + 1;
        }

        for (int i = 0; i < words_to_use; i++) {
            task->offsets[i] = i + 1;
        }
        task->offsets[words_to_use] = 0;
        task->n = words_to_use;
        task->i = 0;
        task->iters_done = 0;
        memset(task->c, 0, MAX_OFFSETS_LENGTH);

        buf->num_tasks++;
        buf->num_anas += fact(words_to_use);
    }
}

int main(int argc, char **argv) {
    int num_threads = 7;
    int num_buffers = 4;
    int n_words = 4;

    if (argc > 1) num_threads = atoi(argv[1]);
    if (argc > 2) num_buffers = atoi(argv[2]);
    if (argc > 3) n_words = atoi(argv[3]);

    printf("AVX Cruncher Benchmark\n");
    printf("  Threads: %d\n", num_threads);
    printf("  Buffers: %d (each %d tasks)\n", num_buffers, PERMUT_TASKS_IN_KERNEL_TASK);
    printf("  Words per task (n): %d → %lu permutations/task\n", n_words, (unsigned long)fact(n_words));

    /* Dummy target hashes (won't match anything) — use 19 to match production */
    #define NUM_TARGET_HASHES 19
    uint32_t hashes[4 * NUM_TARGET_HASHES];
    for (int i = 0; i < 4 * NUM_TARGET_HASHES; i++) hashes[i] = 0xdeadbe00 + i;
    uint32_t hashes_reversed[NUM_TARGET_HASHES * MAX_STR_LENGTH / 4];
    memset(hashes_reversed, 0, sizeof(hashes_reversed));

    tasks_buffers tasks_buffs;
    tasks_buffers_create(&tasks_buffs);

    cruncher_config cfg = {
        .tasks_buffs = &tasks_buffs,
        .hashes = hashes,
        .hashes_num = NUM_TARGET_HASHES,
        .hashes_reversed = hashes_reversed,
    };

    /* Create cruncher threads */
    void **ctxs = calloc(num_threads, sizeof(void *));
    pthread_t *threads = calloc(num_threads, sizeof(pthread_t));

    cruncher_ops *ops = &avx2_cruncher_ops;

    for (int i = 0; i < num_threads; i++) {
        ctxs[i] = calloc(1, ops->ctx_size);
        ops->create(ctxs[i], &cfg, i);
    }

    /* Pre-fill buffers */
    uint64_t total_anas = 0;
    for (int i = 0; i < num_buffers; i++) {
        tasks_buffer *buf = tasks_buffer_allocate();
        fill_buffer_with_tasks(buf, n_words);
        total_anas += buf->num_anas;
        tasks_buffers_add_buffer(&tasks_buffs, buf);
    }
    tasks_buffers_close(&tasks_buffs);

    printf("  Total anagrams to hash: %lu\n\n", (unsigned long)total_anas);

    /* Launch threads */
    uint64_t start = current_micros();

    for (int i = 0; i < num_threads; i++) {
        pthread_create(&threads[i], NULL, ops->run, ctxs[i]);
    }
    for (int i = 0; i < num_threads; i++) {
        pthread_join(threads[i], NULL);
    }

    uint64_t end = current_micros();
    double elapsed_sec = (double)(end - start) / 1000000.0;

    /* Collect stats */
    uint64_t total_consumed = 0;
    for (int i = 0; i < num_threads; i++) {
        total_consumed += ops->get_total_anas(ctxs[i]);
    }

    printf("Results:\n");
    printf("  Elapsed: %.3f sec\n", elapsed_sec);
    printf("  Hashes computed: %lu\n", (unsigned long)total_consumed);
    printf("  Throughput: %.2f M hashes/sec\n", (double)total_consumed / elapsed_sec / 1e6);
    printf("  Per thread: %.2f M hashes/sec\n", (double)total_consumed / elapsed_sec / 1e6 / num_threads);

    /* Cleanup */
    for (int i = 0; i < num_threads; i++) {
        ops->destroy(ctxs[i]);
        free(ctxs[i]);
    }
    free(ctxs);
    free(threads);
    tasks_buffers_free(&tasks_buffs);

    return 0;
}
