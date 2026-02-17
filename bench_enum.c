#include <stdio.h>
#include <string.h>
#include <pthread.h>
#include "cpu_cruncher.h"
#include "dict.h"
#include "seedphrase.h"
#include "os.h"

/*
 * Benchmark for CPU dictionary enumeration.
 * Measures single-thread and multi-thread task production throughput.
 *
 * Usage: bench_enum [max_threads]
 *   max_threads: test 1, 2, 4, ... up to max_threads (default: num_cpu_cores)
 */

/* Consumer thread: drains produced buffers, counts tasks/anas */
typedef struct {
    tasks_buffers *buffs;
    uint64_t total_tasks;
    uint64_t total_anas;
} consumer_ctx;

static void *consumer_thread(void *arg) {
    consumer_ctx *cctx = arg;
    tasks_buffer *buf;
    while (1) {
        tasks_buffers_get_buffer(cctx->buffs, &buf);
        if (buf == NULL) break;
        cctx->total_tasks += buf->num_tasks;
        cctx->total_anas += buf->num_anas;
        tasks_buffers_recycle(cctx->buffs, buf);
    }
    return NULL;
}

/* Shared dict data passed to run_benchmark */
static char_counts_strings *dict_by_char[CHARCOUNT][MAX_DICT_SIZE];
static int dict_by_char_len[CHARCOUNT];

static int cmp_ccs_length_desc(const void *a, const void *b) {
    const char_counts_strings *ca = *(const char_counts_strings *const *)a;
    const char_counts_strings *cb = *(const char_counts_strings *const *)b;
    return (int)cb->counts.length - (int)ca->counts.length;
}

static double run_benchmark(int num_threads, char_counts *seed, double baseline_secs) {
    tasks_buffers buffs;
    tasks_buffers_create(&buffs);

    /* Create CPU cruncher contexts */
    volatile uint32_t shared_l0_counter = 0;
    cpu_cruncher_ctx ctxs[num_threads];
    for (int i = 0; i < num_threads; i++) {
        cpu_cruncher_ctx_create(&ctxs[i], i, num_threads, seed, &dict_by_char, dict_by_char_len, &buffs, &shared_l0_counter);
    }

    /* Start consumer */
    consumer_ctx cctx = { .buffs = &buffs, .total_tasks = 0, .total_anas = 0 };
    pthread_t consumer;
    pthread_create(&consumer, NULL, consumer_thread, &cctx);

    /* Start producer threads */
    uint64_t t0 = current_micros();

    pthread_t threads[num_threads];
    for (int i = 0; i < num_threads; i++) {
        pthread_create(&threads[i], NULL, run_cpu_cruncher_thread, &ctxs[i]);
    }

    /* Wait for producers to finish */
    for (int i = 0; i < num_threads; i++) {
        pthread_join(threads[i], NULL);
    }

    /* Signal consumer to stop and wait */
    tasks_buffers_close(&buffs);
    pthread_join(consumer, NULL);

    uint64_t t1 = current_micros();
    double secs = (double)(t1 - t0) / 1e6;
    double tasks_per_sec = cctx.total_tasks / secs;
    double anas_per_sec = cctx.total_anas / secs;

    printf("  %2d thread(s): %8lu tasks, %10lu anas in %.3fs",
           num_threads, (unsigned long)cctx.total_tasks, (unsigned long)cctx.total_anas, secs);
    printf("  | %.0f tasks/s, %.1fM anas/s", tasks_per_sec, anas_per_sec / 1e6);

    if (baseline_secs > 0) {
        double speedup = baseline_secs / secs;
        double efficiency = speedup / num_threads * 100.0;
        printf("  | %.2fx (%.0f%% eff)", speedup, efficiency);
    }
    printf("\n");

    tasks_buffers_free(&buffs);
    return secs;
}

int main(int argc, char *argv[]) {
    int max_threads = num_cpu_cores();
    if (argc > 1) max_threads = atoi(argv[1]);
    if (max_threads < 1) max_threads = 1;

    /* Load dictionary */
    char_counts seed;
    char_counts_create(seed_phrase_str, &seed);

    char_counts_strings dict[MAX_DICT_SIZE];
    uint32_t dict_length = 0;
    int err = read_dict("input.dict", dict, &dict_length, &seed);
    if (err) {
        fprintf(stderr, "Failed to read input.dict (run from project root)\n");
        return 1;
    }

    /* Organize dict_by_char (same as main.c) */
    memset(dict_by_char_len, 0, sizeof(dict_by_char_len));
    for (uint32_t i = 0; i < dict_length; i++) {
        for (int ci = 0; ci < CHARCOUNT; ci++) {
            if (dict[i].counts.counts[ci]) {
                dict_by_char[ci][dict_by_char_len[ci]++] = &dict[i];
                break;
            }
        }
    }

    /* Sort each bucket by descending word length for work-stealing balance */
    for (int ci = 0; ci < CHARCOUNT; ci++) {
        if (dict_by_char_len[ci] > 1) {
            qsort(dict_by_char[ci], dict_by_char_len[ci], sizeof(char_counts_strings*), cmp_ccs_length_desc);
        }
    }

    printf("CPU Enumeration Benchmark\n");
    printf("  Dictionary: %u entries\n", dict_length);
    printf("  Max words: %d\n", MAX_WORD_LENGTH);
    printf("  Max threads: %d\n\n", max_threads);

    /* Single-thread baseline */
    double baseline = run_benchmark(1, &seed, 0);

    /* Multi-thread runs: 2, 4, 8, ... */
    for (int n = 2; n <= max_threads; n *= 2) {
        run_benchmark(n, &seed, baseline);
    }

    /* Also run at max_threads if not a power of 2 and not already tested */
    if (max_threads > 1 && (max_threads & (max_threads - 1)) != 0) {
        run_benchmark(max_threads, &seed, baseline);
    }

    /* Cleanup */
    for (uint32_t i = 0; i < dict_length; i++) {
        char_counts_strings_free(&dict[i]);
    }

    return 0;
}
