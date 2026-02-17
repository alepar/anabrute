#ifndef ANABRUTE_CRUNCHER_TYPES_H
#define ANABRUTE_CRUNCHER_TYPES_H

#include "permut_types.h"
#include "task_buffers.h"

typedef struct cpu_cruncher_ctx_s {

    // parallelization over devices
    uint32_t num_cpu_crunchers;
    uint32_t cpu_cruncher_id;

    // atomic work stealing: shared counter for top-level dict entries
    volatile uint32_t *shared_l0_counter;

    // job definition
    char_counts* seed_phrase;
    char_counts_strings* (*dict_by_char)[CHARCOUNT][MAX_DICT_SIZE];
    int* dict_by_char_len;

    // progress stats
    volatile int progress_l0_index;

    tasks_buffer* local_buffers[MAX_WORD_LENGTH+1];  // per-N buffers for uniform SIMD group dispatch

    // per-thread buffer free-list (avoids global mutex for obtain)
    #define LOCAL_FREE_CAP 4
    tasks_buffer* local_free[LOCAL_FREE_CAP];
    int local_free_count;

    // output
    tasks_buffers* tasks_buffs;

} cpu_cruncher_ctx;

void cpu_cruncher_ctx_create(cpu_cruncher_ctx* cruncher, uint32_t cpu_cruncher_id, uint32_t num_cpu_crunchers,
                             char_counts* seed_phrase, char_counts_strings* (*dict_by_char)[CHARCOUNT][MAX_DICT_SIZE], int* dict_by_char_len,
                             tasks_buffers* tasks_buffs, volatile uint32_t *shared_l0_counter);

void* run_cpu_cruncher_thread(void *ptr);

#endif //ANABRUTE_CRUNCHER_TYPES_H
