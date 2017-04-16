#ifndef ANABRUTE_CRUNCHER_TYPES_H
#define ANABRUTE_CRUNCHER_TYPES_H

#include <stdint.h>

#include "permut_types.h"

typedef struct task_buffer_s {

} task_buffer;

typedef struct task_buffers_s {

} task_buffers;

typedef struct cpu_cruncher_ctx_s {

    // parallelization over devices
    uint32_t num_cpu_crunchers;
    uint32_t cpu_cruncher_id;

    // job definition
    char_counts* seed_phrase;
    char_counts_strings* (*dict_by_char)[CHARCOUNT][MAX_DICT_SIZE];
    int* dict_by_char_len;

    // progress stats
    volatile int progress_l0_index;

    task_buffer* local_buffer;

    // output
    task_buffers* task_buffs;

} cpu_cruncher_ctx;

void cpu_cruncher_ctx_create(cpu_cruncher_ctx* cruncher, uint32_t cpu_cruncher_id, uint32_t num_cpu_crunchers,
                             char_counts* seed_phrase, char_counts_strings* (*dict_by_char)[CHARCOUNT][MAX_DICT_SIZE], int* dict_by_char_len,
                             task_buffers* task_buffs);

#endif //ANABRUTE_CRUNCHER_TYPES_H
