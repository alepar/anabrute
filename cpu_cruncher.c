#include "cpu_cruncher.h"

void cpu_cruncher_ctx_create(cpu_cruncher_ctx* cruncher, uint32_t cpu_cruncher_id, uint32_t num_cpu_crunchers,
                             char_counts* seed_phrase, char_counts_strings* (*dict_by_char)[CHARCOUNT][MAX_DICT_SIZE], int* dict_by_char_len,
                             tasks_buffers* tasks_buffs)
{
    cruncher->num_cpu_crunchers = num_cpu_crunchers;
    cruncher->cpu_cruncher_id = cpu_cruncher_id;

    cruncher->seed_phrase = seed_phrase;
    cruncher->dict_by_char = dict_by_char;
    cruncher->dict_by_char_len = dict_by_char_len;

    cruncher->progress_l0_index = 0;
    cruncher->local_buffer = NULL;
    cruncher->tasks_buffs = tasks_buffs;
}
