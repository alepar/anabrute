#include "cpu_cruncher.h"
#include "os.h"

void cpu_cruncher_ctx_create(cpu_cruncher_ctx* cruncher, uint32_t cpu_cruncher_id, uint32_t num_cpu_crunchers,
                             char_counts* seed_phrase, char_counts_strings* (*dict_by_char)[CHARCOUNT][MAX_DICT_SIZE], int* dict_by_char_len,
                             tasks_buffers* tasks_buffs, volatile uint32_t *shared_l0_counter)
{
    cruncher->num_cpu_crunchers = num_cpu_crunchers;
    cruncher->cpu_cruncher_id = cpu_cruncher_id;
    cruncher->shared_l0_counter = shared_l0_counter;

    cruncher->seed_phrase = seed_phrase;
    cruncher->dict_by_char = dict_by_char;
    cruncher->dict_by_char_len = dict_by_char_len;

    cruncher->progress_l0_index = 0;
    for (int i = 0; i <= MAX_WORD_LENGTH; i++) {
        cruncher->local_buffers[i] = NULL;
    }
    cruncher->local_free_count = 0;
    cruncher->tasks_buffs = tasks_buffs;
}

static tasks_buffer* cpu_obtain_buffer(cpu_cruncher_ctx *ctx) {
    // Check local cache first (no lock)
    if (ctx->local_free_count > 0) {
        tasks_buffer *buf = ctx->local_free[--ctx->local_free_count];
        tasks_buffer_reset(buf);
        return buf;
    }

    // Bulk grab from global free-list
    pthread_mutex_lock(&ctx->tasks_buffs->mutex);
    while (ctx->local_free_count < LOCAL_FREE_CAP && ctx->tasks_buffs->num_free > 0) {
        ctx->local_free[ctx->local_free_count++] = ctx->tasks_buffs->free_arr[--ctx->tasks_buffs->num_free];
    }
    pthread_mutex_unlock(&ctx->tasks_buffs->mutex);

    if (ctx->local_free_count > 0) {
        tasks_buffer *buf = ctx->local_free[--ctx->local_free_count];
        tasks_buffer_reset(buf);
        return buf;
    }

    // Nothing available — allocate fresh
    return tasks_buffer_allocate();
}

int submit_tasks(cpu_cruncher_ctx* ctx, int8_t permut[], int permut_len, char *all_strs) {
    permut[permut_len] = 0;

    // Count permutable words to determine N for buffer routing
    int n = 0;
    for (int i = 0; permut[i]; i++) {
        if (permut[i] > 0) n++;
    }
    if (n > MAX_WORD_LENGTH) n = MAX_WORD_LENGTH;  // safety clamp

    tasks_buffer **bufp = &ctx->local_buffers[n];

    int errcode = 0;
    if (*bufp != NULL && tasks_buffer_isfull(*bufp)) {
        errcode = tasks_buffers_add_buffer(ctx->tasks_buffs, *bufp);
        ret_iferr(errcode, "cpu cruncher failed to pass buffer to gpu crunchers");
        *bufp = NULL;
    }

    if (*bufp == NULL) {
        *bufp = cpu_obtain_buffer(ctx);
        ret_iferr(!*bufp, "cpu cruncher failed to allocate local buffer");
    }

    tasks_buffer_add_task(*bufp, all_strs, permut);

    return 0;
}

int recurse_combs(cpu_cruncher_ctx* ctx, char *all_strs, string_idx_and_count sics[], int sics_len, int sics_idx, int8_t permut[], int permut_len, int start_idx) {
    int errcode=0;

    if (sics_idx >= sics_len) {
        int si, di=0;
        for (si = 0; si < sics_len; si++) {
            if (sics[si].count) {
                for (;permut[di];di++);
                permut[di] = sics[si].offset+1;
            }
        }

        if(errcode = submit_tasks(ctx, permut, permut_len, all_strs)) {
            return errcode;
        }

        for (di=0; di<permut_len; di++) {
            if (permut[di] > 0) {
                permut[di] = 0;
            }
        }
    } else if (start_idx > permut_len && sics[sics_idx].count > 0) {
        // failsafe
    } else if (sics[sics_idx].count > 1 || start_idx > 0) {
        for (int i=start_idx; i<permut_len; i++) {
            if (permut[i] == 0) {
                permut[i] = -sics[sics_idx].offset-1;
                sics[sics_idx].count--;

                if (sics[sics_idx].count == 0) {
                    errcode = recurse_combs(ctx, all_strs, sics, sics_len, sics_idx+1, permut, permut_len, 0);
                } else if (sics[sics_idx].count <= permut_len-i-1) {
                    errcode = recurse_combs(ctx, all_strs, sics, sics_len, sics_idx, permut, permut_len, i+1);
                }

                if (errcode) return errcode;

                sics[sics_idx].count++;
                permut[i] = 0;
            }
        }
    } else {
        return recurse_combs(ctx, all_strs, sics, sics_len, sics_idx+1, permut, permut_len, 0);
    }

    return errcode;
}

int recurse_string_combs(cpu_cruncher_ctx* ctx, stack_item *stack, int stack_len, int stack_idx, int string_idx, string_and_count *scs, int scs_idx) {
    int errcode=0;
    if (stack_idx >= stack_len) {
        string_idx_and_count sics[scs_idx];

        uint8_t strs_count = 0;
        for (int i=0; i<scs_idx; i++) {
            if (scs[i].count) {
                strs_count += strlen(scs[i].str)+1;
            }
        }

        uint8_t word_count = 0;
        char all_strs[MAX_STR_LENGTH];
        memset(all_strs, 0, MAX_STR_LENGTH);
        int8_t all_offs=0;
        int sics_len=0;
        for (int i=0; i<scs_idx; i++) {
            if (scs[i].count) {
                word_count += scs[i].count;
                sics[sics_len].count = scs[i].count;
                sics[sics_len].offset = all_offs;
                sics_len++;
                int slen = strlen(scs[i].str) + 1;  /* include null terminator */
                memcpy(all_strs + all_offs, scs[i].str, slen);
                all_offs += slen;
            }
        }

        int8_t permut[MAX_OFFSETS_LENGTH];
        memset(permut, 0, MAX_OFFSETS_LENGTH);
        return recurse_combs(ctx, all_strs, sics, sics_len, 0, permut, word_count, 0);
    } else if (stack[stack_idx].ccs->strings_len > string_idx+1) {
        const uint8_t orig_count = stack[stack_idx].count;
        for (uint8_t i=0; i <= orig_count; i++) {
            stack[stack_idx].count = orig_count-i;

            scs[scs_idx].str = stack[stack_idx].ccs->strings[string_idx];
            scs[scs_idx].count = i;
            errcode=recurse_string_combs(ctx, stack, stack_len, stack_idx, string_idx + 1, scs, scs_idx + 1);
            if (errcode) return errcode;
        }
        stack[stack_idx].count = orig_count;
    } else {
        scs[scs_idx].str = stack[stack_idx].ccs->strings[string_idx];
        scs[scs_idx].count = stack[stack_idx].count;
        return recurse_string_combs(ctx, stack, stack_len, stack_idx + 1, 0, scs, scs_idx + 1);
    }

    return errcode;
}

int recurse_dict_words(cpu_cruncher_ctx* ctx, char_counts *remainder, int curchar, int curdictidx, stack_item *stack, int stack_len, string_and_count *scs) {
/*    if (debug_flag) {
        printf("\t%d\t%d\t%d\t%d\t||\t", remainder->length, curchar, curdictidx, stack_len);
        for (int i=0; i<stack_len; i++) {
            printf("%s", stack[i].ccs->strings[0]);
            if (stack[i].count > 1) {
                printf("*%d ", stack[i].count);
            } else {
                printf(" ");
            }
        }
        printf("\n");
    }*/

    int word_count=0;
    for (int i=0; i<stack_len; i++) {
        word_count+=stack[i].count;
    }
    if(word_count > MAX_WORD_LENGTH) {
        return 0;
    }

    if (remainder->length == 0) {
        return recurse_string_combs(ctx, stack, stack_len, 0, 0, scs, 0);
    }

    for (;curchar<CHARCOUNT && !remainder->counts[curchar]; curchar++)
        if(curchar >= CHARCOUNT) {
            return 0;
        }

    int errcode=0;

    if (stack_len == 0) {
        // Atomic work stealing: each thread grabs the next available index
        uint32_t i;
        while ((i = __sync_fetch_and_add(ctx->shared_l0_counter, 1)) < (uint32_t)ctx->dict_by_char_len[curchar]) {
            ctx->progress_l0_index = i;

            stack[0].ccs = (*ctx->dict_by_char)[curchar][i];

            char_counts next_remainder;
            char_counts_copy(remainder, &next_remainder);
            for (uint8_t ccs_count=1; char_counts_subtract(&next_remainder, &(*ctx->dict_by_char)[curchar][i]->counts); ccs_count++) {
                stack[0].count = ccs_count;

                int next_char = curchar;
                int next_idx = i+1;

                if(next_remainder.counts[next_char] == 0) {
                    next_char++;
                    next_idx = 0;
                }

                errcode = recurse_dict_words(ctx, &next_remainder, next_char, next_idx, stack, 1, scs);
                if (errcode) return errcode;
            }
        }
    } else {
        for (int i=curdictidx; i<ctx->dict_by_char_len[curchar]; i++) {
            stack[stack_len].ccs = (*ctx->dict_by_char)[curchar][i];

            char_counts next_remainder;
            char_counts_copy(remainder, &next_remainder);
            for (uint8_t ccs_count=1; char_counts_subtract(&next_remainder, &(*ctx->dict_by_char)[curchar][i]->counts); ccs_count++) {
                stack[stack_len].count = ccs_count;

                int next_char = curchar;
                int next_idx = i+1;

                if(next_remainder.counts[next_char] == 0) {
                    next_char++;
                    next_idx = 0;
                }

                errcode = recurse_dict_words(ctx, &next_remainder, next_char, next_idx, stack, stack_len + 1, scs);
                if (errcode) return errcode;
            }
        }
    }

    return errcode;
}

void* run_cpu_cruncher_thread(void *ptr) {
    cpu_cruncher_ctx *ctx = ptr;
    set_thread_high_priority();

    char_counts local_remainder;
    char_counts_copy(ctx->seed_phrase, &local_remainder);

    stack_item stack[20];
    string_and_count scs[120];

    int errcode = recurse_dict_words(ctx, &local_remainder, 0, 0, stack, 0, scs);

    // Flush all per-N buffers that have remaining tasks
    for (int n = 0; n <= MAX_WORD_LENGTH; n++) {
        if (ctx->local_buffers[n] != NULL && ctx->local_buffers[n]->num_tasks > 0) {
            errcode = tasks_buffers_add_buffer(ctx->tasks_buffs, ctx->local_buffers[n]);
            ret_iferr(errcode, "cpu cruncher failed to pass last buffer to gpu crunchers");
            ctx->local_buffers[n] = NULL;
        }
    }

    // Free local free-list buffers
    for (int i = 0; i < ctx->local_free_count; i++) {
        tasks_buffer_free(ctx->local_free[i]);
    }
    ctx->local_free_count = 0;

    ctx->progress_l0_index = ctx->dict_by_char_len[0]; // mark this cpu cruncher as done

    if (errcode) fprintf(stderr, "[cpucruncher %d] errcode %d\n", ctx->cpu_cruncher_id, errcode);
    return NULL;
}