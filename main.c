#include <math.h>
#include "common.h"
#include "avx_cruncher.h"
#include "cpu_cruncher.h"
#include "cruncher.h"
#include "opencl_cruncher.h"
#ifdef __APPLE__
#include "metal_cruncher.h"
#endif
#include "dict.h"
#include "fact.h"
#include "hashes.h"
#include "os.h"
#include "permut_types.h"

static const char* size_suffixes[] = {"", "K", "M", "G", "T", "P"};
void format_bignum(uint64_t value, char *dst, uint16_t div) {
    int divs = 0;
    double dval = (double)value;
    while (dval / div >= 1.0) {
        dval /= div;
        divs++;
    }
    if (divs >= 3) // G and above: show 1 decimal
        sprintf(dst, "%.1f%s", dval, size_suffixes[divs]);
    else
        sprintf(dst, "%d%s", (int)dval, size_suffixes[divs]);
}

static int cmp_ccs_length_desc(const void *a, const void *b) {
    const char_counts_strings *ca = *(const char_counts_strings *const *)a;
    const char_counts_strings *cb = *(const char_counts_strings *const *)b;
    return (int)cb->counts.length - (int)ca->counts.length;
}

int main(int argc, char *argv[]) {

    // === read dict

    char_counts_strings* dict_by_char[CHARCOUNT][MAX_DICT_SIZE];
    int dict_by_char_len[CHARCOUNT] = {0};

    char_counts seed_phrase;
    char_counts_create(seed_phrase_str, &seed_phrase);

    char_counts_strings dict[MAX_DICT_SIZE];
    uint32_t dict_length = 0;

    read_dict("input.dict", dict, &dict_length, &seed_phrase);

    for (int i=0; i<dict_length; i++) {
        for (int ci=0; ci<CHARCOUNT; ci++) {
            if (dict[i].counts.counts[ci]) {
                dict_by_char[ci][dict_by_char_len[ci]++] = &dict[i];
                break;
            }
        }
    }
    // Sort each bucket by descending word length for better work-stealing balance
    for (int ci = 0; ci < CHARCOUNT; ci++) {
        if (dict_by_char_len[ci] > 1) {
            qsort(dict_by_char[ci], dict_by_char_len[ci], sizeof(char_counts_strings*), cmp_ccs_length_desc);
        }
    }

    // === precompute per-L0-entry weights for ETA estimation
    // Weight reflects exponential growth of sub-combinations with remaining chars.
    // Actual anagram counts calibrate the scale; weights provide the shape.
    int l0_len = dict_by_char_len[0];
    double *l0_cum_weight = malloc((l0_len + 1) * sizeof(double));
    l0_cum_weight[0] = 0.0;
    for (int i = 0; i < l0_len; i++) {
        int remaining = seed_phrase.length - dict_by_char[0][i]->counts.length;
        l0_cum_weight[i + 1] = l0_cum_weight[i] + pow(3.0, remaining);
    }
    double l0_total_weight = l0_cum_weight[l0_len];

    // === setup shared cpu/gpu cruncher stuff

    tasks_buffers tasks_buffs;
    tasks_buffers_create(&tasks_buffs);

    uint32_t *hashes;
    const uint32_t hashes_num = read_hashes("input.hashes", &hashes);
    ret_iferr(!hashes_num, "failed to read hashes");
    ret_iferr(!hashes, "failed to allocate hashes");

    // === probe and create crunchers ===

    cruncher_ops *all_backends[] = {
#ifdef __APPLE__
        &metal_cruncher_ops,
#endif
        &opencl_cruncher_ops,
        &avx512_cruncher_ops,
        &avx2_cruncher_ops,
        &scalar_cruncher_ops,
        NULL
    };

    // Shared output buffer
    uint32_t *hashes_reversed = calloc(hashes_num, MAX_STR_LENGTH);
    ret_iferr(!hashes_reversed, "failed to allocate hashes_reversed");

    cruncher_config cruncher_cfg = {
        .tasks_buffs = &tasks_buffs,
        .hashes = hashes,
        .hashes_num = hashes_num,
        .hashes_reversed = hashes_reversed,
    };

    #define MAX_CRUNCHER_INSTANCES 64
    typedef struct {
        cruncher_ops *ops;
        void *ctx;
        pthread_t thread;
    } cruncher_instance;

    cruncher_instance crunchers[MAX_CRUNCHER_INSTANCES];
    uint32_t num_crunchers = 0;

    printf("Probing cruncher backends:\n");
    bool have_gpu = false;        // Metal or discrete OpenCL GPU
    bool have_cpu_accel = false;  // AVX-512 or AVX2
    for (int bi = 0; all_backends[bi]; bi++) {
        cruncher_ops *ops = all_backends[bi];

        bool is_gpu = (ops != &avx512_cruncher_ops && ops != &avx2_cruncher_ops && ops != &scalar_cruncher_ops);

        // Skip OpenCL if native GPU (Metal) already active
        if (have_gpu && ops == &opencl_cruncher_ops) {
            printf("  %s: skipped (native GPU already active)\n", ops->name);
            continue;
        }

        // Skip CPU-bound backends if GPU is available (they'd just contend)
        if (have_gpu && !is_gpu) {
            printf("  %s: skipped (GPU backend active)\n", ops->name);
            continue;
        }

        // Skip lower-priority CPU backends if we already have a faster one
        if (have_cpu_accel && (ops == &avx2_cruncher_ops || ops == &scalar_cruncher_ops)) {
            printf("  %s: skipped (faster CPU backend active)\n", ops->name);
            continue;
        }

        uint32_t count = ops->probe();
        if (!count) continue;

        printf("  %s: %d instance(s)\n", ops->name, count);

        if (is_gpu) have_gpu = true;
        if (ops == &avx512_cruncher_ops || ops == &avx2_cruncher_ops) have_cpu_accel = true;

        for (uint32_t i = 0; i < count && num_crunchers < MAX_CRUNCHER_INSTANCES; i++) {
            cruncher_instance *ci = &crunchers[num_crunchers];
            ci->ops = ops;
            ci->ctx = calloc(1, ops->ctx_size);
            int err = ops->create(ci->ctx, &cruncher_cfg, i);
            ret_iferr(err, "failed to create cruncher instance");
            num_crunchers++;
        }
    }
    printf("%d cruncher instance(s) total\n\n", num_crunchers);

    // === create cpu cruncher contexts
    // GPU backends don't compete for CPU — use all cores for enumeration.
    // CPU-bound crunchers (AVX-512, AVX2, scalar) share cores — limit enumerators to 2.
    uint32_t total_cores = num_cpu_cores();
    uint32_t num_cpu_crunchers = have_gpu ? total_cores : 2;
    volatile uint32_t shared_l0_counter = 0;
    volatile uint64_t shared_anas_produced = 0;
    cpu_cruncher_ctx cpu_cruncher_ctxs[num_cpu_crunchers];
    for (uint32_t id=0; id<num_cpu_crunchers; id++) {
        cpu_cruncher_ctx_create(cpu_cruncher_ctxs+id, id, num_cpu_crunchers, &seed_phrase, &dict_by_char, dict_by_char_len, &tasks_buffs, &shared_l0_counter, &shared_anas_produced);
    }

    // === create and start cruncher threads

    printf("searching through anas up to %d words\n", MAX_WORD_LENGTH);

    struct timeval t0, t1;
    gettimeofday(&t0, 0);

    // Start CPU (dict enumeration) threads — priority set inside run_cpu_cruncher_thread
    pthread_t cpu_threads[num_cpu_crunchers];
    for (int i=0; i<num_cpu_crunchers; i++) {
        int err = pthread_create(cpu_threads+i, NULL, run_cpu_cruncher_thread, cpu_cruncher_ctxs+i);
        ret_iferr(err, "failed to create cpu thread");
    }

    // Start cruncher threads
    for (uint32_t i = 0; i < num_crunchers; i++) {
        int err = pthread_create(&crunchers[i].thread, NULL, crunchers[i].ops->run, crunchers[i].ctx);
        ret_iferr(err, "failed to create cruncher thread");
    }

    // === monitor and display progress

    bool hash_is_printed[hashes_num];
    memset(hash_is_printed, 0, sizeof(bool) * hashes_num);
    char strbuf[1024];

    while (1) {
        sleep(1);

        // Check if any cruncher still running
        bool any_running = false;
        for (uint32_t i = 0; i < num_crunchers; i++) {
            any_running |= crunchers[i].ops->is_running(crunchers[i].ctx);
        }

        // Print newly found hashes from shared buffer
        for (int hi = 0; hi < hashes_num; hi++) {
            if (!hash_is_printed[hi] && hashes_reversed[hi * MAX_STR_LENGTH / 4]) {
                hash_to_ascii(hashes + hi * 4, strbuf);
                printf("\033[2K\r%s:  %s\n", strbuf, (char *)(hashes_reversed + hi * MAX_STR_LENGTH / 4));
                hash_is_printed[hi] = true;
            }
        }

        // CPU progress (shared atomic counter)
        uint32_t cpu_progress = shared_l0_counter;
        if (cpu_progress > (uint32_t)dict_by_char_len[0]) cpu_progress = dict_by_char_len[0];

        gettimeofday(&t1, 0);
        long elapsed_secs = t1.tv_sec-t0.tv_sec;

        // Per-backend stats
        int pos = sprintf(strbuf, "%02ld:%02ld:%02ld | %d cpus: %u/%d | %d buffs",
               elapsed_secs/3600, (elapsed_secs/60)%60, elapsed_secs%60,
               num_cpu_crunchers, cpu_progress, dict_by_char_len[0],
               tasks_buffs.ring_count);

        float total_aps = 0;
        uint64_t total_consumed = 0;
        for (uint32_t i = 0; i < num_crunchers; i++) {
            total_consumed += crunchers[i].ops->get_total_anas(crunchers[i].ctx);
        }

        cruncher_ops *seen_ops[MAX_CRUNCHER_INSTANCES];
        uint32_t seen_count = 0;
        for (uint32_t i = 0; i < num_crunchers; i++) {
            bool already = false;
            for (uint32_t j = 0; j < seen_count; j++) {
                if (seen_ops[j] == crunchers[i].ops) { already = true; break; }
            }
            if (already) continue;
            seen_ops[seen_count++] = crunchers[i].ops;

            float backend_busy = 0, backend_aps = 0;
            uint32_t backend_count = 0;
            for (uint32_t k = 0; k < num_crunchers; k++) {
                if (crunchers[k].ops != crunchers[i].ops) continue;
                float busy, aps;
                crunchers[k].ops->get_stats(crunchers[k].ctx, &busy, &aps);
                backend_busy += busy;
                backend_aps += aps;
                backend_count++;
            }
            backend_busy /= backend_count;

            char aps_str[32];
            format_bignum(backend_aps, aps_str, 1000);
            pos += sprintf(strbuf + pos, " | %s(%d): %sAna/s %.0f%%",
                           crunchers[i].ops->name, backend_count, aps_str, backend_busy);
            total_aps += backend_aps;
        }

        // ETA: weighted progress (theoretical shape) calibrated by actual anas_produced (scale)
        uint64_t anas_produced = shared_anas_produced;
        double done_weight = l0_cum_weight[cpu_progress < l0_len ? cpu_progress : l0_len];
        if (done_weight > 0 && total_aps > 0 && elapsed_secs > 2) {
            double scale = (double)anas_produced / done_weight;
            double estimated_total = scale * l0_total_weight;
            double remaining_anas = estimated_total - (double)total_consumed;
            if (remaining_anas < 0) remaining_anas = 0;
            long eta_secs = (long)(remaining_anas / total_aps);
            pos += sprintf(strbuf + pos, " | ETA %02ld:%02ld:%02ld",
                           eta_secs/3600, (eta_secs/60)%60, eta_secs%60);
        }

        printf("\033[2K\r%s", strbuf);
        fflush(stdout);

        // if all cpu threads are done - send poison pill to crunchers
        bool cpus_done = true;
        for (uint32_t i = 0; i < num_cpu_crunchers; i++) {
            if (cpu_cruncher_ctxs[i].progress_l0_index < dict_by_char_len[0]) {
                cpus_done = false;
                break;
            }
        }
        if (cpus_done) {
            tasks_buffers_close(&tasks_buffs);
        }

        if (!any_running) {
            // Final hash scan — crunchers merge results before setting is_running=false,
            // so the shared buffer is up to date by the time we get here
            for (int hi = 0; hi < hashes_num; hi++) {
                if (!hash_is_printed[hi] && hashes_reversed[hi * MAX_STR_LENGTH / 4]) {
                    hash_to_ascii(hashes + hi * 4, strbuf);
                    printf("\033[2K\r%s:  %s\n", strbuf, (char *)(hashes_reversed + hi * MAX_STR_LENGTH / 4));
                    hash_is_printed[hi] = true;
                }
            }
            printf("\033[2K\r\n");
            break;
        }
    }

    // === cleanup
    for (uint32_t i=0; i<num_cpu_crunchers; i++) {
        pthread_join(cpu_threads[i], NULL);
    }
    for (uint32_t i = 0; i < num_crunchers; i++) {
        pthread_join(crunchers[i].thread, NULL);
    }

    // === final stats
    gettimeofday(&t1, 0);
    float wall_secs = (float)(t1.tv_sec - t0.tv_sec) + (float)(t1.tv_usec - t0.tv_usec) / 1000000.0f;

    uint64_t grand_total_anas = 0;
    // Per-backend totals (reuse seen_ops pattern)
    {
        cruncher_ops *seen[MAX_CRUNCHER_INSTANCES];
        uint32_t sc = 0;
        for (uint32_t i = 0; i < num_crunchers; i++) {
            bool already = false;
            for (uint32_t j = 0; j < sc; j++) {
                if (seen[j] == crunchers[i].ops) { already = true; break; }
            }
            if (already) continue;
            seen[sc++] = crunchers[i].ops;

            uint64_t backend_anas = 0;
            uint32_t backend_count = 0;
            for (uint32_t k = 0; k < num_crunchers; k++) {
                if (crunchers[k].ops != crunchers[i].ops) continue;
                backend_anas += crunchers[k].ops->get_total_anas(crunchers[k].ctx);
                backend_count++;
            }
            grand_total_anas += backend_anas;

            char anas_str[32], aps_str[32];
            format_bignum(backend_anas, anas_str, 1024);
            format_bignum((uint64_t)(backend_anas / wall_secs), aps_str, 1000);
            printf("  %s(%d): %s anas, %sAna/s avg\n",
                   crunchers[i].ops->name, backend_count, anas_str, aps_str);
        }
    }

    char total_str[32], total_aps_str[32];
    format_bignum(grand_total_anas, total_str, 1024);
    format_bignum((uint64_t)(grand_total_anas / wall_secs), total_aps_str, 1000);
    printf("  total: %s anas in %.1fs, %sAna/s effective\n", total_str, wall_secs, total_aps_str);

    for (uint32_t i = 0; i < num_crunchers; i++) {
        crunchers[i].ops->destroy(crunchers[i].ctx);
        free(crunchers[i].ctx);
    }
    free(hashes_reversed);
    free(l0_cum_weight);
}
