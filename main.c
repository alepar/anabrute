#include "common.h"
#include "avx_cruncher.h"
#include "cpu_cruncher.h"
#include "cruncher.h"
#include "opencl_cruncher.h"
#include "dict.h"
#include "fact.h"
#include "hashes.h"
#include "os.h"
#include "permut_types.h"

static const char* size_suffixes[] = {"", "K", "M", "G", "T", "P"};
void format_bignum(uint64_t value, char *dst, uint16_t div) {
    int divs = 0;
    while (value/div > 1) {
        value = value/div;
        divs++;
    }
    sprintf(dst, "%lu%s", value, size_suffixes[divs]);
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
    // maybe resort dict_by_char? by length or char occurs?

    // === setup shared cpu/gpu cruncher stuff

    tasks_buffers tasks_buffs;
    tasks_buffers_create(&tasks_buffs);

    uint32_t *hashes;
    const uint32_t hashes_num = read_hashes("input.hashes", &hashes);
    ret_iferr(!hashes_num, "failed to read hashes");
    ret_iferr(!hashes, "failed to allocate hashes");

    // === probe and create crunchers ===

    cruncher_ops *all_backends[] = {
        &opencl_cruncher_ops,
        &avx_cruncher_ops,
        // Future: &metal_cruncher_ops,
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
    for (int bi = 0; all_backends[bi]; bi++) {
        cruncher_ops *ops = all_backends[bi];
        uint32_t count = ops->probe();
        if (!count) continue;

        printf("  %s: %d instance(s)\n", ops->name, count);

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

    uint32_t num_cpu_crunchers = num_cpu_cores();
    cpu_cruncher_ctx cpu_cruncher_ctxs[num_cpu_crunchers];
    for (uint32_t id=0; id<num_cpu_crunchers; id++) {
        cpu_cruncher_ctx_create(cpu_cruncher_ctxs+id, id, num_cpu_crunchers, &seed_phrase, &dict_by_char, dict_by_char_len, &tasks_buffs);
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
                printf("%s:  %s\n", strbuf, (char *)(hashes_reversed + hi * MAX_STR_LENGTH / 4));
                hash_is_printed[hi] = true;
            }
        }

        // CPU progress
        int min=dict_by_char_len[0], max=0;
        for (int i=0; i<num_cpu_crunchers; i++) {
            const int progress = cpu_cruncher_ctxs[i].progress_l0_index;
            if (progress > max) max = progress;
            if (progress < min) min = progress;
        }

        // Aggregate stats from all crunchers
        float overall_busy = 0, overall_anas_per_sec = 0;
        for (uint32_t i = 0; i < num_crunchers; i++) {
            float busy, aps;
            crunchers[i].ops->get_stats(crunchers[i].ctx, &busy, &aps);
            overall_busy += busy;
            overall_anas_per_sec += aps;
        }
        if (num_crunchers > 0) overall_busy /= num_crunchers;
        format_bignum(overall_anas_per_sec, strbuf, 1000);

        gettimeofday(&t1, 0);
        long elapsed_secs = t1.tv_sec-t0.tv_sec;

        printf("%02ld:%02ld:%02ld | %d cpus: %d-%d/%d | %d buffs | %d crunchers, %sAna/s, %.0f%% effic\r",
               elapsed_secs/3600, (elapsed_secs/60)%60, elapsed_secs%60,
               num_cpu_crunchers, min, max, dict_by_char_len[0],
               tasks_buffs.num_ready,
               num_crunchers, strbuf, overall_busy);
        fflush(stdout);

        // if cpu's are done - send poison pill to crunchers
        if (min >= dict_by_char_len[0] && max >= dict_by_char_len[0]) {
            tasks_buffers_close(&tasks_buffs);
        }

        if (!any_running) {
            // Final hash scan — crunchers merge results before setting is_running=false,
            // so the shared buffer is up to date by the time we get here
            for (int hi = 0; hi < hashes_num; hi++) {
                if (!hash_is_printed[hi] && hashes_reversed[hi * MAX_STR_LENGTH / 4]) {
                    hash_to_ascii(hashes + hi * 4, strbuf);
                    printf("%s:  %s\n", strbuf, (char *)(hashes_reversed + hi * MAX_STR_LENGTH / 4));
                    hash_is_printed[hi] = true;
                }
            }
            printf("\n");
            break;
        }
    }

    // === cleanup
    for (uint32_t i=0; i<num_cpu_crunchers; i++) {
        pthread_join(cpu_threads[i], NULL);
    }
    for (uint32_t i = 0; i < num_crunchers; i++) {
        pthread_join(crunchers[i].thread, NULL);
        crunchers[i].ops->destroy(crunchers[i].ctx);
        free(crunchers[i].ctx);
    }
    free(hashes_reversed);
}
