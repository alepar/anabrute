#include "common.h"
#include "cpu_cruncher.h"
#include "fact.h"
#include "gpu_cruncher.h"
#include "hashes.h"
#include "os.h"
#include "permut_types.h"

int read_dict(char_counts_strings *dict, uint32_t *dict_length, char_counts *seed_phrase) {
    FILE *dictFile = fopen("input.dict", "r");
    if (!dictFile) {
        fprintf(stderr, "dict file not found!\n");
        return -1;
    }

    char buf1[100] = {0}, buf2[100] = {0};
    char *buflines[] = {buf1, buf2};
    int lineidx = 0;

    while(fgets(buflines[lineidx], 100, dictFile) != NULL) {
        char *const str = buflines[lineidx];
        const size_t len = strlen(str);
        if (str[len-1] == '\n' || str[len-1] == '\r') {
            str[len-1] = 0;
        }
        if (str[len-2] == '\n' || str[len-2] == '\r') {
            str[len-2] = 0;
        }

        if (strcmp(buflines[0], buflines[1])) {
            lineidx = 1-lineidx;
            if (char_counts_strings_create(str, &dict[*dict_length])) {
                continue;
            }

            if (char_counts_contains(seed_phrase, &dict[*dict_length].counts)) {
                int i;

                for (i=0; i<*dict_length; i++) {
                    if (char_counts_equal(&dict[i].counts, &dict[*dict_length].counts)) {
                        break;
                    }
                }
                if (i==*dict_length) {
                    (*dict_length)++;
                    if (*dict_length > MAX_DICT_SIZE) {
                        fprintf(stderr, "dict overflow! %d\n", *dict_length);
                        return -2;
                    }
                }

                if (char_counts_strings_addstring(&dict[i], str)) {
                    fprintf(stderr, "strings overflow! %d", dict[i].strings_len);
                    return -3;
                }
            }
        }
    }
    fclose(dictFile);
    return 0;
}

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

    read_dict(dict, &dict_length, &seed_phrase);

    for (int i=0; i<dict_length; i++) {
        for (int ci=0; ci<CHARCOUNT; ci++) {
            if (dict[i].counts.counts[ci]) {
                dict_by_char[ci][dict_by_char_len[ci]++] = &dict[i];
                break;
            }
        }
    }
    // maybe resort dict_by_char? by length or char occurs?

    // setup shared cpu/gpu cruncher studf

    tasks_buffers tasks_buffs;
    tasks_buffers_create(&tasks_buffs);

    // === setup opencl / gpu cruncher contexts

    cl_platform_id platform_id;
    cl_uint num_platforms;
    clGetPlatformIDs (1, &platform_id, &num_platforms);
    ret_iferr(!num_platforms, "no platforms");

    cl_uint num_gpu_crunchers;
    clGetDeviceIDs (platform_id, CL_DEVICE_TYPE_ALL, 0, NULL, &num_gpu_crunchers);
    cl_device_id device_ids[num_gpu_crunchers];
    clGetDeviceIDs (platform_id, CL_DEVICE_TYPE_ALL, num_gpu_crunchers, device_ids, &num_gpu_crunchers);
    ret_iferr(!num_gpu_crunchers, "no devices");

    uint32_t num_gpus = 0;
    for (int i=0; i<num_gpu_crunchers; i++) {
        cl_device_type dev_type;
        clGetDeviceInfo (device_ids[i], CL_DEVICE_TYPE, sizeof(dev_type), &dev_type, NULL);
        if (dev_type > CL_DEVICE_TYPE_CPU) {
            num_gpus++;
        }
    }

    if (num_gpus) {
        int d=0;
        for (int s=0; s<num_gpu_crunchers; s++) {
            cl_device_type dev_type;
            clGetDeviceInfo (device_ids[s], CL_DEVICE_TYPE, sizeof(dev_type), &dev_type, NULL);
            if (dev_type > CL_DEVICE_TYPE_CPU) {
                device_ids[d++] = device_ids[s];
            }
        }
        num_gpu_crunchers = num_gpus;
    }

    char char_buf[1024];
    for (int i=0; i<num_gpu_crunchers; i++) {
        cl_ulong local_mem; char local_mem_str[32];
        cl_ulong global_mem; char global_mem_str[32];

        clGetDeviceInfo(device_ids[i], CL_DEVICE_GLOBAL_MEM_SIZE, 8, &global_mem, NULL);
        clGetDeviceInfo(device_ids[i], CL_DEVICE_LOCAL_MEM_SIZE, 8, &local_mem, NULL);
        clGetDeviceInfo (device_ids[i], CL_DEVICE_NAME, 1024, char_buf, NULL);

        format_bignum(global_mem, global_mem_str, 1024);
        format_bignum(local_mem, local_mem_str, 1024);
        printf("OpenCL device #%d: %s (g:%siB l:%siB)\n", i+1, char_buf, global_mem_str, local_mem_str);
    }
    printf("\n");

    uint32_t *hashes;
    const uint32_t hashes_num = read_hashes("input.hashes", &hashes);
    ret_iferr(!hashes_num, "failed to read hashes");
    ret_iferr(!hashes, "failed to allocate hashes");

    cl_int errcode;
    gpu_cruncher_ctx gpu_cruncher_ctxs[num_gpu_crunchers];
    for (uint32_t i=0; i<num_gpu_crunchers; i++) {
        errcode = gpu_cruncher_ctx_create(gpu_cruncher_ctxs+i, platform_id, device_ids[i], &tasks_buffs, hashes, hashes_num);
        ret_iferr(errcode, "failed to create gpu_cruncher_ctx");
    }

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

    pthread_t cpu_threads[num_cpu_crunchers];
    for (int i=0; i<num_cpu_crunchers; i++) {
        int err = pthread_create(cpu_threads+i, NULL, run_cpu_cruncher_thread, cpu_cruncher_ctxs+i);
        ret_iferr(err, "failed to create cpu thread");
    }

    pthread_t gpu_threads[num_gpu_crunchers];
    for (int i=0; i<num_gpu_crunchers; i++) {
        int err = pthread_create(gpu_threads+i, NULL, run_gpu_cruncher_thread, gpu_cruncher_ctxs+i);
        ret_iferr(err, "failed to create gpu thread");
    }

    // === monitor and display progress

    bool hash_is_reversed[hashes_num]; memset(hash_is_reversed, 0, sizeof(bool)*hashes_num);
    char strbuf[1024], strbuf2[1024];
    while (1) {
        sleep(1);

        bool gpu_is_running = false;
        for (int i=0; i<num_gpu_crunchers; i++) {
            gpu_is_running |= gpu_cruncher_ctxs[i].is_running;
        }

        if (!gpu_is_running) {
            // force last hashes refresh
            for (int gi = 0; gi < num_gpu_crunchers; gi++) {
                gpu_cruncher_ctx_read_hashes_reversed(gpu_cruncher_ctxs+gi);
            }
        }

        // print out new hashes as we go
        for (int hi=0; hi<hashes_num; hi++) {
            if (!hash_is_reversed[hi]) {
                for (int gi = 0; gi < num_gpu_crunchers; gi++) {
                    if (gpu_cruncher_ctxs[gi].hashes_reversed[hi*MAX_STR_LENGTH/4]) {
                        hash_to_ascii(hashes+hi*4, strbuf);
                        printf("%s:  %s\n", strbuf, (char*)(gpu_cruncher_ctxs[gi].hashes_reversed) + hi*MAX_STR_LENGTH);
                        hash_is_reversed[hi] = true;
                    }
                }
            }
        }

        int min=dict_by_char_len[0], max=0;
        for (int i=0; i<num_cpu_crunchers; i++) {
            const int progress = cpu_cruncher_ctxs[i].progress_l0_index;
            if (progress > max) max = progress;
            if (progress < min) min = progress;
        }

        float busy_percentage, overall_busy_percentage=0;
        float anas_per_sec, overall_anas_per_sec=0;
        uint32_t buffs_gpus_consumed = 0;
        uint64_t overall_anas_consumed = 0;
        for (int i=0; i<num_gpu_crunchers; i++) {
            buffs_gpus_consumed += gpu_cruncher_ctxs[i].consumed_bufs;
            gpu_cruncher_get_stats(gpu_cruncher_ctxs+i, &busy_percentage, &anas_per_sec);
            overall_busy_percentage += busy_percentage;
            overall_anas_per_sec += anas_per_sec;
            overall_anas_consumed += gpu_cruncher_ctxs[i].consumed_anas;
        }
        overall_busy_percentage /= num_gpu_crunchers;
        format_bignum(overall_anas_per_sec, strbuf, 1000);

        gettimeofday(&t1, 0);
        long elapsed_secs = t1.tv_sec-t0.tv_sec;

        format_bignum(overall_anas_consumed, strbuf2, 1000);

        printf("%02dh:%02dm:%02ds | %d cpus: %d-%d/%d | %d buffs | %d gpus, %sAna/s, %.lf%% effic | %d buffs, %s anas done\r",
               elapsed_secs/3600, (elapsed_secs/60)%60, elapsed_secs%60,
               num_cpu_crunchers, min, max, dict_by_char_len[0],
               tasks_buffs.num_ready,
               num_gpu_crunchers, strbuf, overall_busy_percentage,
               buffs_gpus_consumed, strbuf2);
        fflush(stdout);

        // if cpu's are done - send poison pill to gpus
        if (min >= dict_by_char_len[0] && max >= dict_by_char_len[0]) {
            tasks_buffers_close(&tasks_buffs);
        }

        if (!gpu_is_running) {
            printf("\n");
            break;
        }
    }

    for (uint32_t i=0; i<num_cpu_crunchers; i++) {
        int err = pthread_join(cpu_threads[i], NULL);
        ret_iferr(err, "failed to join cpu thread");
    }
    for (uint32_t i=0; i<num_gpu_crunchers; i++) {
        int err = pthread_join(gpu_threads[i], NULL);
        ret_iferr(err, "failed to join gpu thread");
    }
    for (uint32_t i=0; i<num_gpu_crunchers; i++) {
        cl_int err = gpu_cruncher_ctx_free(gpu_cruncher_ctxs+i);
        ret_iferr(err, "failed to free gpu cruncher ctx");
    }
}