#include "common.h"
#include "cpu_cruncher.h"
#include "fact.h"
#include "gpu_cruncher.h"
#include "hashes.h"
#include "os.h"
#include "permut_types.h"

int submit_tasks(cpu_cruncher_ctx* ctx, int8_t permut[], int permut_len, char *all_strs) {
    int permutable_count = 0;
    for (int i=0; i<permut_len; i++) {
        if (permut[i] > 0) {
            permutable_count++;
        }
    }

    if (permutable_count > 11) { // TODO skip lengthes > 11 for now
        return 0;
    }

/*
    if (ctx->cpu_cruncher_id == 0) {
        for (int j=0; j<permut_len; j++) {
            char offset = permut[j];
            if (offset < 0) {
                offset = -offset;
            } else {
                printf("*");
            }
            offset--;
            printf("%s ", all_strs+offset);
        }
        printf("\n");
    }
*/

    permut[permut_len] = 0;

    int errcode = 0;
    if (ctx->local_buffer != NULL && tasks_buffer_isfull(ctx->local_buffer)) {
        errcode = tasks_buffers_add_buffer(ctx->tasks_buffs, ctx->local_buffer);
        ret_iferr(errcode, "cpu cruncher failed to pass buffer to gpu crunchers");
        ctx->local_buffer = NULL;
    }

    if (ctx->local_buffer == NULL) {
        ctx->local_buffer = tasks_buffer_allocate();
        ret_iferr(!ctx->local_buffer, "cpu cruncher failed to allocate local buffer");
    }

    tasks_buffer_add_task(ctx->local_buffer, all_strs, permut);

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
        char all_strs[strs_count];
        int8_t all_offs=0;
        int sics_len=0;
        for (int i=0; i<scs_idx; i++) {
            if (scs[i].count) {
                word_count += scs[i].count;
                sics[sics_len].count = scs[i].count;
                sics[sics_len].offset = all_offs;
                sics_len++;
                for (int j=0; j<=strlen(scs[i].str); j++) {
                    all_strs[all_offs++] = scs[i].str[j];
                }
            }
        }

        int8_t permut[word_count];
        memset(permut, 0, word_count);
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
/*
    // TODO debug
    if (stack_len == 1) {
        if (strcmp(stack[0].ccs->strings[0], "outstations")) {
            if (debug_flag) {
                printf("flushing\n");
                gpu_cruncher_ctx_flush_tasks_buffer(gpu_cruncher_ctx);
                printf("waiting\n");
                gpu_cruncher_ctx_wait_for_cur_kernel(gpu_cruncher_ctx);
            }
            debug_flag=0;
        } else {
            debug_flag=1;
        }
    }
*/

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

    // TODO skip
    int word_count=0;
    for (int i=0; i<stack_len; i++) {
        word_count+=stack[i].count;
    }
    if(word_count > 9) {
        return 0;
    }

    if (remainder->length == 0) {
        return recurse_string_combs(ctx, stack, stack_len, 0, 0, scs, 0);
    }

    for (;curchar<CHARCOUNT & !remainder->counts[curchar]; curchar++)
    if(curchar >= CHARCOUNT) {
        return 0;
    }

    int step = 1;
    if (stack_len == 0) {
        step = ctx->num_cpu_crunchers;
    }

    int errcode=0;
    for (int i=curdictidx; i<ctx->dict_by_char_len[curchar]; i+=step) {
        if (stack_len == 0) {
            ctx->progress_l0_index = i;
        }

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

    return errcode;
}

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
void format_bignum(uint64_t size, char *dst, uint16_t div) {
    int divs = 0;
    while (size/div > 1) {
        size = size/div;
        divs++;
    }
    sprintf(dst, "%lu%s", size, size_suffixes[divs]);
}

const uint32_t read_hashes(char *file_name, uint32_t **hashes) {
    FILE *const fd = fopen(file_name, "r");
    if (!fd) {
        return 0;
    }

    fseek(fd, 0L, SEEK_END);
    const uint32_t file_size = (const uint32_t) ftell(fd);
    rewind(fd);

    const uint32_t hashes_num_est = (file_size + 1) / 33;
    uint32_t hashes_num = 0;

    *hashes = malloc(hashes_num_est*16);

    char buf[128];
    while(fgets(buf, sizeof(buf), fd) != NULL) {
        for (int i=0; i<sizeof(buf); i++) {
            if (buf[i] == '\n' || buf[i] == '\r') {
                buf[i] = 0;
            }
        }
        if (strlen(buf) != 32) {
            fprintf(stderr, "not a hash! (%s)\n", buf);
        }

        if (hashes_num>hashes_num_est) {
            fprintf(stderr, "too many hashes? skipping tail...\n");
            break;
        }

        ascii_to_hash(buf, &((*hashes)[hashes_num*4]));
        hashes_num++;
    }

    return hashes_num;
}

void* run_cpu_cruncher_thread(void *ptr) {
    cpu_cruncher_ctx *ctx = ptr;

    char_counts local_remainder;
    char_counts_copy(ctx->seed_phrase, &local_remainder);

    stack_item stack[20];
    string_and_count scs[120];

    int errcode = recurse_dict_words(ctx, &local_remainder, 0, ctx->cpu_cruncher_id, stack, 0, scs);

    if (ctx->local_buffer != NULL && ctx->local_buffer->num_tasks > 0) {
        errcode = tasks_buffers_add_buffer(ctx->tasks_buffs, ctx->local_buffer);
        ctx->progress_l0_index = ctx->dict_by_char_len[0]; // mark this cpu cruncher as done
        ret_iferr(errcode, "cpu cruncher failed to pass last buffer to gpu crunchers");
        ctx->local_buffer = NULL;
    }

    ctx->progress_l0_index = ctx->dict_by_char_len[0]; // mark this cpu cruncher as done

    if (errcode) fprintf(stderr, "[cpucruncher %d] errcode %d\n", ctx->cpu_cruncher_id, errcode);
    return NULL;
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
        errcode = gpu_cruncher_ctx_create(gpu_cruncher_ctxs+i, platform_id, device_ids[i], &tasks_buffs);
        ret_iferr(errcode, "failed to create gpu_cruncher_ctx");
        errcode = gpu_cruncher_ctx_set_input_hashes(gpu_cruncher_ctxs+i, hashes, hashes_num);
        ret_iferr(errcode, "failed to set input hashes");
    }

    // === create cpu cruncher contexts

    uint32_t num_cpu_crunchers = num_cpu_cores();
    cpu_cruncher_ctx cpu_cruncher_ctxs[num_cpu_crunchers];
    for (uint32_t id=0; id<num_cpu_crunchers; id++) {
        cpu_cruncher_ctx_create(cpu_cruncher_ctxs+id, id, num_cpu_crunchers, &seed_phrase, &dict_by_char, dict_by_char_len, &tasks_buffs);
    }

    // === create and start cruncher threads

    struct timeval t0;
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

    while (1) {
        sleep(1);

        int min=dict_by_char_len[0], max=0;
        for (int i=0; i<num_cpu_crunchers; i++) {
            const int progress = cpu_cruncher_ctxs[i].progress_l0_index;
            if (progress > max) max = progress;
            if (progress < min) min = progress;
        }

        uint32_t buffs_gpus_consumed = 0;
        for (int i=0; i<num_gpu_crunchers; i++) {
            buffs_gpus_consumed += gpu_cruncher_ctxs[i].consumed_bufs;
        }
        printf("\r%d cpus: %d-%d/%d | %d buffs | %d gpus | %d buffs done\r", num_cpu_crunchers, min, max, dict_by_char_len[0], tasks_buffs.num_ready, num_gpu_crunchers, buffs_gpus_consumed);
        fflush(stdout);

        if (min >= dict_by_char_len[0] && max >= dict_by_char_len[0]) {
            tasks_buffers_close(&tasks_buffs);
        }

        bool gpu_is_running = false;
        for (int i=0; i<num_gpu_crunchers; i++) {
            gpu_is_running |= gpu_cruncher_ctxs[i].is_running;
        }
        if (!gpu_is_running) {
            printf("\n");
            break;
        }
    }
    
    struct timeval t1;
    gettimeofday(&t1, 0);

    for (uint32_t i=0; i<num_cpu_crunchers; i++) {
        int err = pthread_join(cpu_threads[i], NULL);
        ret_iferr(err, "failed to join cpu thread");
    }
    for (uint32_t i=0; i<num_gpu_crunchers; i++) {
        int err = pthread_join(gpu_threads[i], NULL);
        ret_iferr(err, "failed to join gpu thread");
    }

    // TODO handle exit codes

    // TODO free gpu_cruncher_ctx

    long elapsed_millis = (t1.tv_sec-t0.tv_sec)*1000 + (t1.tv_usec-t0.tv_usec)/1000;
    printf("done in %lusec\n", elapsed_millis/1000);
}