#include <errno.h>
#include <pthread.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#ifdef __APPLE__
    #include <unitypes.h>
    #include <event.h>
#else
    #include <sys/time.h>
#endif

#include "anatypes.h"
#include "fact.h"
#include "hashes.h"
#include "ocl_layer.h"

stack_item stack[20];
string_and_count scs[120];
char_counts_strings* dict_by_char[CHARCOUNT][MAX_DICT_SIZE];
int dict_by_char_len[CHARCOUNT] = {0};

int submit_tasks(anactx* anactx, int8_t permut[], int permut_len, char *all_strs) {
    int permutable_count = 0;
    for (int i=0; i<permut_len; i++) {
        if (permut[i] > 0) {
            permutable_count++;
        }
    }

    if (permutable_count > 11) { // TODO skip lengthes > 11 for now
        return 0;
    }

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

/*
    permut_task task;
    cl_int errcode;

    // TODO superfluos memory copy
    permut[permut_len] = 0;
    memcpy(&task.all_strs, all_strs, MAX_STR_LENGTH);
    memcpy(&task.offsets, permut, MAX_OFFSETS_LENGTH);

    uint64_t permut_iters = fact(permutable_count);
    for (uint64_t batchi=0; batchi < (permut_iters/MAX_ITERS_PER_TASK+1); batchi++) {
        task.start_from = batchi*MAX_ITERS_PER_TASK+1;
        errcode = anactx_submit_permut_task(anactx, &task);
        ret_iferr(errcode, "failed to submit task");
    }
*/

    return 0;
}

int recurse_combs(anactx* anactx, char *all_strs, string_idx_and_count sics[], int sics_len, int sics_idx, int8_t permut[], int permut_len, int start_idx) {
    int errcode=0;

    if (sics_idx >= sics_len) {
        int si, di=0;
        for (si = 0; si < sics_len; si++) {
            if (sics[si].count) {
                for (;permut[di];di++);
                permut[di] = sics[si].offset+1;
            }
        }

        if(errcode = submit_tasks(anactx, permut, permut_len, all_strs)) {
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
                    errcode = recurse_combs(anactx, all_strs, sics, sics_len, sics_idx+1, permut, permut_len, 0);
                } else if (sics[sics_idx].count <= permut_len-i-1) {
                    errcode = recurse_combs(anactx, all_strs, sics, sics_len, sics_idx, permut, permut_len, i+1);
                }

                if (errcode) return errcode;

                sics[sics_idx].count++;
                permut[i] = 0;
            }
        }
    } else {
        return recurse_combs(anactx, all_strs, sics, sics_len, sics_idx+1, permut, permut_len, 0);
    }

    return errcode;
}

int recurse_string_combs(anactx* anactx, stack_item *stack, int stack_len, int stack_idx, int string_idx, string_and_count *scs, int scs_idx) {
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
        return recurse_combs(anactx, all_strs, sics, sics_len, 0, permut, word_count, 0);
    } else if (stack[stack_idx].ccs->strings_len > string_idx+1) {
        const uint8_t orig_count = stack[stack_idx].count;
        for (uint8_t i=0; i <= orig_count; i++) {
            stack[stack_idx].count = orig_count-i;

            scs[scs_idx].str = stack[stack_idx].ccs->strings[string_idx];
            scs[scs_idx].count = i;
            errcode=recurse_string_combs(anactx, stack, stack_len, stack_idx, string_idx + 1, scs, scs_idx + 1);
            if (errcode) return errcode;
        }
        stack[stack_idx].count = orig_count;
    } else {
        scs[scs_idx].str = stack[stack_idx].ccs->strings[string_idx];
        scs[scs_idx].count = stack[stack_idx].count;
        return recurse_string_combs(anactx, stack, stack_len, stack_idx + 1, 0, scs, scs_idx + 1);
    }

    return errcode;
}

int recurse_dict_words(anactx* anactx, char_counts *remainder, int curchar, int curdictidx, int stack_len) {
/*
    // TODO debug
    if (stack_len == 1) {
        if (strcmp(stack[0].ccs->strings[0], "outstations")) {
            if (debug_flag) {
                printf("flushing\n");
                anactx_flush_tasks_buffer(anactx);
                printf("waiting\n");
                anactx_wait_for_cur_kernel(anactx);
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
    if(word_count > 7) {
        return 0;
    }

    if (remainder->length == 0) {
        return recurse_string_combs(anactx, stack, stack_len, 0, 0, scs, 0);
    }

    for (;curchar<CHARCOUNT & !remainder->counts[curchar]; curchar++)
    if(curchar >= CHARCOUNT) {
        return 0;
    }

    int step = 1;
    // TODO parallelization
/*
    if (stack_len == 0) {
        step = anactx->num_threads;
    }
*/

    int errcode=0;
    for (int i=curdictidx; i<dict_by_char_len[curchar]; i+=step) {
        if (stack_len == 0) {
            printf("L0 %d/%d: %s\n", i, dict_by_char_len[curchar], dict_by_char[curchar][i]->strings[0]);
        }

        stack[stack_len].ccs = dict_by_char[curchar][i];

        char_counts next_remainder;
        char_counts_copy(remainder, &next_remainder);
        for (uint8_t ccs_count=1; char_counts_subtract(&next_remainder, &dict_by_char[curchar][i]->counts); ccs_count++) {
            stack[stack_len].count = ccs_count;

            int next_char = curchar;
            int next_idx = i+1;

            if(next_remainder.counts[next_char] == 0) {
                next_char++;
                next_idx = 0;
            }

            errcode = recurse_dict_words(anactx, &next_remainder, next_char, next_idx, stack_len + 1);
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

void* run_brute_thread(void *ptr) {
    anactx *anactx = ptr;

    char_counts local_remainer;
    char_counts_copy(anactx->seed_phrase, &local_remainer);
    int errcode = recurse_dict_words(anactx, &local_remainer, 0, anactx->thread_id, 0);

    if (errcode) fprintf(stderr, "[Thread %d] errcode %d\n", anactx->thread_id, errcode);
    return NULL;
}

int main(int argc, char *argv[]) {

    // ----------- read dict

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

    // todo resort dict_by_char?
    //   by length or char occurs?

    // ----------- setup opencl

    cl_platform_id platform_id;
    cl_uint num_platforms;
    clGetPlatformIDs (1, &platform_id, &num_platforms);
    ret_iferr(!num_platforms, "no platforms");

    cl_uint num_devices;
    clGetDeviceIDs (platform_id, CL_DEVICE_TYPE_ALL, 0, NULL, &num_devices);
    cl_device_id device_ids[num_devices];
    clGetDeviceIDs (platform_id, CL_DEVICE_TYPE_ALL, num_devices, device_ids, &num_devices);
    ret_iferr(!num_devices, "no devices");

    uint32_t num_gpus = 0;
    for (int i=0; i<num_devices; i++) {
        cl_device_type dev_type;
        clGetDeviceInfo (device_ids[i], CL_DEVICE_TYPE, sizeof(dev_type), &dev_type, NULL);
        if (dev_type > CL_DEVICE_TYPE_CPU) {
            num_gpus++;
        }
    }

    if (num_gpus) {
        int d=0;
        for (int s=0; s<num_devices; s++) {
            cl_device_type dev_type;
            clGetDeviceInfo (device_ids[s], CL_DEVICE_TYPE, sizeof(dev_type), &dev_type, NULL);
            if (dev_type > CL_DEVICE_TYPE_CPU) {
                device_ids[d++] = device_ids[s];
            }
        }
        num_devices = num_gpus;
    }

    char char_buf[1024];
    for (int i=0; i<num_devices; i++) {
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
    anactx anactxs[num_devices];
    for (uint32_t i=0; i<num_devices; i++) {
        errcode = anactx_create(&anactxs[i], platform_id, device_ids[i]);
        ret_iferr(errcode, "failed to create anactx");

        anactxs[i].num_threads = num_devices;
        anactxs[i].thread_id = i;
        anactxs[i].seed_phrase = &seed_phrase;

        errcode = anactx_set_input_hashes(&anactxs[i], hashes, hashes_num);
        ret_iferr(errcode, "failed to set input hashes");
    }

    // ----------- run

    run_brute_thread(&anactxs[0]);
//    run_brute_thread(&anactxs[1]);

//    printf("starting %d threads\n", num_devices);
//    pthread_t threads[num_devices];
//    for (uint32_t i=0; i<num_devices; i++) {
//        int err = pthread_create(&threads[i], NULL, run_brute_thread, &anactxs[i]);
//        ret_iferr(err, "failed to create thread");
//    }

    // TODO print out stats
    // TODO join with threads

//    for (uint32_t i=0; i<num_devices; i++) {
//        int err = pthread_join(threads[i], NULL);
//        ret_iferr(err, "failed to join thread");
//    }

    // TODO free anactx
    printf("done\n");
}