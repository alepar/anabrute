#ifndef ANABRUTE_TASK_BUFFERS_H
#define ANABRUTE_TASK_BUFFERS_H

#include "common.h"

typedef struct permut_task_s {
    char all_strs[MAX_STR_LENGTH];
    int8_t offsets[MAX_OFFSETS_LENGTH];  // positives - permutable all_strs+(a[offset-1]-1), negatives - fixed all_strs+(-offset-1), zeroes - end terminators;
    uint8_t a[MAX_OFFSETS_LENGTH];
    uint8_t c[MAX_OFFSETS_LENGTH];
    uint16_t i;     // 32_t to align structure to longs
    uint16_t n;
    uint32_t iters_done;
} permut_task;

typedef struct tasks_buffer_s {
    permut_task *permut_tasks;
    uint32_t num_tasks;
    uint64_t num_anas;
} tasks_buffer;

tasks_buffer* tasks_buffer_allocate();
void tasks_buffer_free(tasks_buffer* buf);
void tasks_buffer_reset(tasks_buffer* buf);
bool tasks_buffer_isfull(tasks_buffer* buf);
void tasks_buffer_add_task(tasks_buffer* buf, char* all_strs, int8_t* offsets);

typedef struct tasks_buffers_s {
    // Ring buffer for ready tasks (O(1) insert/remove)
    tasks_buffer* ring[TASKS_BUFFERS_SIZE];
    uint32_t ring_head;    // producer writes at ring[head % SIZE]
    uint32_t ring_tail;    // consumer reads at ring[tail % SIZE]
    volatile uint32_t ring_count;  // occupied slots (volatile for lock-free peek)
    volatile bool is_closed;

    // Free-list: returned buffers available for reuse (no malloc/free after warmup)
    tasks_buffer* free_arr[TASKS_BUFFERS_SIZE];
    uint32_t num_free;

    pthread_mutex_t mutex;
    pthread_cond_t not_full;
    pthread_cond_t not_empty;
} tasks_buffers;

int tasks_buffers_create(tasks_buffers* buffs);
int tasks_buffers_free(tasks_buffers* buffs);
int tasks_buffers_add_buffer(tasks_buffers* buffs, tasks_buffer* buf);
int tasks_buffers_get_buffer(tasks_buffers* buffs, tasks_buffer** buf);
int tasks_buffers_close(tasks_buffers* buffs);
int tasks_buffers_num_ready(tasks_buffers* buffs);
tasks_buffer* tasks_buffers_obtain(tasks_buffers* buffs);   // get from free-list or allocate
void tasks_buffers_recycle(tasks_buffers* buffs, tasks_buffer* buf);  // return to free-list

#endif //ANABRUTE_TASK_BUFFERS_H
