#ifndef ANABRUTE_TASK_BUFFERS_H
#define ANABRUTE_TASK_BUFFERS_H

#include "common.h"

typedef struct permut_task_s {
    char all_strs[MAX_STR_LENGTH];
    int8_t offsets[MAX_OFFSETS_LENGTH];  // positives - permutable all_strs+(a[offset-1]-1), negatives - fixed all_strs+(-offset-1), zeroes - end terminators;
    uint8_t a[MAX_OFFSETS_LENGTH];
    uint8_t c[MAX_OFFSETS_LENGTH];
    uint32_t i;     // 32_t to align structure to longs
    uint32_t n;
} permut_task;

typedef struct tasks_buffer_s {
    permut_task *permut_tasks;
    uint32_t num_tasks;
} tasks_buffer;

tasks_buffer* tasks_buffer_allocate();
void tasks_buffer_free(tasks_buffer* buf);
bool tasks_buffer_isfull(tasks_buffer* buf);
void tasks_buffer_add_task(tasks_buffer* buf, char* all_strs, int8_t* offsets);

typedef struct tasks_buffers_s {
    volatile tasks_buffer* arr[TASKS_BUFFERS_SIZE];
    volatile uint32_t num_ready;
    volatile bool is_closed;

    pthread_mutex_t mutex;
    pthread_cond_t inc_cond;
    pthread_cond_t dec_cond;
} tasks_buffers;

int tasks_buffers_create(tasks_buffers* buffs);
int tasks_buffers_free(tasks_buffers* buffs);
int tasks_buffers_add_buffer(tasks_buffers* buffs, tasks_buffer* buf);
int tasks_buffers_get_buffer(tasks_buffers* buffs, tasks_buffer** buf);
int tasks_buffers_close(tasks_buffers* buffs);
int tasks_buffers_num_ready(tasks_buffers* buffs);

#endif //ANABRUTE_TASK_BUFFERS_H
