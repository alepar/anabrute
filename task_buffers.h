#ifndef ANABRUTE_TASK_BUFFERS_H
#define ANABRUTE_TASK_BUFFERS_H

#include "common.h"

typedef struct permut_task_s {
    char all_strs[MAX_STR_LENGTH];
    int8_t offsets[MAX_OFFSETS_LENGTH];  // positives - permutable, negatives - fixed, zeroes - empty; abs(offset)-1 to get offset in all_strs
} permut_task;

typedef struct tasks_buffer_s {
    permut_task *permut_tasks;
    uint32_t num_tasks;
} tasks_buffer;

tasks_buffer* tasks_buffer_allocate();
void tasks_buffer_free(tasks_buffer* buf);
bool tasks_buffer_isfull(tasks_buffer* buf);
void tasks_buffer_add_task(tasks_buffer* buf, char* all_strs, int8_t* offsets);

#define TASKS_BUFFERS_SIZE 256

typedef struct tasks_buffers_s {
    volatile tasks_buffer* arr[TASKS_BUFFERS_SIZE];
    volatile uint32_t num_ready;
    pthread_mutex_t mutex;
    pthread_cond_t inc_cond;
    pthread_cond_t dec_cond;
} tasks_buffers;

int tasks_buffers_create(tasks_buffers* buffs);
int tasks_buffers_free(tasks_buffers* buffs);
int tasks_buffers_add_buffer(tasks_buffers* buffs, tasks_buffer* buf);
int tasks_buffers_get_buffer(tasks_buffers* buffs, tasks_buffer** buf);

#endif //ANABRUTE_TASK_BUFFERS_H
