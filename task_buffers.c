#include "task_buffers.h"
#include "fact.h"

tasks_buffer* tasks_buffer_allocate() {
    tasks_buffer* buffer = calloc(1, sizeof(tasks_buffer));
    if(!buffer) return NULL;

    buffer->permut_tasks = calloc(PERMUT_TASKS_IN_KERNEL_TASK, sizeof(permut_task));
    buffer->num_tasks = 0;
    buffer->num_anas = 0;
    return buffer;
}

void tasks_buffer_free(tasks_buffer* buf) {
    if (buf) {
        free(buf->permut_tasks);
        free(buf);
    }
}

void tasks_buffer_reset(tasks_buffer* buf) {
    buf->num_tasks = 0;
    buf->num_anas = 0;
}

bool tasks_buffer_isfull(tasks_buffer* buf) {
    return buf->num_tasks >= PERMUT_TASKS_IN_KERNEL_TASK;
}

void tasks_buffer_add_task(tasks_buffer* buf, char* all_strs, int8_t* offsets) {
    permut_task *dst_task = buf->permut_tasks + buf->num_tasks;

    int permutable_count = 0;
    int a_idx = 0;
    for (int i=0; offsets[i]; i++) {
        if (offsets[i] > 0) {
            permutable_count++;

            dst_task->a[a_idx] = offsets[i];
            offsets[i] = a_idx+1;
            a_idx++;
        }
    }

    dst_task->n = permutable_count;
    dst_task->i = 0;
    dst_task->iters_done = 0;

    memcpy(&dst_task->all_strs, all_strs, MAX_STR_LENGTH);
    memcpy(&dst_task->offsets, offsets, MAX_OFFSETS_LENGTH);

    memset(&dst_task->c, 0, MAX_OFFSETS_LENGTH);

    buf->num_tasks++;
    buf->num_anas += fact(permutable_count);
}

int tasks_buffers_create(tasks_buffers* buffs) {
    buffs->ring_head = 0;
    buffs->ring_tail = 0;
    buffs->ring_count = 0;
    buffs->num_free = 0;
    buffs->is_closed = false;

    for (int i=0; i<TASKS_BUFFERS_SIZE; i++) {
        buffs->ring[i] = NULL;
        buffs->free_arr[i] = NULL;
    }

    int errcode;
    errcode = pthread_mutex_init(&buffs->mutex, NULL);
    ret_iferr(errcode, "failed to init mutex while creating tasks buffers");
    errcode = pthread_cond_init(&buffs->not_full, NULL);
    ret_iferr(errcode, "failed to init not_full cond while creating tasks buffers");
    errcode = pthread_cond_init(&buffs->not_empty, NULL);
    ret_iferr(errcode, "failed to init not_empty cond while creating tasks buffers");

    return 0;
}

int tasks_buffers_free(tasks_buffers* buffs) {
    // Free any buffers still in the ring
    while (buffs->ring_count > 0) {
        uint32_t idx = buffs->ring_tail % TASKS_BUFFERS_SIZE;
        tasks_buffer_free(buffs->ring[idx]);
        buffs->ring[idx] = NULL;
        buffs->ring_tail++;
        buffs->ring_count--;
    }
    // Free any buffers in the free-list
    for (uint32_t i = 0; i < buffs->num_free; i++) {
        tasks_buffer_free(buffs->free_arr[i]);
        buffs->free_arr[i] = NULL;
    }
    buffs->num_free = 0;

    int errcode = 0;
    errcode |= pthread_mutex_destroy(&buffs->mutex);
    errcode |= pthread_cond_destroy(&buffs->not_full);
    errcode |= pthread_cond_destroy(&buffs->not_empty);
    return errcode;
}

int tasks_buffers_add_buffer(tasks_buffers* buffs, tasks_buffer* buf) {
    int errcode=0;
    errcode = pthread_mutex_lock(&buffs->mutex);
    ret_iferr(errcode, "failed to lock mutex while adding buffer");

    while (buffs->ring_count >= TASKS_BUFFERS_SIZE) {
        errcode = pthread_cond_wait(&buffs->not_full, &buffs->mutex);
        if (errcode) {
            pthread_mutex_unlock(&buffs->mutex);
            ret_iferr(errcode, "failed to wait for free space while adding buffer");
        }
    }

    buffs->ring[buffs->ring_head % TASKS_BUFFERS_SIZE] = buf;
    buffs->ring_head++;
    buffs->ring_count++;

    errcode = pthread_cond_signal(&buffs->not_empty);
    if (errcode) {
        pthread_mutex_unlock(&buffs->mutex);
        ret_iferr(errcode, "failed to signal not_empty while adding buffer");
    }

    pthread_mutex_unlock(&buffs->mutex);

    return 0;
}


int tasks_buffers_get_buffer(tasks_buffers* buffs, tasks_buffer** buf) {
    int errcode=0;
    errcode = pthread_mutex_lock(&buffs->mutex);
    ret_iferr(errcode, "failed to lock mutex while removing buffer");

    while (buffs->ring_count == 0) {
        if (buffs->is_closed) {
            pthread_mutex_unlock(&buffs->mutex);
            *buf = NULL;
            return 0;
        }

        errcode = pthread_cond_wait(&buffs->not_empty, &buffs->mutex);
        if (errcode) {
            pthread_mutex_unlock(&buffs->mutex);
            ret_iferr(errcode, "failed to wait for available buffer while removing buffer");
        }
    }

    *buf = buffs->ring[buffs->ring_tail % TASKS_BUFFERS_SIZE];
    buffs->ring[buffs->ring_tail % TASKS_BUFFERS_SIZE] = NULL;
    buffs->ring_tail++;
    buffs->ring_count--;

    errcode = pthread_cond_signal(&buffs->not_full);
    if (errcode) {
        pthread_mutex_unlock(&buffs->mutex);
        ret_iferr(errcode, "failed to signal not_full while removing buffer");
    }

    pthread_mutex_unlock(&buffs->mutex);

    return 0;
}

int tasks_buffers_close(tasks_buffers* buffs) {
    int errcode=0;
    errcode = pthread_mutex_lock(&buffs->mutex);
    ret_iferr(errcode, "failed to lock mutex while closing buffer");

    buffs->is_closed = true;

    errcode = pthread_cond_broadcast(&buffs->not_empty);
    if (errcode) {
        pthread_mutex_unlock(&buffs->mutex);
        ret_iferr(errcode, "failed to broadcast close");
    }

    pthread_mutex_unlock(&buffs->mutex);

    return 0;
}

int tasks_buffers_num_ready(tasks_buffers* buffs) {
    // opportunistically peek
    if (buffs->is_closed) {
        return -1;
    }

    return buffs->ring_count;
}

tasks_buffer* tasks_buffers_obtain(tasks_buffers* buffs) {
    pthread_mutex_lock(&buffs->mutex);
    tasks_buffer *buf = NULL;
    if (buffs->num_free > 0) {
        buf = buffs->free_arr[--buffs->num_free];
    }
    pthread_mutex_unlock(&buffs->mutex);

    if (buf) {
        tasks_buffer_reset(buf);
        return buf;
    }
    return tasks_buffer_allocate();
}

void tasks_buffers_recycle(tasks_buffers* buffs, tasks_buffer* buf) {
    pthread_mutex_lock(&buffs->mutex);
    if (buffs->num_free < TASKS_BUFFERS_SIZE) {
        buffs->free_arr[buffs->num_free++] = buf;
        pthread_mutex_unlock(&buffs->mutex);
    } else {
        pthread_mutex_unlock(&buffs->mutex);
        free(buf->permut_tasks);
        free(buf);
    }
}
