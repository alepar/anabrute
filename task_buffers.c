#include "task_buffers.h"

tasks_buffer* tasks_buffer_allocate() {
    tasks_buffer* buffer = calloc(1, sizeof(tasks_buffer));
    if(!buffer) return NULL;

    buffer->permut_tasks = calloc(PERMUT_TASKS_IN_BATCH, sizeof(permut_task));
    buffer->num_tasks = 0;
    return buffer;
}

void tasks_buffer_free(tasks_buffer* buf) {
    free(buf->permut_tasks);
    free(buf);
}

bool tasks_buffer_isfull(tasks_buffer* buf) {
    return buf->num_tasks >= PERMUT_TASKS_IN_BATCH;
}

void tasks_buffer_add_task(tasks_buffer* buf, char* all_strs, int8_t* offsets) {
    memcpy(&((buf->permut_tasks+buf->num_tasks)->all_strs), all_strs, MAX_STR_LENGTH);
    memcpy(&((buf->permut_tasks+buf->num_tasks)->offsets), offsets, MAX_OFFSETS_LENGTH);

    buf->num_tasks++;
}

int tasks_buffers_create(tasks_buffers* buffs) {
    buffs->num_ready = 0;

    for (int i=0; i<TASKS_BUFFERS_SIZE; i++) {
        buffs->arr[i] = NULL;
    }

    int errcode;
    errcode = pthread_mutex_init(&buffs->mutex, NULL);
    ret_iferr(errcode, "failed to init mutex while creating tasks buffers");
    errcode = pthread_cond_init(&buffs->inc_cond, NULL);
    ret_iferr(errcode, "failed to init inc_cond while creating tasks buffers");
    errcode = pthread_cond_init(&buffs->dec_cond, NULL);
    ret_iferr(errcode, "failed to init dec_cond while creating tasks buffers");

    return 0;
}

int tasks_buffers_free(tasks_buffers* buffs) {
    int errcode = 0;
    errcode |= pthread_mutex_destroy(&buffs->mutex);
    errcode |= pthread_cond_destroy(&buffs->inc_cond);
    errcode |= pthread_cond_destroy(&buffs->dec_cond);
    return errcode;
}

int tasks_buffers_add_buffer(tasks_buffers* buffs, tasks_buffer* buf) {
    int errcode=0;
    errcode = pthread_mutex_lock(&buffs->mutex);
    ret_iferr(errcode, "failed to lock mutex while adding buffer");

    while (buffs->num_ready >= TASKS_BUFFERS_SIZE) {
        errcode = pthread_cond_wait(&buffs->dec_cond, &buffs->mutex);
        if (errcode) {
            pthread_mutex_unlock(&buffs->mutex);
            ret_iferr(errcode, "failed to wait for free space while adding buffer");
        }
    }

    for (int i=0; i<TASKS_BUFFERS_SIZE; i++) {
        if (buffs->arr[i] == NULL) {
            buffs->arr[i] = buf;
            break;
        }
    }

    buffs->num_ready++;

    errcode = pthread_cond_signal(&buffs->inc_cond);
    if (errcode) {
        pthread_mutex_unlock(&buffs->mutex);
        ret_iferr(errcode, "failed to signal increase while adding buffer");
    }

    pthread_mutex_unlock(&buffs->mutex);

    return 0;
}


int tasks_buffers_get_buffer(tasks_buffers* buffs, tasks_buffer** buf) {
    int errcode=0;
    errcode = pthread_mutex_lock(&buffs->mutex);
    ret_iferr(errcode, "failed to lock mutex while removing buffer");

    while (buffs->num_ready == 0) {
        errcode = pthread_cond_wait(&buffs->inc_cond, &buffs->mutex);
        if (errcode) {
            pthread_mutex_unlock(&buffs->mutex);
            ret_iferr(errcode, "failed to wait for available buffer while removing buffer");
        }
    }

    for (int i=0; i<TASKS_BUFFERS_SIZE; i++) {
        if (buffs->arr[i] != NULL) {
            *buf = buffs->arr[i];
            buffs->arr[i] = NULL;
            break;
        }
    }

    buffs->num_ready--;

    errcode = pthread_cond_signal(&buffs->dec_cond);
    if (errcode) {
        pthread_mutex_unlock(&buffs->mutex);
        ret_iferr(errcode, "failed to signal decrease while adding buffer");
    }

    pthread_mutex_unlock(&buffs->mutex);

    return 0;
}
