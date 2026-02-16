#ifndef ANABRUTE_CRUNCHER_H
#define ANABRUTE_CRUNCHER_H

#include "common.h"
#include "task_buffers.h"

typedef struct cruncher_config_s {
    tasks_buffers *tasks_buffs;
    uint32_t *hashes;
    uint32_t hashes_num;
    uint32_t *hashes_reversed;  // shared output buffer (hashes_num * MAX_STR_LENGTH bytes)
} cruncher_config;

typedef struct cruncher_ops_s {
    const char *name;
    uint32_t (*probe)(void);
    int (*create)(void *ctx, cruncher_config *cfg, uint32_t instance_id);
    void *(*run)(void *ctx);
    void (*get_stats)(void *ctx, float *busy_pct, float *anas_per_sec);
    bool (*is_running)(void *ctx);
    int (*destroy)(void *ctx);
    size_t ctx_size;
} cruncher_ops;

#endif //ANABRUTE_CRUNCHER_H
