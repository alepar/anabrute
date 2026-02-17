#include "avx_cruncher.h"
#include "os.h"
#include "task_buffers.h"
#include "md5_avx2.h"
#include "md5_avx512.h"

#include <string.h>
#include <stdio.h>

/* Context for one CPU cruncher thread */
typedef struct {
    cruncher_config *cfg;
    volatile bool is_running;
    volatile uint64_t consumed_bufs;
    volatile uint64_t consumed_anas;
    uint64_t task_time_start;
    uint64_t task_time_end;
} avx_cruncher_ctx;

/* ---------- PUTCHAR_SCALAR: same byte-packing as the OpenCL kernel's PUTCHAR ---------- */
#define PUTCHAR_SCALAR(buf, index, val) \
    (buf)[(index) >> 2] = ((buf)[(index) >> 2] & ~(0xffU << (((index) & 3) << 3))) + ((uint32_t)(val) << (((index) & 3) << 3))

/* ---------- Scalar MD5 — identical algorithm to kernels/permut.cl md5() ---------- */

#define F(x, y, z) ((z) ^ ((x) & ((y) ^ (z))))
#define G(x, y, z) ((y) ^ ((z) & ((x) ^ (y))))
#define H(x, y, z) ((x) ^ (y) ^ (z))
#define I(x, y, z) ((y) ^ ((x) | ~(z)))

#define STEP(f, a, b, c, d, x, t, s) \
    (a) += f((b), (c), (d)) + (x) + (t); \
    (a) = (((a) << (s)) | (((a) & 0xffffffff) >> (32 - (s)))); \
    (a) += (b);

#define GET(i) (key[(i)])

static void md5_scalar(const uint32_t *key, uint32_t *hash) {
    uint32_t a, b, c, d;

    a = 0x67452301;
    b = 0xefcdab89;
    c = 0x98badcfe;
    d = 0x10325476;

    /* Round 1 */
    STEP(F, a, b, c, d, GET(0), 0xd76aa478, 7)
    STEP(F, d, a, b, c, GET(1), 0xe8c7b756, 12)
    STEP(F, c, d, a, b, GET(2), 0x242070db, 17)
    STEP(F, b, c, d, a, GET(3), 0xc1bdceee, 22)
    STEP(F, a, b, c, d, GET(4), 0xf57c0faf, 7)
    STEP(F, d, a, b, c, GET(5), 0x4787c62a, 12)
    STEP(F, c, d, a, b, GET(6), 0xa8304613, 17)
    STEP(F, b, c, d, a, GET(7), 0xfd469501, 22)
    STEP(F, a, b, c, d, GET(8), 0x698098d8, 7)
    STEP(F, d, a, b, c, GET(9), 0x8b44f7af, 12)
    STEP(F, c, d, a, b, GET(10), 0xffff5bb1, 17)
    STEP(F, b, c, d, a, GET(11), 0x895cd7be, 22)
    STEP(F, a, b, c, d, GET(12), 0x6b901122, 7)
    STEP(F, d, a, b, c, GET(13), 0xfd987193, 12)
    STEP(F, c, d, a, b, GET(14), 0xa679438e, 17)
    STEP(F, b, c, d, a, GET(15), 0x49b40821, 22)

    /* Round 2 */
    STEP(G, a, b, c, d, GET(1), 0xf61e2562, 5)
    STEP(G, d, a, b, c, GET(6), 0xc040b340, 9)
    STEP(G, c, d, a, b, GET(11), 0x265e5a51, 14)
    STEP(G, b, c, d, a, GET(0), 0xe9b6c7aa, 20)
    STEP(G, a, b, c, d, GET(5), 0xd62f105d, 5)
    STEP(G, d, a, b, c, GET(10), 0x02441453, 9)
    STEP(G, c, d, a, b, GET(15), 0xd8a1e681, 14)
    STEP(G, b, c, d, a, GET(4), 0xe7d3fbc8, 20)
    STEP(G, a, b, c, d, GET(9), 0x21e1cde6, 5)
    STEP(G, d, a, b, c, GET(14), 0xc33707d6, 9)
    STEP(G, c, d, a, b, GET(3), 0xf4d50d87, 14)
    STEP(G, b, c, d, a, GET(8), 0x455a14ed, 20)
    STEP(G, a, b, c, d, GET(13), 0xa9e3e905, 5)
    STEP(G, d, a, b, c, GET(2), 0xfcefa3f8, 9)
    STEP(G, c, d, a, b, GET(7), 0x676f02d9, 14)
    STEP(G, b, c, d, a, GET(12), 0x8d2a4c8a, 20)

    /* Round 3 */
    STEP(H, a, b, c, d, GET(5), 0xfffa3942, 4)
    STEP(H, d, a, b, c, GET(8), 0x8771f681, 11)
    STEP(H, c, d, a, b, GET(11), 0x6d9d6122, 16)
    STEP(H, b, c, d, a, GET(14), 0xfde5380c, 23)
    STEP(H, a, b, c, d, GET(1), 0xa4beea44, 4)
    STEP(H, d, a, b, c, GET(4), 0x4bdecfa9, 11)
    STEP(H, c, d, a, b, GET(7), 0xf6bb4b60, 16)
    STEP(H, b, c, d, a, GET(10), 0xbebfbc70, 23)
    STEP(H, a, b, c, d, GET(13), 0x289b7ec6, 4)
    STEP(H, d, a, b, c, GET(0), 0xeaa127fa, 11)
    STEP(H, c, d, a, b, GET(3), 0xd4ef3085, 16)
    STEP(H, b, c, d, a, GET(6), 0x04881d05, 23)
    STEP(H, a, b, c, d, GET(9), 0xd9d4d039, 4)
    STEP(H, d, a, b, c, GET(12), 0xe6db99e5, 11)
    STEP(H, c, d, a, b, GET(15), 0x1fa27cf8, 16)
    STEP(H, b, c, d, a, GET(2), 0xc4ac5665, 23)

    /* Round 4 */
    STEP(I, a, b, c, d, GET(0), 0xf4292244, 6)
    STEP(I, d, a, b, c, GET(7), 0x432aff97, 10)
    STEP(I, c, d, a, b, GET(14), 0xab9423a7, 15)
    STEP(I, b, c, d, a, GET(5), 0xfc93a039, 21)
    STEP(I, a, b, c, d, GET(12), 0x655b59c3, 6)
    STEP(I, d, a, b, c, GET(3), 0x8f0ccc92, 10)
    STEP(I, c, d, a, b, GET(10), 0xffeff47d, 15)
    STEP(I, b, c, d, a, GET(1), 0x85845dd1, 21)
    STEP(I, a, b, c, d, GET(8), 0x6fa87e4f, 6)
    STEP(I, d, a, b, c, GET(15), 0xfe2ce6e0, 10)
    STEP(I, c, d, a, b, GET(6), 0xa3014314, 15)
    STEP(I, b, c, d, a, GET(13), 0x4e0811a1, 21)
    STEP(I, a, b, c, d, GET(4), 0xf7537e82, 6)
    STEP(I, d, a, b, c, GET(11), 0xbd3af235, 10)
    STEP(I, c, d, a, b, GET(2), 0x2ad7d2bb, 15)
    STEP(I, b, c, d, a, GET(9), 0xeb86d391, 21)

    hash[0] = a + 0x67452301;
    hash[1] = b + 0xefcdab89;
    hash[2] = c + 0x98badcfe;
    hash[3] = d + 0x10325476;
}

#undef F
#undef G
#undef H
#undef I
#undef STEP
#undef GET

/* ---------- String construction from permut_task ---------- */

/*
 * Constructs the candidate string from a permut_task into a uint32_t[16] key buffer
 * suitable for MD5. Applies MD5 padding. Returns the string length (excluding padding).
 *
 * Uses memcpy instead of byte-at-a-time PUTCHAR_SCALAR for ~4-8x faster string
 * construction. The key buffer is pre-zeroed so MD5 padding bytes beyond the string
 * are already correct.
 *
 * Offset resolution logic (matches OpenCL kernel exactly):
 *   - Iterate offsets[] until hitting 0
 *   - Positive offset: look up a[offset-1] to get 1-based byte offset in all_strs, subtract 1
 *   - Negative offset: use (-offset)-1 as byte offset
 *   - Copy characters, add space between words
 *   - Strip trailing space, add 0x80 padding byte, write bit length at offset 56-57
 */
static int construct_string(permut_task *task, uint32_t *key) {
    memset(key, 0, 64);
    char *dst = (char *)key;
    int wcs = 0;
    for (int io = 0; task->offsets[io]; io++) {
        int8_t off = task->offsets[io];
        if (off < 0) {
            off = -off - 1;
        } else {
            off = task->a[off - 1] - 1;
        }

        /* Find word length and copy with memcpy */
        const char *word = &task->all_strs[(uint8_t)off];
        int len = 0;
        while (word[len]) len++;
        memcpy(dst + wcs, word, len);
        wcs += len;
        dst[wcs] = ' ';
        wcs++;
    }
    wcs--;  /* remove trailing space */

    /* MD5 padding */
    dst[wcs] = (char)0x80;
    dst[56] = (char)(wcs << 3);
    dst[57] = (char)(wcs >> 5);
    return wcs;
}

/* ---------- Heap's permutation algorithm ---------- */

/*
 * Advance Heap's permutation algorithm to the next permutation.
 * Returns true if a new permutation was generated, false if exhausted.
 *
 * Uses a tmp-variable swap (functionally equivalent to the OpenCL kernel's XOR swap).
 */
static bool heap_next(permut_task *task) {
    while (task->i < task->n) {
        if (task->c[task->i] < task->i) {
            if (task->i % 2 == 0) {
                uint8_t tmp = task->a[0];
                task->a[0] = task->a[task->i];
                task->a[task->i] = tmp;
            } else {
                uint8_t tmp = task->a[task->c[task->i]];
                task->a[task->c[task->i]] = task->a[task->i];
                task->a[task->i] = tmp;
            }
            task->c[task->i]++;
            task->i = 0;
            return true;
        } else {
            task->c[task->i] = 0;
            task->i++;
        }
    }
    return false;
}

/* ---------- Process one task (all permutations) ---------- */

static void check_hashes(cruncher_config *cfg, uint32_t *hash, uint32_t *key, int wcs) {
    for (uint32_t ih = 0; ih < cfg->hashes_num; ih++) {
        if (hash[0] == cfg->hashes[4 * ih] &&
            hash[1] == cfg->hashes[4 * ih + 1] &&
            hash[2] == cfg->hashes[4 * ih + 2] &&
            hash[3] == cfg->hashes[4 * ih + 3]) {
            PUTCHAR_SCALAR(key, wcs, 0);
            memcpy(cfg->hashes_reversed + ih * MAX_STR_LENGTH / 4,
                   key, MAX_STR_LENGTH);
        }
    }
}

/*
 * SIMD hash check: compare 16 hash[0] values against all targets using AVX-512.
 * Only extracts and does full 4-word comparison for lanes with hash[0] match.
 * Returns without extracting anything for the ~100% case of no matches.
 */
#if defined(__x86_64__) && defined(__AVX512F__)
static void check_hashes_avx512(cruncher_config *cfg,
                                __m512i ha, __m512i hb, __m512i hc, __m512i hd,
                                uint32_t keys[16][16], int wcs_arr[16], int count) {
    /* SIMD early-exit: compare hash[0] across all 16 lanes against all targets */
    __mmask16 any_match = 0;
    for (uint32_t ih = 0; ih < cfg->hashes_num; ih++) {
        any_match |= _mm512_cmpeq_epi32_mask(ha,
                         _mm512_set1_epi32((int32_t)cfg->hashes[4 * ih]));
    }
    if (!any_match) return;  /* fast path: no hash[0] match in any lane */

    /* Slow path: extract scalars and do full 4-word comparison */
    uint32_t a_vals[16], b_vals[16], c_vals[16], d_vals[16];
    _mm512_storeu_si512(a_vals, ha);
    _mm512_storeu_si512(b_vals, hb);
    _mm512_storeu_si512(c_vals, hc);
    _mm512_storeu_si512(d_vals, hd);

    __mmask16 remaining = any_match & (((uint32_t)1 << count) - 1);
    while (remaining) {
        int lane = __builtin_ctz(remaining);
        remaining &= remaining - 1;
        uint32_t hash[4] = {a_vals[lane], b_vals[lane], c_vals[lane], d_vals[lane]};
        check_hashes(cfg, hash, keys[lane], wcs_arr[lane]);
    }
}
#endif

static void process_task(avx_cruncher_ctx *actx, permut_task *task) {
    if (task->i >= task->n) return;
    cruncher_config *cfg = actx->cfg;

#if defined(__x86_64__) && defined(__AVX512F__)
    /* --- Precompute word lengths (indexed by byte offset in all_strs) ---
     * The set of byte offsets never changes during permutation (only their order
     * in a[] changes), so we can compute strlen once per task. */
    uint8_t wlen[MAX_STR_LENGTH];
    memset(wlen, 0, sizeof(wlen));
    int num_offsets = 0;
    for (int io = 0; task->offsets[io]; io++) {
        int8_t off = task->offsets[io];
        int byte_off = (off < 0) ? (-off - 1) : (task->a[off - 1] - 1);
        if (!wlen[byte_off]) {
            int l = 0;
            while (task->all_strs[byte_off + l]) l++;
            wlen[byte_off] = l;
        }
        num_offsets = io + 1;
    }

    /* --- Main permutation loop with inlined string construction ---
     *
     * All permutations of a task have the same total string length (same words,
     * different order). Zero all key buffers once upfront; subsequent permutations
     * only overwrite word data bytes (0..wcs-1) + constant padding at wcs, 56-57.
     * Bytes wcs+1..55 and 58..63 stay 0 from the initial memset.
     */
    uint32_t keys[16][16];
    memset(keys, 0, sizeof(keys));
    int wcs_arr[16];
    int batch = 0;

    do {
        char *dst = (char *)keys[batch];
        int wcs = 0;
        for (int io = 0; io < num_offsets; io++) {
            int8_t off = task->offsets[io];
            int byte_off = (off < 0) ? (-off - 1) : (task->a[off - 1] - 1);
            int len = wlen[byte_off];
            memcpy(dst + wcs, &task->all_strs[byte_off], len);
            wcs += len;
            dst[wcs++] = ' ';
        }
        wcs--;
        dst[wcs] = (char)0x80;
        dst[56] = (char)(wcs << 3);
        dst[57] = (char)(wcs >> 5);
        wcs_arr[batch] = wcs;
        batch++;

        if (batch == 16) {
            __m512i ha, hb, hc, hd;
            md5_avx512_x16_vec((const uint32_t *)keys, &ha, &hb, &hc, &hd);
            check_hashes_avx512(cfg, ha, hb, hc, hd, keys, wcs_arr, 16);
            batch = 0;
        }
    } while (heap_next(task));

    /* Tail: pad to 16 with duplicates of last key */
    if (batch > 0) {
        for (int i = batch; i < 16; i++)
            memcpy(keys[i], keys[batch - 1], 64);
        __m512i ha, hb, hc, hd;
        md5_avx512_x16_vec((const uint32_t *)keys, &ha, &hb, &hc, &hd);
        check_hashes_avx512(cfg, ha, hb, hc, hd, keys, wcs_arr, batch);
    }
#elif defined(__x86_64__) || defined(_M_AMD64)
    /* Batch 8 permutations for AVX2 MD5 */
    uint32_t keys[8][16];
    int wcs_arr[8];
    int batch = 0;

    do {
        wcs_arr[batch] = construct_string(task, keys[batch]);
        batch++;

        if (batch == 8) {
            const uint32_t *key_ptrs[8];
            uint32_t *hash_ptrs[8];
            uint32_t hashes_out[8][4];
            for (int i = 0; i < 8; i++) {
                key_ptrs[i] = keys[i];
                hash_ptrs[i] = hashes_out[i];
            }

            md5_avx2_x8(key_ptrs, hash_ptrs);

            for (int i = 0; i < 8; i++) {
                check_hashes(cfg, hashes_out[i], keys[i], wcs_arr[i]);
            }
            batch = 0;
        }
    } while (heap_next(task));

    /* Tail: pad to 8 with duplicates of last key, hash all, check only real ones */
    if (batch > 0) {
        const uint32_t *key_ptrs[8];
        uint32_t *hash_ptrs[8];
        uint32_t hashes_out[8][4];
        for (int i = 0; i < 8; i++) {
            key_ptrs[i] = (i < batch) ? keys[i] : keys[batch - 1];
            hash_ptrs[i] = hashes_out[i];
        }
        md5_avx2_x8(key_ptrs, hash_ptrs);
        for (int i = 0; i < batch; i++) {
            check_hashes(cfg, hashes_out[i], keys[i], wcs_arr[i]);
        }
    }
#else
    do {
        uint32_t key[16];
        int wcs = construct_string(task, key);
        uint32_t hash[4];
        md5_scalar(key, hash);
        check_hashes(cfg, hash, key, wcs);
    } while (heap_next(task));
#endif
}

/* ---------- Vtable functions ---------- */

static uint32_t avx_probe(void) {
#if defined(__x86_64__) || defined(_M_AMD64)
    uint32_t cores = num_cpu_cores();
    printf("  avx: %d cores available, suggesting %d thread(s)\n", cores, cores);
    return cores;
#else
    return 0;
#endif
}

static uint32_t scalar_probe(void) {
    uint32_t cores = num_cpu_cores();
    printf("  cpu: %d cores available, suggesting %d thread(s)\n", cores, cores);
    return cores;
}

static int avx_create(void *ctx, cruncher_config *cfg, uint32_t instance_id) {
    avx_cruncher_ctx *actx = ctx;
    actx->cfg = cfg;
    actx->is_running = false;
    actx->consumed_bufs = 0;
    actx->consumed_anas = 0;
    actx->task_time_start = 0;
    actx->task_time_end = 0;
    return 0;
}

static void *avx_run(void *ctx) {
    avx_cruncher_ctx *actx = ctx;
    actx->is_running = true;
    actx->task_time_start = current_micros();

    tasks_buffer *buf;
    while (1) {
        tasks_buffers_get_buffer(actx->cfg->tasks_buffs, &buf);
        if (buf == NULL) break;

        for (uint32_t i = 0; i < buf->num_tasks; i++) {
            process_task(actx, &buf->permut_tasks[i]);
        }

        actx->consumed_anas += buf->num_anas;
        actx->consumed_bufs++;
        tasks_buffers_recycle(actx->cfg->tasks_buffs, buf);
    }

    actx->task_time_end = current_micros();
    actx->is_running = false;
    return NULL;
}

static void avx_get_stats(void *ctx, float *busy_pct, float *anas_per_sec) {
    avx_cruncher_ctx *actx = ctx;
    uint64_t now = current_micros();
    uint64_t start = actx->task_time_start;
    uint64_t end = actx->task_time_end;

    if (!start) {
        *busy_pct = 0;
        *anas_per_sec = 0;
        return;
    }

    uint64_t elapsed = (end ? end : now) - start;
    if (elapsed == 0) elapsed = 1;

    /* CPU thread is always 100% busy while running (no idle gaps between
     * kernel dispatches like the GPU backend), so this binary metric is
     * accurate — unlike the GPU's windowed kernel-time measurement. */
    *busy_pct = actx->is_running ? 100.0f : 0.0f;
    *anas_per_sec = (float)actx->consumed_anas / ((float)elapsed / 1000000.0f);
}

static uint64_t avx_get_total_anas(void *ctx) {
    return ((avx_cruncher_ctx *)ctx)->consumed_anas;
}

static bool avx_is_running(void *ctx) {
    return ((avx_cruncher_ctx *)ctx)->is_running;
}

static int avx_destroy(void *ctx) {
    return 0;
}

cruncher_ops avx_cruncher_ops = {
    .name = "avx",
    .probe = avx_probe,
    .create = avx_create,
    .run = avx_run,
    .get_stats = avx_get_stats,
    .get_total_anas = avx_get_total_anas,
    .is_running = avx_is_running,
    .destroy = avx_destroy,
    .ctx_size = sizeof(avx_cruncher_ctx),
};

cruncher_ops scalar_cruncher_ops = {
    .name = "cpu",
    .probe = scalar_probe,
    .create = avx_create,
    .run = avx_run,
    .get_stats = avx_get_stats,
    .get_total_anas = avx_get_total_anas,
    .is_running = avx_is_running,
    .destroy = avx_destroy,
    .ctx_size = sizeof(avx_cruncher_ctx),
};
