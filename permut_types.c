#include "common.h"
#include "permut_types.h"

#if defined(__x86_64__) || defined(_M_AMD64)
#include <immintrin.h>
#elif defined(__aarch64__) || defined(_M_ARM64)
#include <arm_neon.h>
#endif

bool char_counts_create(const char *s, char_counts *cc) {
    memset(cc->counts, 0, 16);
    cc->length = 0;

    for (int i=0; s[i]; i++) {
        cc->length++;
        int idx = char_to_index(s[i]);
        if (idx >= 0) {
            cc->counts[idx]++;
        } else {
            return true;
        }
    }

    return false;
}

bool char_counts_subtract(char_counts *from, char_counts *what) {
    if (from->length < what->length) {
        return false;
    }

#if defined(__x86_64__) || defined(_M_AMD64)
    /* SSE2: check all 12 counts atomically, then store if no underflow */
    __m128i from_v = _mm_loadu_si128((__m128i *)from->counts);
    __m128i what_v = _mm_loadu_si128((__m128i *)what->counts);
    __m128i sat = _mm_subs_epu8(from_v, what_v);   /* saturating: clamps at 0 */
    __m128i real = _mm_sub_epi8(from_v, what_v);    /* wrapping: underflows wrap */
    /* If sat != real for any of first 12 bytes, underflow occurred */
    if (_mm_movemask_epi8(_mm_xor_si128(sat, real)) & 0xFFF)
        return false;
    _mm_storeu_si128((__m128i *)from->counts, sat);
#elif defined(__aarch64__) || defined(_M_ARM64)
    /* NEON: check all 12 counts atomically */
    uint8x16_t from_v = vld1q_u8(from->counts);
    uint8x16_t what_v = vld1q_u8(what->counts);
    uint8x16_t lt = vcltq_u8(from_v, what_v);
    /* Check only first 12 bytes for underflow */
    uint64_t lo = vgetq_lane_u64(vreinterpretq_u64_u8(lt), 0);
    uint32_t hi = vgetq_lane_u32(vreinterpretq_u32_u8(lt), 2);
    if (lo | hi)
        return false;
    vst1q_u8(from->counts, vsubq_u8(from_v, what_v));
#else
    for (int i = 0; i < CHARCOUNT; i++) {
        if (from->counts[i] < what->counts[i])
            return false;
    }
    for (int i = 0; i < CHARCOUNT; i++) {
        from->counts[i] -= what->counts[i];
    }
#endif

    from->length -= what->length;
    return true;
}

void char_counts_copy(char_counts *src, char_counts *dst) {
    memcpy(dst, src, sizeof(char_counts));
}

uint8_t char_counts_equal(char_counts *l, char_counts *r) {
    if (l->length != r->length) {
        return 0;
    }

    for(int i=0; i<CHARCOUNT; i++) {
        if (l->counts[i] != r->counts[i]) {
            return 0;
        }
    }

    return 1;
}

bool char_counts_strings_create(const char *s, char_counts_strings *ccs) {
    ccs->strings_len = 0;
    ccs->strings = malloc(sizeof(char*)*MAX_STRINGS_SIZE);
    return char_counts_create(s, &ccs->counts);
}

bool char_counts_strings_addstring(char_counts_strings *ccs, const char *s) {
    const size_t len = strlen(s) + 1;
    char* sc = malloc(len);
    memcpy(sc, s, len);
    ccs->strings[ccs->strings_len++] = sc;
    return ccs->strings_len > MAX_STRINGS_SIZE;
}

void char_counts_strings_free(char_counts_strings *ccs) {
    for (int i = 0; i < ccs->strings_len; i++) {
        free(ccs->strings[i]);
    }
    free(ccs->strings);
    ccs->strings = NULL;
    ccs->strings_len = 0;
}

bool char_counts_contains(char_counts* cc, char_counts* subcc) {
    if (cc->length < subcc->length) {
        return false;
    }

    for (int i=0; i<CHARCOUNT; i++) {
        if (cc->counts[i] < subcc->counts[i]) {
            return false;
        }
    }

    return true;
}