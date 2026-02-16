#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include "hashes.h"

static void write_file(const char *path, const char *content) {
    FILE *f = fopen(path, "w");
    assert(f && "failed to create test file");
    fputs(content, f);
    fclose(f);
}

/*
 * Test 1: Valid file with 3 known MD5 hashes.
 * Verifies read_hashes returns correct count and parsed values roundtrip.
 */
void test_valid_hashes(void) {
    const char *path = "/tmp/anabrute_test_hashes_valid.txt";
    write_file(path,
        "d41d8cd98f00b204e9800998ecf8427e\n"   /* MD5("") */
        "0cc175b9c0f1b6a831c399e269772661\n"   /* MD5("a") */
        "92eb5ffee6ae2fec3ad71c777531578f\n");  /* MD5("abc") */

    uint32_t *hashes = NULL;
    uint32_t count = read_hashes(path, &hashes);

    assert(count == 3);
    assert(hashes != NULL);

    /* Roundtrip: parse -> format -> compare against original */
    char buf[33];
    hash_to_ascii(hashes + 0*4, buf);
    assert(strcmp(buf, "d41d8cd98f00b204e9800998ecf8427e") == 0);

    hash_to_ascii(hashes + 1*4, buf);
    assert(strcmp(buf, "0cc175b9c0f1b6a831c399e269772661") == 0);

    hash_to_ascii(hashes + 2*4, buf);
    assert(strcmp(buf, "92eb5ffee6ae2fec3ad71c777531578f") == 0);

    free(hashes);
    unlink(path);
    printf("  PASS: test_valid_hashes\n");
}

/*
 * Test 2: File with an invalid (non-32-char) line mixed in.
 * BUGS.md #3: invalid lines should be skipped, not parsed.
 * Before fix: count=4 (invalid line counted). After fix: count=3.
 */
void test_invalid_line_skipped(void) {
    const char *path = "/tmp/anabrute_test_hashes_invalid.txt";
    write_file(path,
        "d41d8cd98f00b204e9800998ecf8427e\n"
        "0cc175b9c0f1b6a831c399e269772661\n"
        "not_a_valid_hash\n"
        "92eb5ffee6ae2fec3ad71c777531578f\n");

    uint32_t *hashes = NULL;
    uint32_t count = read_hashes(path, &hashes);

    assert(count == 3 && "invalid line should not be counted");

    /* Verify the 3 valid hashes were parsed correctly */
    char buf[33];
    hash_to_ascii(hashes + 0*4, buf);
    assert(strcmp(buf, "d41d8cd98f00b204e9800998ecf8427e") == 0);
    hash_to_ascii(hashes + 1*4, buf);
    assert(strcmp(buf, "0cc175b9c0f1b6a831c399e269772661") == 0);
    hash_to_ascii(hashes + 2*4, buf);
    assert(strcmp(buf, "92eb5ffee6ae2fec3ad71c777531578f") == 0);

    free(hashes);
    unlink(path);
    printf("  PASS: test_invalid_line_skipped\n");
}

/*
 * Test 3: Short invalid line makes file size estimate too low,
 * causing more lines than allocated slots.
 * BUGS.md #2 + #3: the short line throws off (file_size+1)/33 estimate,
 * and without the continue, it writes past the allocated buffer.
 * Under ASan: crashes with heap-buffer-overflow before the fix.
 */
void test_short_invalid_line_no_overflow(void) {
    const char *path = "/tmp/anabrute_test_hashes_overflow.txt";
    /* File: 33 + 2 + 33 = 68 bytes. Estimate: (68+1)/33 = 2. Actual lines: 3. */
    write_file(path,
        "d41d8cd98f00b204e9800998ecf8427e\n"
        "x\n"
        "0cc175b9c0f1b6a831c399e269772661\n");

    uint32_t *hashes = NULL;
    uint32_t count = read_hashes(path, &hashes);

    /* After fix: invalid "x" skipped, count=2, no overflow */
    assert(count == 2);

    free(hashes);
    unlink(path);
    printf("  PASS: test_short_invalid_line_no_overflow\n");
}

int main(void) {
    printf("test_hash_parsing:\n");
    test_valid_hashes();
    test_invalid_line_skipped();
    test_short_invalid_line_no_overflow();
    printf("All hash parsing tests passed!\n");
    return 0;
}
