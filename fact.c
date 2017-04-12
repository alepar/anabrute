#include "fact.h"


uint64_t facts[21];

void fact_init() {
    facts[0] = 1;
    for (int i=1; i<21; i++) {
        facts[i] = i*facts[i-1];
    }
}

uint64_t fact(int x) {
    if (x < 0) {
        return 0;
    }
    if (x>20) {
        return 0;
    }

    return facts[x];
}