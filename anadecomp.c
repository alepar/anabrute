#include <stdio.h>
#include "fact.h"

#define NUM_DIGITS 8

int main(int argc, char* argv[]) {
    fact_init();

    uint64_t n;
    uint8_t digits[NUM_DIGITS] = {0};

    for (n=fact(NUM_DIGITS)-10; n<=fact(NUM_DIGITS); n++) {
        uint64_t a = n;
        uint8_t picked[NUM_DIGITS] = {0};

        for (int d=NUM_DIGITS-1; d>=0; d--) {
            uint64_t ord = (a - 1) / fact(d);
            a -= ord*fact(d);

            for (uint8_t dd=0; dd<NUM_DIGITS; dd++) {
                if (!picked[dd]) {
                    if (ord == 0) {
                        picked[dd] = 1;
                        digits[d] = dd+1;
                        break;
                    } else {
                        ord--;
                    }
                }
            }
        }

        printf("%lu:\t", n);
        for (int d=NUM_DIGITS-1; d>=0; d--) {
            printf("%c ", digits[d] + '0');
        }
        printf("\n");
    }

}