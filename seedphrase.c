#include "seedphrase.h"

const char* seed_phrase_str = "tyranousplutotwits";

int char_to_index(char c) {
    switch (c) {
        case 's':   return 0;
        case 't':   return 1;
        case 'a':   return 2;
        case 'o':   return 3;
        case 'i':   return 4;
        case 'r':   return 5;
        case 'n':   return 6;
        case 'p':   return 7;
        case 'l':   return 8;
        case 'u':   return 9;
        case 'y':   return 10;
        case 'w':   return 11;
        default:    return -1;
    }
}
