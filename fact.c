#include "fact.h"

uint64_t fact(int x) {
    switch(x) {
        case 0: 	return 1L;
        case 1: 	return 1L;
        case 2: 	return 2L;
        case 3: 	return 6L;
        case 4: 	return 24L;
        case 5: 	return 120L;
        case 6: 	return 720L;
        case 7: 	return 5040L;
        case 8: 	return 40320L;
        case 9: 	return 362880L;
        case 10: 	return 3628800L;
        case 11: 	return 39916800L;
        case 12: 	return 479001600L;
        case 13: 	return 6227020800L;
        case 14: 	return 87178291200L;
        case 15: 	return 1307674368000L;
        case 16: 	return 20922789888000L;
        case 17: 	return 355687428096000L;
        case 18: 	return 6402373705728000L;
        case 19: 	return 121645100408832000L;
        case 20: 	return 2432902008176640000L;
        default:    return 0L;
    }
}