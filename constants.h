#ifndef ANABRUTE_CONSTANTS_H
#define ANABRUTE_CONSTANTS_H

// defines task size for gpu cruncher
// peak at ~256-512K
#define PERMUT_TASKS_IN_BATCH 256*1024
// peak at ~512
#define MAX_ITERS_PER_TASK 512
// 512*512*1024 == 256M (11! < 256M < 12!)

// has to be in sync with constants in permut.cl
#define MAX_STR_LENGTH 40
#define MAX_OFFSETS_LENGTH 20


#endif //ANABRUTE_CONSTANTS_H
