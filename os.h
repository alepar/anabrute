#ifndef ANABRUTE_OS_H
#define ANABRUTE_OS_H

#include "common.h"

uint32_t num_cpu_cores();
uint64_t current_micros();
void set_thread_high_priority(void);

#endif //ANABRUTE_OS_H
