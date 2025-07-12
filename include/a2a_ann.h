#ifndef A2A_ANN_H
#define A2A_ANN_H

#include "ann_config.h"
#include <stdatomic.h>

static atomic_int ANN_NUM_THREADS = 1;


void ann_set_num_threads(int num_threads);

int ann_get_num_threads(void);

int a2a_annsearch(const DTYPE* C, const int N, const int L, const int K, 
    const int Kc, int* IDX, DTYPE* D, const int max_iter);


#endif // A2A_ANN_H