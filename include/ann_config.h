#ifndef ANN_CONFIG_H
#define ANN_CONFIG_H

#include <stdatomic.h>

#define MAX_MEMORY_USAGE_RATIO 0.8        // Maximum memory usage ratio
#define MIN_QUERIES_PER_BLOCK 1           // Minimum number of queries per block

static atomic_int ANN_NUM_TREADS = 1;

void ann_set_num_threads(int n);

int ann_get_num_threads(void);

void ann_set_num_threads_cblas(int n);

int get_num_threads_cblas(void);

unsigned long get_available_memory_bytes(void);

#endif // ANN_CONFIG_H