#ifndef ANN_BENCHMARK_H
#define ANN_BENCHMARK_H
#include "a2a_ann.h"

// ANSI escape codes for text colors
#define DEFAULT        "\033[0m"
#define BOLD_RED       "\033[1;31m"
#define BOLD_GREEN     "\033[1;32m"
#define BOLD_BLUE      "\033[1;34m"

#define THREAD_CASES 5   // Number of thread cases to benchmark
#define CLUSTER_CASES 5  // Number of cluster cases to benchmark
#define MAX_MEMORY_USAGE_RATIO 0.5


// Function to set terminal color
void setColor(const char *colorCode);


int ann_benchmark(const char *filename, int *nthreads, int *num_clusters, 
    const char *output_file, parallelization_type_t parallelization_mode);


int ann_recall_vs_throughput(const char *filename, int nthreads, int *num_clusters, 
    const char *output_file, parallelization_type_t parallelization_mode);

#endif