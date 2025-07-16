#ifndef KNN_BENCHMARK_H
#define KNN_BENCHMARK_H
#include "a2a_knn.h"


// Function to set terminal color
void setColor(const char *colorCode);

// ANSI escape codes for text colors
#define DEFAULT        "\033[0m"
#define BOLD_RED       "\033[1;31m"
#define BOLD_GREEN     "\033[1;32m"
#define BOLD_BLUE      "\033[1;34m"

#define THREAD_CASES 7   // Number of thread cases to benchmark
#define MAX_MEMORY_USAGE_RATIO 0.5


int knn_benchmark(const char *filename, int *nthreads, int *cblas_threads, 
    const char *output_file, parallelization_type_t parallelization_mode);

#endif // KNN_BENCHMARK_H
