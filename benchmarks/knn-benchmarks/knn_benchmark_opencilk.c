#include <stdio.h>
#include <stdlib.h>
#include "knn_benchmark.h"


int main(int argc, char *argv[])
{
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <dataset> <benchmark_output>\n", argv[0]);
        return EXIT_FAILURE;
    }

    int nthreads[THREAD_CASES] = {1, 1, 2, 4, 8, 16, 32};
    int cblas_threads[THREAD_CASES] = {4, 1, 1, 1, 1, 1, 1};

    const parallelization_type_t par_type = PAR_OPENCILK;
    if (knn_benchmark(argv[1], nthreads, cblas_threads, argv[2], par_type)) {
        fprintf(stderr, "KNN Benchmark failed for parallelization mode %d.\n", par_type);
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}