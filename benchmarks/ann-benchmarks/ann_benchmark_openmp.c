#include <stdio.h>
#include <stdlib.h>
#include "ann_benchmark.h"


int main(int argc, char *argv[]) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <dataset> <benchmark_output>\n", argv[0]);
        return EXIT_FAILURE;
    }

    int ann_nthreads[THREAD_CASES] = {1, 2, 4, 8, 16};
    int num_clusters[CLUSTER_CASES] = {5, 10, 20, 50, 100};
    parallelization_type_t parallelization_mode = PAR_PTHREADS;

    if (ann_benchmark(argv[1], ann_nthreads, num_clusters, argv[2], parallelization_mode)) {
        fprintf(stderr, "A2A Benchmark failed for %s.\n", argv[1]);
        return EXIT_FAILURE;
    }

    int nthreads = 4;
    if (ann_recall_vs_throughput(argv[1], nthreads, num_clusters, argv[2], parallelization_mode)) {
        fprintf(stderr, "ANN PTHREADS Benchmark failed for %s.\n", argv[1]);
        return EXIT_FAILURE;
    }

    parallelization_mode = PAR_OPENMP;
    
    if (ann_recall_vs_throughput(argv[1], nthreads, num_clusters, argv[2], parallelization_mode)) {
        fprintf(stderr, "ANN OPENMP Benchmark failed for %s.\n", argv[1]);
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}