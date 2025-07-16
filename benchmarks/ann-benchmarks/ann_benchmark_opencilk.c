#include <stdio.h>
#include <stdlib.h>
#include "ann_benchmark.h"


int main(int argc, char *argv[]) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <dataset> <benchmark_output>\n", argv[0]);
        return EXIT_FAILURE;
    }

    int num_clusters[CLUSTER_CASES] = {5, 10, 20, 50, 100};
    const parallelization_type_t parallelization_mode = PAR_OPENCILK;

    int nthreads = 4;
    if (ann_recall_vs_throughput(argv[1], nthreads, num_clusters, argv[2], parallelization_mode)) {
        fprintf(stderr, "ANN OPENCILK Benchmark failed for %s.\n", argv[1]);
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}