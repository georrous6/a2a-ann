#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <string.h>
#include "ioutil.h"
#include "a2a_knn.h"


// Function to set terminal color
void setColor(const char *colorCode) {
    printf("%s", colorCode);
}

// ANSI escape codes for text colors
#define DEFAULT        "\033[0m"
#define BOLD_RED       "\033[1;31m"
#define BOLD_GREEN     "\033[1;32m"
#define BOLD_BLUE      "\033[1;34m"

#define THREAD_CASES 7   // Number of thread cases to benchmark
#define MAX_MEMORY_USAGE_RATIO 0.5


int knn_benchmark(const char *filename, int *nthreads, int *cblas_threads, const char *output_file, 
    parallelization_type_t parallelization_mode) {
    setColor(BOLD_BLUE);
    printf("\nOpening %s ...\n", filename);
    setColor(DEFAULT);
    int M, N, L, K;
    float *train = NULL, *test = NULL, *my_distances = NULL;
    int *neighbors = NULL, *my_neighbors = NULL;
    int status = EXIT_FAILURE;
    struct timeval tstart, tend;
    int aa, bb;

    float execution_time[THREAD_CASES];
    float recall[THREAD_CASES];
    float queries_per_sec[THREAD_CASES];

    // load corpus matrix from file
    train = (float *)load_hdf5(filename, "/train", &N, &aa); if (!train) goto cleanup;

    // load queries matrix from file
    test = (float *)load_hdf5(filename, "/test", &M, &bb); if (!test) goto cleanup;

    if (aa != bb) {
        fprintf(stderr, "Error: The number of columns in train (%d) and test (%d) matrices do not match.\n", aa, bb);
        goto cleanup;
    }
    L = aa;

    // load expected indices matrix from file
    neighbors = (int *)load_hdf5(filename, "/neighbors", &aa, &K); if (!neighbors) goto cleanup;
    if (aa != M) {
        fprintf(stderr, "Error: The number of rows in neighbors (%d) does not match the number of queries (%d).\n", aa, M);
        goto cleanup;
    }

    // memory allocation for the estimated distance matrix
    my_distances = (float *)malloc(M * K * sizeof(float)); if (!my_distances) goto cleanup;

    // memory allocation for the estimated index matrix
    my_neighbors = (int *)malloc(M * K * sizeof(int)); if (!my_neighbors) goto cleanup;

    const char *suffix = "";

    switch (parallelization_mode) {
        case PAR_OPENMP:
            suffix = "openmp";
            break;
        case PAR_OPENCILK:
            suffix = "opencilk";
            break;
        default:
            suffix = "pthreads";
            break;
    }

    char qps_name[64], recall_name[64];

    snprintf(qps_name, sizeof(qps_name), "queries_per_sec_%s", suffix);
    snprintf(recall_name, sizeof(recall_name), "recall_%s", suffix);


    for (int t = 0; t < THREAD_CASES; t++) {
        setColor(BOLD_BLUE);
        printf("\nRunning KNN benchmark with %d threads (parallelization mode: %s) ...\n", nthreads[t], suffix);
        setColor(DEFAULT);
        
        gettimeofday(&tstart, NULL);
        if (a2a_knnsearch(test, train, my_neighbors, my_distances, M, N, L, K, 0, nthreads[t], cblas_threads[t],
            MAX_MEMORY_USAGE_RATIO, parallelization_mode)) goto cleanup;
        gettimeofday(&tend, NULL);
        long execution_time_usec = (tend.tv_sec - tstart.tv_sec) * 1000000L + (tend.tv_usec - tstart.tv_usec);
        execution_time[t] = execution_time_usec / 1e6f;  // Convert to seconds
        queries_per_sec[t] = ((float) M) / execution_time[t];

        int found = 0;

        for (int i = 0; i < M; i++) {
            for (int j = 0; j < K; j++) {
                const int index = neighbors[i * K + j];
                for (int k = 0; k < K; k++) {
                    if (index == my_neighbors[i * K + k]) {
                        found++;
                        break;
                    }
                }
            }
        }

        recall[t] = ((float) found) / (M * K) * 100.0f;

        // Save data on every iteration
        if (store_hdf5(nthreads, "nthreads", 1, t + 1, output_file, INT_TYPE, 'a')) {
            fprintf(stderr, "Error storing nthreads data.\n");
            goto cleanup;
        }
        if (store_hdf5(queries_per_sec, qps_name, 1, t + 1, output_file, FLOAT_TYPE, 'a')) {
            fprintf(stderr, "Error storing queries_per_sec data.\n");
            goto cleanup;
        }
        if (store_hdf5(recall, recall_name, 1, t + 1, output_file, FLOAT_TYPE, 'a')) {
            fprintf(stderr, "Error storing recall data.\n");
            goto cleanup;
        }

        printf("\n\n===================\n");
        printf("KNN Benchmark\n");
        printf("Parallelization mode: %s\n", suffix);
        printf("Number of threads: %d\n", nthreads[t]);
        printf("Execution time: %f sec\n", execution_time[t]);
        printf("Recall: %.4f%%\n", recall[t]);
        printf("Queries per sec: %.4f\n", queries_per_sec[t]);

    }

    status = EXIT_SUCCESS;

cleanup:
    if (train) free(train);
    if (test) free(test);
    if (my_distances) free(my_distances);
    if (my_neighbors) free(my_neighbors);
    if (neighbors) free(neighbors);

    return status;
}


int main(int argc, char *argv[])
{
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <dataset> <benchmark_output>\n", argv[0]);
        return EXIT_FAILURE;
    }

    int nthreads[THREAD_CASES] = {1, 1, 2, 4, 8, 16, 32};
    int cblas_threads[THREAD_CASES] = {4, 1, 1, 1, 1, 1, 1};

    parallelization_type_t par_type = PAR_PTHREADS;
    if (knn_benchmark(argv[1], nthreads, cblas_threads, argv[2], par_type)) {
        fprintf(stderr, "KNN Benchmark failed for parallelization mode %d.\n", par_type);
        return EXIT_FAILURE;
    }

    par_type = PAR_OPENMP;
    if (knn_benchmark(argv[1], nthreads, cblas_threads, argv[2], par_type)) {
        fprintf(stderr, "KNN Benchmark failed for parallelization mode %d.\n", par_type);
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}