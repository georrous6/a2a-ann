#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <string.h>
#include "ioutil.h"
#include "knn.h"


// Function to set terminal color
void setColor(const char *colorCode) 
{
    printf("%s", colorCode);
}

// ANSI escape codes for text colors
#define DEFAULT        "\033[0m"
#define BOLD_RED       "\033[1;31m"
#define BOLD_GREEN     "\033[1;32m"
#define BOLD_BLUE      "\033[1;34m"

#define THREAD_CASES 7   // Number of thread cases to benchmark


int knn_benchmark(const char *filename, int *nthreads, int *cblas_nthreads, const char *output_file)
{
    setColor(BOLD_BLUE);
    printf("Running KNN Benchmark %s ...       ", filename);
    setColor(DEFAULT);
    int M, N, L;
    float *C = NULL, *Q = NULL, *my_D = NULL;
    int *test_IDX = NULL, *my_IDX = NULL;
    int status = EXIT_FAILURE;
    struct timeval tstart, tend;

    float execution_time[THREAD_CASES];
    float recall[THREAD_CASES];
    float queries_per_sec[THREAD_CASES];

    // load corpus matrix from file
    C = (float *)load_matrix(filename, "/train", &N, &L); if (!C) goto cleanup;

    // load queries matrix from file
    Q = (float *)load_matrix(filename, "/test", &M, &L); if (!Q) goto cleanup;

    // load expected indices matrix from file
    int a, b;
    test_IDX = (int *)load_matrix(filename, "/neighbors", &a, &b); if (!test_IDX) goto cleanup;
    const int K = b;

    // memory allocation for the estimated distance matrix
    my_D = (float *)malloc(M * K * sizeof(float)); if (!my_D) goto cleanup;

    // memory allocation for the estimated index matrix
    my_IDX = (int *)malloc(M * K * sizeof(int)); if (!my_IDX) goto cleanup;


    for (int t = 0; t < THREAD_CASES; t++) {
        knn_set_num_threads(nthreads[t]);
        knn_set_num_threads_cblas(cblas_nthreads[t]);
        gettimeofday(&tstart, NULL);
        if (knnsearch(Q, C, my_IDX, my_D, M, N, L, K, 1)) goto cleanup;
        gettimeofday(&tend, NULL);
        long execution_time_usec = (tend.tv_sec - tstart.tv_sec) * 1000000L + (tend.tv_usec - tstart.tv_usec);
        execution_time[t] = execution_time_usec / 1e6f;  // Convert to seconds
        queries_per_sec[t] = ((float) M) / execution_time[t];

        int found = 0;

        for (int i = 0; i < M; i++)
        {
            for (int j = 0; j < K; j++)
            {
                const int index = test_IDX[i * K + j];
                for (int k = 0; k < K; k++)
                {
                    if (index == my_IDX[i * K + k])
                    {
                        found++;
                        break;
                    }
                }
            }
        }

        recall[t] = ((float) found) / (M * K) * 100.0;

        printf("\n\n===================\n");
        printf("KNN Benchmark\n");
        printf("Number of threads: %d\n", nthreads[t]);
        printf("Execution time: %f sec\n", execution_time[t]);
        printf("Recall: %.4f%%\n", recall[t]);
        printf("Queries per sec: %.4f\n", queries_per_sec[t]);
    }

    store_matrix(nthreads, "nthreads", 1, THREAD_CASES, output_file, INT_TYPE, 'w');
    store_matrix(queries_per_sec, "queries_per_sec", 1, THREAD_CASES, output_file, FLOAT_TYPE, 'a');
    store_matrix(recall, "recall", 1, THREAD_CASES, output_file, FLOAT_TYPE, 'a');

    status = EXIT_SUCCESS;

cleanup:
    if (C) free(C);
    if (Q) free(Q);
    if (my_D) free(my_D);
    if (my_IDX) free(my_IDX);
    if (test_IDX) free(test_IDX);

    return status;
}


int main(int argc, char *argv[])
{
    if (argc < 3) 
    {
        fprintf(stderr, "Usage: %s <dataset> <benchmark_output>\n", argv[0]);
        return EXIT_FAILURE;
    }

    int knn_nthreads[THREAD_CASES] = {1, 1, 2, 4, 8, 16, 32};
    int cblas_nthreads[THREAD_CASES] = {-1, 1, 1, 1, 1, 1, 1}; // OpenBLAS threads

    if (knn_benchmark(argv[1], knn_nthreads, cblas_nthreads, argv[2])) 
    {
        fprintf(stderr, "KNN Benchmark failed for %s.\n", argv[1]);
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}