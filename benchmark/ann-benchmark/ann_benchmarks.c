#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <string.h>
#include "ioutil.h"
#include "a2a_ann.h"


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

#define THREAD_CASES 6   // Number of thread cases to benchmark
#define CLUSTER_CASES 5  // Number of cluster cases to benchmark


int ann_benchmark(const char *filename, int *nthreads, int *num_clusters, const char *output_file)
{
    setColor(BOLD_BLUE);
    printf("Running ANN Benchmark %s ...       ", filename);
    setColor(DEFAULT);
    int N, L;
    float *C = NULL, *my_D = NULL;
    int *test_IDX = NULL, *my_IDX = NULL;
    int status = EXIT_FAILURE;
    struct timeval tstart, tend;
    const int max_iter = 10;

    float execution_time[THREAD_CASES][CLUSTER_CASES];
    float recall[THREAD_CASES][CLUSTER_CASES];
    float queries_per_sec[THREAD_CASES][CLUSTER_CASES];

    // load corpus matrix from file
    C = (float *)load_hdf5(filename, "/train_test", &N, &L); if (!C) goto cleanup;

    // load expected indices matrix from file
    int a, b;
    test_IDX = (int *)load_hdf5(filename, "/train_test_neighbors", &a, &b); if (!test_IDX) goto cleanup;
    const int K = b;

    // memory allocation for the estimated distance matrix
    my_D = (float *)malloc(N * K * sizeof(float)); if (!my_D) goto cleanup;

    // memory allocation for the estimated index matrix
    my_IDX = (int *)malloc(N * K * sizeof(int)); if (!my_IDX) goto cleanup;


    for (int t = 0; t < THREAD_CASES; t++) {
        for (int c = 0; c < CLUSTER_CASES; c++) {
            ann_set_num_threads(nthreads[t]);
            gettimeofday(&tstart, NULL);
            if (a2a_annsearch(C, N, L, K, num_clusters[c], my_IDX, my_D, max_iter)) goto cleanup;
            gettimeofday(&tend, NULL);
            long execution_time_usec = (tend.tv_sec - tstart.tv_sec) * 1000000L + (tend.tv_usec - tstart.tv_usec);
            execution_time[t][c] = execution_time_usec / 1e6f;  // Convert to seconds
            queries_per_sec[t][c] = ((float) N) / execution_time[t][c];

            int found = 0;

            for (int i = 0; i < N; i++)
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

            recall[t][c] = ((float) found) / (N * K) * 100.0;

            printf("\n\n===================\n");
            printf("ANN Benchmark\n");
            printf("Number of threads: %d\n", nthreads[t]);
            printf("Number of clusters: %d\n", num_clusters[c]);
            printf("Execution time: %f sec\n", execution_time[t][c]);
            printf("Recall: %.4f%%\n", recall[t][c]);
            printf("Queries per sec: %.4f\n", queries_per_sec[t][c]);
        }
    }

    store_hdf5(nthreads, "nthreads", 1, THREAD_CASES, output_file, INT_TYPE, 'w');
    store_hdf5(num_clusters, "num_clusters", 1, CLUSTER_CASES, output_file, INT_TYPE, 'a');
    store_hdf5(queries_per_sec, "queries_per_sec", THREAD_CASES, CLUSTER_CASES, output_file, FLOAT_TYPE, 'a');
    store_hdf5(recall, "recall", THREAD_CASES, CLUSTER_CASES, output_file, FLOAT_TYPE, 'a');

    status = EXIT_SUCCESS;

cleanup:
    if (C) free(C);
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

    int ann_nthreads[THREAD_CASES] = {1, 2, 4, 8, 16, 32};
    int num_clusters[CLUSTER_CASES] = {10, 20, 50, 100, 200};

    if (ann_benchmark(argv[1], ann_nthreads, num_clusters, argv[2])) 
    {
        fprintf(stderr, "A2A Benchmark failed for %s.\n", argv[1]);
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}