#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include <string.h>
#include "ioutil.h"
#include "knnsearch.h"


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

#define THREAD_CASES 5  // Number of thread cases to benchmark


int benchmark(int (*func)(const double*, const double*, int*, double*, const int, const int, const int, int, const int, int), const char *filename, double *recall, double *queries_per_sec, long *execution_time, const int nthreads)
{
    setColor(BOLD_BLUE);
    printf("Running test %s ...       ", filename);
    setColor(DEFAULT);
    int M, N, L;
    double *C = NULL, *Q = NULL, *my_D = NULL;
    int *test_IDX = NULL, *my_IDX = NULL;
    int status = EXIT_FAILURE;
    struct timeval tstart, tend;

    // load corpus matrix from file
    C = (double *)load_matrix(filename, "/train", &N, &L); if (!C) goto cleanup;

    // load queries matrix from file
    Q = (double *)load_matrix(filename, "/test", &M, &L); if (!Q) goto cleanup;

    // load expected indices matrix from file
    int a, b;
    test_IDX = (int *)load_matrix(filename, "/neighbors", &a, &b); if (!test_IDX) goto cleanup;
    const int K = b;

    // memory allocation for the estimated distance matrix
    my_D = (double *)malloc(M * K * sizeof(double)); if (!my_D) goto cleanup;

    // memory allocation for the estimated index matrix
    my_IDX = (int *)malloc(M * K * sizeof(int)); if (!my_IDX) goto cleanup;

    gettimeofday(&tstart, NULL);
    if (func(Q, C, my_IDX, my_D, M, N, L, K, 1, nthreads)) goto cleanup;
    gettimeofday(&tend, NULL);
    *execution_time = tend.tv_sec - tstart.tv_sec;
    *queries_per_sec = ((double) M) / *execution_time;

    status = EXIT_SUCCESS;
    int found = 0;

    for (int i = 0; i < M; i++)
    {
        for (int j = 0; j < K; j++)
        {
            const int index = test_IDX[i * K + j];
            for (int t = 0; t < K; t++)
            {
                if (index == my_IDX[i * K + t])
                {
                    found++;
                    break;
                }
            }
        }
    }

    *recall = ((double) found) / (M * K) * 100.0;

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

    double recall[THREAD_CASES];
    double queries_per_sec[THREAD_CASES];
    int nthreads[THREAD_CASES] = {1, 2, 4, 8, 16};
    long execution_time[THREAD_CASES];

    for (int i = 0; i < THREAD_CASES; i++)
    {
        if (benchmark(knnsearch, argv[1], &recall[i], &queries_per_sec[i], &execution_time[i], nthreads[i]) == EXIT_SUCCESS)
        {
            printf("\n===================\n");
            printf("Algorithm: Exact k-NN\n");
            printf("Number of threads: %d\n", nthreads[i]);
            printf("Execution time: %ld sec\n", execution_time[i]);
            printf("Recall: %.4lf%%\n", recall[i]);
            printf("Queries per sec: %.4lf\n", queries_per_sec[i]);
        }
        else
        {
            return EXIT_FAILURE;
        }
    }

    store_matrix(queries_per_sec, "queries_per_sec", 1, THREAD_CASES, argv[2], DOUBLE_TYPE, 'w');
    store_matrix(nthreads, "nthreads", 1, THREAD_CASES, argv[2], INT_TYPE, 'a');

    return EXIT_SUCCESS;
}