#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <string.h>
#include "ioutil.h"
#include "a2a_knn.h"


#define NUM_THREADS 4
#define MAX_MEMORY_USAGE_RATIO 0.1


int main(int argc, char *argv[])
{
    int M, N, L, K;
    float *train = NULL, *test = NULL, *my_distances = NULL;
    int *neighbors = NULL, *my_neighbors = NULL;
    int status = EXIT_FAILURE;
    struct timeval tstart, tend;
    int aa, bb, cc, dd;

    if (argc != 2) {
        fprintf(stderr, "Usage: %s <filename>\n", argv[0]);
        return EXIT_FAILURE;
    }

    const char* filename = argv[1];

    float execution_time;
    float recall;
    float queries_per_sec;

    // load corpus matrix from file
    train = (float *)load_hdf5(filename, "/train", &N, &aa); if (!train) goto cleanup;

    // load queries matrix from file
    test = (float *)load_hdf5(filename, "/test", &M, &bb); if (!test) goto cleanup;

    if (aa != bb) {
        fprintf(stderr, "Inconsistent dimensions for train and test matrices: %d vs %d\n", aa, bb);
        goto cleanup;
    }
    L = aa;

    // load expected indices matrix from file
    neighbors = (int *)load_hdf5(filename, "/neighbors", &aa, &K); if (!neighbors) goto cleanup;

    if (aa != M) {
        fprintf(stderr, "Inconsistent train and neighbors dimensions: %d vs %d\n", aa, M);
        goto cleanup;
    }

    // memory allocation for the estimated distance matrix
    my_distances = (float *)malloc(M * K * sizeof(float)); if (!my_distances) goto cleanup;

    // memory allocation for the estimated index matrix
    my_neighbors = (int *)malloc(M * K * sizeof(int)); if (!my_neighbors) goto cleanup;

    gettimeofday(&tstart, NULL);
    if (a2a_knnsearch(test, train, my_neighbors, my_distances, M, N, L, K, 1, 
        NUM_THREADS, 1, MAX_MEMORY_USAGE_RATIO)) goto cleanup;
    gettimeofday(&tend, NULL);
    long execution_time_usec = (tend.tv_sec - tstart.tv_sec) * 1000000L + (tend.tv_usec - tstart.tv_usec);
    execution_time = (float)execution_time_usec / 1e6f;
    queries_per_sec = ((float) M) / execution_time;

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

    recall = ((float) found) / (M * K) * 100.0f;

    printf("\n\n===================\n");
    printf("Number of threads: %d\n", NUM_THREADS);
    printf("Execution time: %.4f sec\n", execution_time);
    printf("Recall: %.4f%%\n", recall);
    printf("Queries per sec: %.4f\n", queries_per_sec);

    status = EXIT_SUCCESS;

cleanup:
    if (train) free(train);
    if (test) free(test);
    if (my_distances) free(my_distances);
    if (my_neighbors) free(my_neighbors);
    if (neighbors) free(neighbors);

    return status;
}