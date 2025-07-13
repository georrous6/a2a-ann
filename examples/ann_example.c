#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <string.h>
#include <time.h>
#include "ioutil.h"
#include "a2a_ann.h"

#define NUM_THREADS 4    // Number of threads
#define NUM_CLUSTERS 5  // Number of clusters


int main(int argc, char *argv[])
{
    if (argc != 2) 
    {
        fprintf(stderr, "Usage: %s <filename>\n", argv[0]);
        return EXIT_FAILURE;
    }

    srand(time(NULL));

    const char *filename = argv[1];

    float *train = NULL, *test = NULL, *train_test = NULL, *my_all_to_all_distances = NULL;
    int *all_to_all_neighbors = NULL, *my_all_to_all_neighbors = NULL;
    int status = EXIT_FAILURE;
    struct timeval tstart, tend;
    const int max_iter = 1;
    int aa, bb, cc, dd;

    float execution_time;
    float recall;
    float queries_per_sec;

    train = (float *)load_hdf5(filename, "/train", &aa, &bb); if (!train) goto cleanup;

    test = (float *)load_hdf5(filename, "/test", &cc, &dd); if (!test) goto cleanup;

    if (bb != dd) {
        fprintf(stderr, "Inconsistent number of columns for train and test matrices\n");
        goto cleanup;
    }

    // Concatenate train and test matrices
    const int N = aa + cc;
    const int L = bb;
    train_test = (float *)malloc(N * L * sizeof(float)); if (!train_test) goto cleanup;
    memcpy(train_test, train, aa * L * sizeof(float));
    memcpy(train_test + aa * L, test, cc * L * sizeof(float));

    // load expected indices matrix from file
    all_to_all_neighbors = (int *)load_hdf5(filename, "/all_to_all_neighbors", &aa, &bb); if (!all_to_all_neighbors) goto cleanup;

    if (aa != N) {
        fprintf(stderr, "Inconsistent dimensions for neighbors matrix\n");
        goto cleanup;
    }
    const int K = bb;

    // memory allocation for the estimated distance matrix
    my_all_to_all_distances = (float *)malloc(N * K * sizeof(float)); if (!my_all_to_all_distances) goto cleanup;

    // memory allocation for the estimated index matrix
    my_all_to_all_neighbors = (int *)malloc(N * K * sizeof(int)); if (!my_all_to_all_neighbors) goto cleanup;


    ann_set_num_threads(NUM_THREADS);
    gettimeofday(&tstart, NULL);
    if (a2a_annsearch(train_test, N, L, K, NUM_CLUSTERS, my_all_to_all_neighbors, my_all_to_all_distances, max_iter)) goto cleanup;
    gettimeofday(&tend, NULL);
    long execution_time_usec = (tend.tv_sec - tstart.tv_sec) * 1000000L + (tend.tv_usec - tstart.tv_usec);
    execution_time = execution_time_usec / 1e6f;  // Convert to seconds
    queries_per_sec = ((float) N) / execution_time;

    int found = 0;

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < K; j++) {
            const int index = all_to_all_neighbors[i * K + j];
            for (int k = 0; k < K; k++) {
                if (index == my_all_to_all_neighbors[i * K + k]) {
                    found++;
                    break;
                }
            }
        }
    }

    recall = ((float) found) / (N * K) * 100.0;

    printf("\n\n===================\n");
    printf("Number of threads: %d\n", NUM_THREADS);
    printf("Number of clusters: %d\n", NUM_CLUSTERS);
    printf("Execution time: %f sec\n", execution_time);
    printf("Recall: %.4f%%\n", recall);
    printf("Queries per sec: %.4f\n", queries_per_sec);

    status = EXIT_SUCCESS;

cleanup:
    if (train) free(train);
    if (test) free(test);
    if (train_test) free(train_test);
    if (my_all_to_all_distances) free(my_all_to_all_distances);
    if (my_all_to_all_neighbors) free(my_all_to_all_neighbors);
    if (all_to_all_neighbors) free(all_to_all_neighbors);


    return status;
}