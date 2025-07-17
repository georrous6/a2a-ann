#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <string.h>
#include "ioutil.h"
#include "ann_benchmark.h"


// Function to set terminal color
void setColor(const char *colorCode) {
    printf("%s", colorCode);
}


int ann_benchmark(const char *filename, int *nthreads, int *num_clusters, 
    const char *output_file, parallelization_type_t parallelization_mode) {

    setColor(BOLD_BLUE);
    printf("\nOpening %s ...\n", filename);
    setColor(DEFAULT);
    float *train = NULL, *test = NULL, *train_test = NULL, *my_all_to_all_distances = NULL;
    int *all_to_all_neighbors = NULL, *my_all_to_all_neighbors = NULL;
    int status = EXIT_FAILURE;
    struct timeval tstart, tend;
    int aa, bb, cc, dd;

    float execution_time[THREAD_CASES][CLUSTER_CASES];
    float recall[THREAD_CASES][CLUSTER_CASES];
    float queries_per_sec[THREAD_CASES][CLUSTER_CASES];

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

    if (store_hdf5(nthreads, "nthreads", 1, THREAD_CASES, output_file, INT_TYPE, 'a')) {
        fprintf(stderr, "Error storing nthreads data.\n");
        goto cleanup;
    }

    if (store_hdf5(num_clusters, "num_clusters", 1, CLUSTER_CASES, output_file, INT_TYPE, 'a')) {
        fprintf(stderr, "Error storing num_clusters data.\n");
        goto cleanup;
    }

    for (int t = 0; t < THREAD_CASES; t++) {
        for (int c = 0; c < CLUSTER_CASES; c++) {

            setColor(BOLD_BLUE);
            printf("\nRunning ANN benchmark for %d threads and %d clusters...\n", nthreads[t], num_clusters[c]);
            setColor(DEFAULT);
            
            gettimeofday(&tstart, NULL);
            if (a2a_annsearch(train_test, N, L, K, num_clusters[c], my_all_to_all_neighbors, 
                my_all_to_all_distances, nthreads[t], MAX_MEMORY_USAGE_RATIO, parallelization_mode)) goto cleanup;
            gettimeofday(&tend, NULL);
            long execution_time_usec = (tend.tv_sec - tstart.tv_sec) * 1000000L + (tend.tv_usec - tstart.tv_usec);
            execution_time[t][c] = execution_time_usec / 1e6f;  // Convert to seconds
            queries_per_sec[t][c] = ((float) N) / execution_time[t][c];

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

            recall[t][c] = ((float) found) / (N * K) * 100.0;

            printf("\n\n===================\n");
            printf("ANN Benchmark\n");
            printf("Parallelization mode: %s\n", parallelization_mode == PAR_PTHREADS ? "pthreads" :
                   (parallelization_mode == PAR_OPENMP ? "openmp" : "opencilk"));
            printf("Number of threads: %d\n", nthreads[t]);
            printf("Number of clusters: %d\n", num_clusters[c]);
            printf("Execution time: %f sec\n", execution_time[t][c]);
            printf("Recall: %.4f%%\n", recall[t][c]);
            printf("Queries per sec: %.4f\n", queries_per_sec[t][c]);

            if (store_hdf5(queries_per_sec, "queries_per_sec", t + 1, c + 1, output_file, FLOAT_TYPE, 'a')) {
                fprintf(stderr, "Error storing queries_per_sec data.\n");
                goto cleanup;
            }
            if (store_hdf5(recall, "recall", t + 1, c + 1, output_file, FLOAT_TYPE, 'a')) {
                fprintf(stderr, "Error storing recall data.\n");
                goto cleanup;
            }
        }
    }

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



int ann_recall_vs_throughput(const char *filename, int nthreads, int *num_clusters, const char *output_file, 
    parallelization_type_t parallelization_mode) {

    setColor(BOLD_BLUE);
    printf("\nOpening %s ...\n", filename);
    setColor(DEFAULT);
    float *train = NULL, *test = NULL, *train_test = NULL, *my_all_to_all_distances = NULL;
    int *all_to_all_neighbors = NULL, *my_all_to_all_neighbors = NULL;
    int status = EXIT_FAILURE;
    struct timeval tstart, tend;
    int aa, bb, cc, dd;

    float execution_time[CLUSTER_CASES];
    float recall[CLUSTER_CASES];
    float queries_per_sec[CLUSTER_CASES];

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

    for (int c = 0; c < CLUSTER_CASES; c++) {
        setColor(BOLD_BLUE);
        printf("\nRunning recall vs throughput  benchmark for %d clusters (parallelization mode: %s) ...\n", num_clusters[c], suffix);
        setColor(DEFAULT);
        
        gettimeofday(&tstart, NULL);
        if (a2a_annsearch(train_test, N, L, K, num_clusters[c], my_all_to_all_neighbors, 
            my_all_to_all_distances, nthreads, MAX_MEMORY_USAGE_RATIO, parallelization_mode)) goto cleanup;
        gettimeofday(&tend, NULL);
        long execution_time_usec = (tend.tv_sec - tstart.tv_sec) * 1000000L + (tend.tv_usec - tstart.tv_usec);
        execution_time[c] = execution_time_usec / 1e6f;  // Convert to seconds
        queries_per_sec[c] = ((float) N) / execution_time[c];

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

        recall[c] = ((float) found) / (N * K) * 100.0;

        printf("\n\n===================\n");
        printf("ANN Benchmark\n");
        printf("Number of threads: %d\n", nthreads);
        printf("Number of clusters: %d\n", num_clusters[c]);
        printf("Execution time: %f sec\n", execution_time[c]);
        printf("Recall: %.4f%%\n", recall[c]);
        printf("Queries per sec: %.4f\n", queries_per_sec[c]);

        // Save data on every iteration
        if (store_hdf5(queries_per_sec, qps_name, 1, c + 1, output_file, FLOAT_TYPE, 'a')) {
            fprintf(stderr, "Error storing queries_per_sec data.\n");
            goto cleanup;
        }
        if (store_hdf5(recall, recall_name, 1, c + 1, output_file, FLOAT_TYPE, 'a')) {
            fprintf(stderr, "Error storing recall data.\n");
            goto cleanup;
        }
    }

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
