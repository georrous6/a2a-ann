#include <stdio.h>
#include <stdlib.h>
#include <math.h>
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

#define TOLERANCE 1e-6


int test_case(const char *filename, double tolerance)
{
    setColor(BOLD_BLUE);
    printf("Running test %s ...       ", filename);
    setColor(DEFAULT);
    int M, N, L, K;
    double *train = NULL, *test = NULL, *my_distances = NULL, *distances = NULL;
    int *neighbors = NULL, *my_neighbors = NULL;
    int status = EXIT_FAILURE;
    int aa, bb, cc, dd;

    // load corpus matrix from file
    train = (double *)load_hdf5(filename, "/train", &N, &aa); if (!train) goto cleanup;

    // load queries matrix from file
    test = (double *)load_hdf5(filename, "/test", &M, &bb); if (!test) goto cleanup;

    if (aa != bb) {
        fprintf(stderr, "Inconsistent number of columns for train and test matrices\n");
        goto cleanup;
    }
    L = aa;

    // load expected distances matrix from file
    distances = (double *)load_hdf5(filename, "/distances", &aa, &bb); if (!distances) goto cleanup;

    // load expected indices matrix from file
    neighbors = (int *)load_hdf5(filename, "/neighbors", &cc, &dd); if (!neighbors) goto cleanup;

    if (aa != M || cc != M || bb != dd) {
        fprintf(stderr, "Inconsistent dimensions for distances or neighbors matrices\n");
        goto cleanup;
    }
    K = bb;

    // memory allocation for the estimated distance matrix
    my_distances = (double *)malloc(M * K * sizeof(double)); if (!my_distances) goto cleanup;

    // memory allocation for the estimated index matrix
    my_neighbors = (int *)malloc(M * K * sizeof(int)); if (!my_neighbors) goto cleanup;

    knn_set_num_threads(-1);
    if (knnsearch(test, train, my_neighbors, my_distances, M, N, L, K, 1)) goto cleanup;

    // test the output with the estimated one
    double x, y;
    for (int i = 0; i < M; i++)
    {
        for (int j = 0; j < K; j++)
        {
            x = distances[i * K + j];
            y = my_distances[i * K + j];
            if (fabs(x - y) >= tolerance)
            {
                printf("Assertion %lf == %lf ", x, y);
                goto cleanup;
            }
        }
    }

    for (int i = 0; i < M; i++)
    {
        for (int j = 0; j < K; j++)
        {
            if (neighbors[i * K + j] != my_neighbors[i * K + j])
            {
                printf("Assertion %d == %d ", neighbors[i * K + j], my_neighbors[i * K + j]);
                goto cleanup;
            }
        }
    }

    status = EXIT_SUCCESS;

cleanup:
    if (train) free(train);
    if (test) free(test);
    if (my_distances) free(my_distances);
    if (my_neighbors) free(my_neighbors);
    if (distances) free(distances);
    if (neighbors) free(neighbors);

    return status;
}


int main(int argc, char *argv[])
{
    if (argc < 2) 
    {
        fprintf(stderr, "Usage: %s <directory_path>\n", argv[0]);
        return EXIT_FAILURE;
    }


    int status = EXIT_FAILURE;
    size_t cnt_passed = 0;
    size_t test_cnt;
    char **file_paths = get_file_paths(argv[1], ".hdf5", &test_cnt, 1);
    if (!file_paths)
    {
        return EXIT_FAILURE;
    }

    for (size_t i = 0; i < test_cnt; i++) {

        if (test_case(file_paths[i], TOLERANCE) == EXIT_SUCCESS) {
            setColor(BOLD_GREEN);
            printf("Passed\n");
            cnt_passed++;
            setColor(DEFAULT);
        }
        else {
            setColor(BOLD_RED);
            printf("Failed\n");
            setColor(DEFAULT);
        }
    }

    if (cnt_passed == test_cnt) {
        setColor(BOLD_GREEN);
        printf("\n==========================\n");
        printf("All tests passed (%zu/%zu)\n", cnt_passed, test_cnt);
        setColor(DEFAULT);
        status = EXIT_SUCCESS;
    }
    else {
        printf("\n==========================\n");
        printf("Tests passed: %zu/%zu\n", cnt_passed, test_cnt);
    }


    // free allocated memory
    for (size_t i = 0; i < test_cnt; i++) {
        free(file_paths[i]);
    }
    free(file_paths);

    return status;
}