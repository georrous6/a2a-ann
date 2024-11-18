#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "ioutil.h"
#include "knnsearch_approx.h"


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


int test_case(const char *filename, double *recall, double *queries_per_sec)
{
    setColor(BOLD_BLUE);
    printf("Running test %s ...       ", filename);
    setColor(DEFAULT);
    int M, N, L;
    double *C = NULL, *Q = NULL, *my_D = NULL;
    int *test_IDX = NULL, *my_IDX = NULL;
    int status = EXIT_FAILURE;

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

    clock_t start = clock();
    if (knnsearch_approx(Q, C, my_IDX, my_D, M, N, L, K, 0, -1)) goto cleanup;
    clock_t end = clock();
    *queries_per_sec = M / (((double)(end - start)) / CLOCKS_PER_SEC);

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

    // print_matrix(test_IDX, "test_IDX", M, *K, INT_TYPE);
    // print_matrix(my_IDX, "my_IDX", M, *K, INT_TYPE);
    *recall = ((double)found) / (M * K) * 100.0;

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
    if (argc < 2) 
    {
        fprintf(stderr, "Usage: %s <directory_path>\n", argv[0]);
        return EXIT_FAILURE;
    }


    size_t cnt_passed = 0;
    size_t test_cnt;
    int found, total;
    double recall, queries_per_sec, recall_avg = 0, queries_per_sec_avg = 0;
    char **file_paths = get_file_paths(argv[1], ".hdf5", &test_cnt, 1);
    if (!file_paths)
    {
        return EXIT_FAILURE;
    }

    for (size_t i = 0; i < test_cnt; i++) 
    {
        if (test_case(file_paths[i], &recall, &queries_per_sec) == EXIT_SUCCESS)
        {
            printf("Recall: %.4lf%% Queries per sec: %.4lf\n", recall, queries_per_sec);
            recall_avg += recall;
            queries_per_sec_avg += queries_per_sec;
        }
        else
        {
            fprintf(stderr, "An error occured\n");
            break;
        }
    }

    recall_avg /= test_cnt;
    queries_per_sec_avg /= test_cnt;
    printf("\n=======================\n");
    printf("Average Recall: %.4lf%%\n", recall_avg);
    printf("Average Queries per sec %.4lf\n", queries_per_sec_avg);

    // free allocated memory
    for (size_t i = 0; i < test_cnt; i++)
    {
        free(file_paths[i]);
    }
    free(file_paths);

    return EXIT_SUCCESS;
}