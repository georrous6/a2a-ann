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


int test_case(const char *filename, double tolerance, int *passed)
{
    setColor(BOLD_BLUE);
    printf("Running test %s ...       ", filename);
    setColor(DEFAULT);
    int M, N, L;
    double *C = NULL, *Q = NULL, *my_D = NULL, *test_D = NULL;
    int *K = NULL, *test_IDX = NULL, *my_IDX = NULL;
    int status = EXIT_FAILURE;
    *passed = 0;

    // load corpus matrix from file
    C = (double *)load_matrix(filename, "C", &N, &L); if (!C) goto cleanup;

    // load queries matrix from file
    Q = (double *)load_matrix(filename, "Q", &M, &L); if (!Q) goto cleanup;

    int a, b;
    // load K value from file
    K = (int *)load_matrix(filename, "K", &a, &b); if (!K) goto cleanup;

    // load expected distances matrix from file
    test_D = (double *)load_matrix(filename, "test_D", &a, &b); if (!test_D) goto cleanup;

    // load expected indices matrix from file
    test_IDX = (int *)load_matrix(filename, "test_IDX", &a, &b); if (!test_IDX) goto cleanup;

    // memory allocation for the estimated distance matrix
    my_D = (double *)malloc(M * (*K) * sizeof(double)); if (!my_D) goto cleanup;

    // memory allocation for the estimated index matrix
    my_IDX = (int *)malloc(M * (*K) * sizeof(int)); if (!my_IDX) goto cleanup;

    knn_set_num_threads(-1);
    if (knnsearch(Q, C, my_IDX, my_D, M, N, L, *K, 1)) goto cleanup;

    status = EXIT_SUCCESS;

    // test the output with the estimated one
    double x, y;
    for (int i = 0; i < M; i++)
    {
        for (int j = 0; j < *K; j++)
        {
            x = test_D[i * (*K) + j];
            y = my_D[i * (*K) + j];
            if (fabs(x - y) >= tolerance)
            {
                printf("Assertion %lf == %lf ", x, y);
                goto cleanup;
            }
        }
    }

    for (int i = 0; i < M; i++)
    {
        for (int j = 0; j < *K; j++)
        {
            if (test_IDX[i * (*K) + j] != my_IDX[i * (*K) + j])
            {
                printf("Assertion %d == %d ", test_IDX[i * (*K) + j], my_IDX[i * (*K) + j]);
                goto cleanup;
            }
        }
    }

    *passed = 1; 

cleanup:
    if (C) free(C);
    if (Q) free(Q);
    if (K) free(K);
    if (my_D) free(my_D);
    if (my_IDX) free(my_IDX);
    if (test_D) free(test_D);
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
    int passed;
    char **file_paths = get_file_paths(argv[1], ".mat", &test_cnt, 1);
    if (!file_paths)
    {
        return EXIT_FAILURE;
    }

    for (size_t i = 0; i < test_cnt; i++) 
    {
        if (test_case(file_paths[i], 1e-6, &passed) == EXIT_SUCCESS)
        {
            if (passed)
            {
                setColor(BOLD_GREEN);
                printf("Passed\n");
                cnt_passed++;
                setColor(DEFAULT);
            }
            else
            {
                setColor(BOLD_RED);
                printf("Failed\n");
                setColor(DEFAULT);
            }
        }
        else
        {
            fprintf(stderr, "An error occured\n");
            break;
        }
    }

    if (cnt_passed == test_cnt)
    {
        setColor(BOLD_GREEN);
        printf("\n==========================\n");
        printf("All tests passed (%zu/%zu)\n", cnt_passed, test_cnt);
        setColor(DEFAULT);
    }
    else
    {
        printf("\n==========================\n");
        printf("Tests passed: %zu/%zu\n", cnt_passed, test_cnt);
    }


    // free allocated memory
    for (size_t i = 0; i < test_cnt; i++)
    {
        free(file_paths[i]);
    }
    free(file_paths);

    return EXIT_SUCCESS;
}