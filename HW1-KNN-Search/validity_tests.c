#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "ioutil.h"
#include "knnsearch.h"


int test_case(const char *filename, double tolerance)
{
    printf("Running test %s ...\t\t", filename);
    int M, N, L;
    double *C = NULL, *Q = NULL, *D = NULL, *expD = NULL;
    int *K = NULL, *expIDX = NULL, *IDX = NULL;
    int status = EXIT_FAILURE;

    // load corpus matrix from file
    C = (double *)load_matrix(filename, "C", &N, &L); if (!C) goto cleanup;

    // load queries matrix from file
    Q = (double *)load_matrix(filename, "Q", &M, &L); if (!Q) goto cleanup;

    // load K nearest neighbors value
    int a, b;
    K = (int *)load_matrix(filename, "K", &a, &b); if (!K) goto cleanup;

    // load expected distances matrix from file
    expD = (double *)load_matrix(filename, "D", &a, &b); if (!expD) goto cleanup;

    // load expected indices matrix from file
    expIDX = (int *)load_matrix(filename, "IDX", &a, &b); if (!expIDX) goto cleanup;

    // memory allocation for the estimated distance matrix
    D = (double *)malloc(M * (*K) * sizeof(double)); if (!D) goto cleanup;

    // memory allocation for the estimated index matrix
    IDX = (int *)malloc(M * (*K) * sizeof(int)); if (!IDX) goto cleanup;

    if (knnsearch_exact(Q, C, IDX, D, M, N, L, *K, 0)) goto cleanup;

    // sort each row vector of the distances matrix
    // to check the correctness of the output
    for (int i = 0; i < M; i++)
    {
        qsort_(D + i * (*K), IDX + i * (*K), 0, (*K) - 1);
    }

    // test the output with the estimated one
    double x, y;
    for (int i = 0; i < M; i++)
    {
        for (int j = 0; j < *K; j++)
        {
            x = expD[i * (*K) + j];
            y = D[i * (*K) + j];
            if (fabs(x - y) >= tolerance * MAX(x, y)) goto cleanup;
        }
    }

    for (int i = 0; i < M; i++)
    {
        for (int j = 0; j < *K; j++)
        {
            if (expIDX[i * (*K) + j] != IDX[i * (*K) + j]) goto cleanup;
        }
    }

    status = EXIT_SUCCESS;

    cleanup:
    if (C) free(C);
    if (Q) free(Q);
    if (K) free(K);
    if (D) free(D);
    if (IDX) free(IDX);
    if (expD) free(expD);
    if (expIDX) free(expIDX);

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
    char **file_paths = get_file_paths(argv[1], ".mat", &test_cnt, 1);
    if (!file_paths)
    {
        return EXIT_FAILURE;
    }

    for (size_t i = 0; i < test_cnt; i++) 
    {
        if (test_case(file_paths[i], 1e-6) == EXIT_SUCCESS)
        {
            printf("Passed\n");
            cnt_passed++;
        }
        else
        {
            printf("Failed\n");
        }
    }

    if (cnt_passed == test_cnt)
        printf("\nAll tests passed (%zu/%zu)\n", cnt_passed, test_cnt);
    else
        printf("\nTests passed: %zu/%zu\n", cnt_passed, test_cnt);


    // free allocated memory
    for (size_t i = 0; i < test_cnt; i++)
    {
        free(file_paths[i]);
    }
    free(file_paths);

    return EXIT_SUCCESS;
}