#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "ioutil.h"
#include "knnsearch.h"


int test_case(const char *filename, double *execution_time)
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

    // memory allocation for the estimated distance matrix
    D = (double *)malloc(M * (*K) * sizeof(double)); if (!D) goto cleanup;

    // memory allocation for the estimated index matrix
    IDX = (int *)malloc(M * (*K) * sizeof(int)); if (!IDX) goto cleanup;

    clock_t start = clock();
    if (knnsearch_exact(Q, C, IDX, D, M, N, L, *K, 0)) goto cleanup;
    clock_t end = clock();
    *execution_time = ((double) (end - start)) / CLOCKS_PER_SEC;

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
    char** file_paths = NULL;
    double *exec_times = NULL;
    int status = EXIT_FAILURE;

    if (argc < 3) 
    {
        fprintf(stderr, "Usage: %s <directory_path> <output_file>\n", argv[0]);
        return status;
    }


    size_t cnt_passed = 0;
    size_t test_cnt;
    file_paths = get_file_paths(argv[1], ".mat", &test_cnt, 1);
    if (!file_paths) 
    {
        goto cleanup;
    }

    // array holding the execution time of each test case
    exec_times = (double *)malloc(sizeof(double) * test_cnt);
    if (!exec_times)
    {
        fprintf(stderr, "Error allocating memory for execution times\n");
        goto cleanup;
    }

    for (size_t i = 0; i < test_cnt; i++) 
    {
        if (test_case(file_paths[i], &exec_times[i]))
            goto cleanup;
    }

    // store execution times in a file
    if (store_matrix((void *)exec_times, "my_execution_times", 1, test_cnt, argv[2], DOUBLE_TYPE))
        goto cleanup;

    status = EXIT_SUCCESS;

    cleanup:
    if (file_paths)
    {
        for (size_t i = 0; i < test_cnt; i++) 
            free(file_paths[i]);
        free(file_paths);
    }
    if (exec_times) free(exec_times);

    return status;
}