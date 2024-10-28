#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <dirent.h>
#include <sys/types.h>
#include <string.h>
#include "ioutil.h"
#include "knnsearch.h"

#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define MIN(a, b) ((a) < (b) ? (a) : (b))


int has_extension(const char *filename, const char *extension) 
{
    size_t len_filename = strlen(filename);
    size_t len_extension = strlen(extension);

    // Check if the file length is at least as long as the extension
    if (len_filename < len_extension) 
    {
        return 0;
    }

    // Compare the end of filename with the extension
    return strcmp(filename + len_filename - len_extension, extension) == 0;
}


int test_case(const char *filename, double tolerance)
{
    printf("Running test %s ...\t\t", filename);
    int M, N, L;

    // load corpus matrix from file
    double *C = (double *)load_matrix(filename, "C", &N, &L);
    if (!C)
    {
        return EXIT_FAILURE;
    }

    // load queries matrix from file
    double *Q = (double *)load_matrix(filename, "Q", &M, &L);
    if (!Q)
    {
        free(C);
        return EXIT_FAILURE;
    }

    // load K nearest neighbors value
    int a, b;
    int *K = (int *)load_matrix(filename, "K", &a, &b);
    if (!K)
    {
        free(C);
        free(Q);
        return EXIT_FAILURE;
    }

    // load expected distances matrix from file
    double *expD = (double *)load_matrix(filename, "D", &a, &b);
    if (!expD)
    {
        free(C);
        free(Q);
        free(K);
        return EXIT_FAILURE;
    }

    // load expected indices matrix from file
    int *expIDX = (int *)load_matrix(filename, "IDX", &a, &b);
    if (!expIDX)
    {
        free(C);
        free(Q);
        free(K);
        free(expD);
        return EXIT_FAILURE;
    }

    // memory allocation for the estimated distance matrix
    double *D = (double *)malloc(M * (*K) * sizeof(double));
    if (!D)
    {
        fprintf(stderr, "Error allocating memory for matrix D\n");
        free(C);
        free(Q);
        free(K);
        free(expD);
        free(expIDX);
        return EXIT_FAILURE;
    }

    // memory allocation for the estimated index matrix
    int *IDX = (int *)malloc(M * (*K) * sizeof(int));
    if (!IDX)
    {
        fprintf(stderr, "Error allocating memory for matrix IDX\n");
        free(C);
        free(Q);
        free(K);
        free(D);
        free(expD);
        free(expIDX);
        return EXIT_FAILURE;
    }

    if (knnsearch_exact(Q, C, IDX, D, M, N, L, *K, 0))
    {
        free(C);
        free(Q);
        free(K);
        free(D);
        free(IDX);
        free(expD);
        free(expIDX);
        return EXIT_FAILURE;
    }

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
            if (fabs(x - y) >= tolerance * MAX(x, y))
            {
                return EXIT_FAILURE;
            }
        }
    }

    for (int i = 0; i < M; i++)
    {
        for (int j = 0; j < *K; j++)
        {
            if (expIDX[i * (*K) + j] != IDX[i * (*K) + j])
            {
                return EXIT_FAILURE;
            }
        }
    }

    free(C);
    free(Q);
    free(K);
    free(D);
    free(IDX);
    free(expD);
    free(expIDX);
    return EXIT_SUCCESS;
}


int main(int argc, char *argv[])
{
    if (argc < 2) 
    {
        fprintf(stderr, "Usage: %s <directory_path>\n", argv[0]);
        return EXIT_FAILURE;
    }

    DIR *dir = opendir(argv[1]);
    if (dir == NULL) 
    {
        fprintf(stderr, "Could not open directory\n");
        return EXIT_FAILURE;
    }

    struct dirent *entry;
    int cnt_passed = 0;
    int cnt_failed = 0;
    while ((entry = readdir(dir)) != NULL) 
    {
        // Skip the current and parent directory entries
        if (entry->d_name[0] == '.' && (entry->d_name[1] == '\0' || (entry->d_name[1] == '.' && entry->d_name[2] == '\0'))) 
        {
            continue;
        }

        // file is not for testing
        if (!has_extension(entry->d_name, ".mat"))
        {
            continue;
        }

        char filename[1024];
        snprintf(filename, sizeof(filename), "%s/%s", argv[1], entry->d_name);
        if (test_case(filename, 1e-6) == EXIT_SUCCESS)
        {
            printf("Passed\n");
            cnt_passed++;
        }
        else
        {
            printf("Failed\n");
            cnt_failed++;
        }
    }

    if (cnt_failed == 0 && cnt_passed > 0)
        printf("\nAll tests passed (%d/%d)\n", cnt_passed, cnt_passed + cnt_failed);
    else if (cnt_failed > 0)
        printf("\nTests passed: %d/%d\n", cnt_passed, cnt_passed + cnt_failed);

    closedir(dir);

    return EXIT_SUCCESS;
}