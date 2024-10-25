#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <string.h>
#include <time.h>
#include <cblas.h>
#include <mat.h>
#include <math.h>

#define DATA_DIM 2
#define CORPUS_DATA_SIZE 20
#define QUERY_DATA_SIZE 1


double* load_matrix_from_file(const char *filename, const char* matname, size_t* rows, size_t* cols)
{
    MATFile *pmat;
    mxArray *arr;
    double *data = NULL;

    // Open the .mat file for reading
    pmat = matOpen(filename, "r");
    if (!pmat) 
    {
        fprintf(stderr, "Error opening MAT-file \'%s\'. %s\n", filename, strerror(errno));
        return NULL;
    }

    // Read the matrix from the .mat file
    arr = matGetVariable(pmat, matname);
    if (!arr) 
    {
        fprintf(stderr, "Variable \'%s\' not found in MAT-file \'%s\'\n", matname, filename);
        matClose(pmat);
        return NULL;
    }

    // Store dimensions
    *rows = mxGetM(arr);
    *cols = mxGetN(arr);

    // Allocate memory to copy matrix data
    data = (double *)malloc((*rows) * (*cols) * sizeof(double));
    if (!data) 
    {
        fprintf(stderr, "Error allocating memory for matrix data\n");
        mxDestroyArray(arr);
        matClose(pmat);
        return NULL;
    }

    // Copy matrix data
    memcpy(data, mxGetPr(arr), (*rows) * (*cols) * sizeof(double));

    // Clean up
    mxDestroyArray(arr);
    matClose(pmat);

    return data;
}


void print_matrix(const double* mat, size_t rows, size_t cols)
{
    for (size_t i = 0; i < rows; i++)
    {
        for (size_t j = 0; j < cols; j++)
        {
            printf("%lf ", mat[i * cols + j]);
        }
        printf("\n");
    }
}


int knnsearch(const double* Q, const double* C, double* D, int M, int N, int K, int k)
{
    // compute D = -2*Q*C'
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, M, N, K, -2.0, Q, K, C, K, 0.0, D, N);

    // stores the squares of the magnitudes of the row vectors of matrix Q
    double *sqrmag_Q = (double *)malloc(M * sizeof(double));
    if (!sqrmag_Q)
    {
        fprintf(stderr, "Error allocating memory for squared magnitudes\n");
        return EXIT_FAILURE;
    }

    // stores the squares of the magnitudes of the row vectors of matrix C
    double *sqrmag_C = (double *)malloc(N * sizeof(double));
    if (!sqrmag_C)
    {
        fprintf(stderr, "Error allocating memory for squared magnitudes\n");
        free(sqrmag_Q);
        return EXIT_FAILURE;
    }

    double tmp;
    for (int i = 0; i < M; i++)
    {
        tmp = 0.0;
        for (int j = 0; j < K ; j++)
        {
            tmp += Q[i * K + j] * Q[i * K + j]; 
        }
        sqrmag_Q[i] = tmp;
    }

    for (int i = 0; i < N; i++)
    {
        tmp = 0.0;
        for (int j = 0; j < K ; j++)
        {
            tmp += C[i * K + j] * C[i * K + j]; 
        }
        sqrmag_C[i] = tmp;
    }

    for (int i = 0; i < M; i++)
    {
        for (int j = 0; j < N; j++)
        {
            D[i * N + j] = sqrt(D[i * N + j] + sqrmag_Q[i] + sqrmag_C[j]);
        }
    }

    free(sqrmag_C);
    free(sqrmag_Q);
    return EXIT_SUCCESS;
}

int main(int argc, char *argv[])
{
    size_t CM, CN, QM, QN;
    double* C = load_matrix_from_file("data.mat", "corpus", &CM, &CN);
    if (!C)
        return EXIT_FAILURE;

    double* Q = load_matrix_from_file("data.mat", "queries", &QM, &QN);
    if (!Q)
    {
        free(C);
        return EXIT_FAILURE;
    }

    if (CN != QN)
    {
        fprintf(stderr, "Invalid dimensions for corpus and query data\n");
        free(C);
        free(Q);
        return EXIT_FAILURE;
    }

    double *D = (double *)malloc(QM * CM * sizeof(double));
    if (!D)
    {
        fprintf(stderr, "Error allocating memory\n");
        free(C);
        free(Q);
        return EXIT_FAILURE;
    }

    if (knnsearch(Q, C, D, QM, CM, QN, 1))
    {
        free(C);
        free(Q);
        free(D);
        return EXIT_FAILURE;
    }

    printf("\n\nCorpus:");
    print_matrix(C, CM, CN);
    printf("\n\nQueries:");
    print_matrix(Q, QM, QN);
    printf("\n\nDistances:");
    print_matrix(D, QM, CM);
    printf("\n");

    free(C);
    free(Q);
    free(D);
    return EXIT_SUCCESS;

}