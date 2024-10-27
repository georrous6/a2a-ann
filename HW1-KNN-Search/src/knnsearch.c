#include "knnsearch.h"
#include <stdio.h>
#include <stdlib.h>
#include <cblas.h>
#include <math.h>


void swap(double *arr, size_t *idx, size_t i, size_t j) 
{
    double dtemp = arr[i];
    arr[i] = arr[j];
    arr[j] = dtemp;
    size_t lutemp = idx[i];
    idx[i] = idx[j];
    idx[j] = lutemp;
}


size_t partition(double *arr, size_t *idx, size_t l, size_t r) 
{
    double pivot = arr[r];
    size_t i = l;
    for (size_t j = l; r > 0 && j <= r - 1; j++) 
    {
        if (arr[j] <= pivot) 
        {
            swap(arr, idx, i, j);
            i++;
        }
    }
    swap(arr, idx, i, r);
    return i;
}


void qselect(double *arr, size_t *idx, size_t l, size_t r, size_t k) 
{
    // Partition the array around the last 
    // element and get the position of the pivot 
    // element in the sorted array
    size_t index = partition(arr, idx, l, r);

    // If position is the same as k.
    if (index - l == k - 1)
        return;

    // If position is more, recur for the left subarray.
    if (index - l > k - 1)
    {
        qselect(arr, idx, l, index - 1, k);
        return;
    }

    // Else recur for the right subarray.
    qselect(arr, idx, index + 1, r, k - index + l - 1);
}


void qsort_(double *arr, size_t *idx, size_t l, size_t r) 
{
    if (l < r) 
    {
        // call partition function to find Partition Index
        size_t index = partition(arr, idx, l, r);

        // Recursively call for left and right
        // half based on Partition Index
        qsort_(arr, idx, l, index - 1);
        qsort_(arr, idx, index + 1, r);
    }
}


int knnsearch_exact(const double* Q, const double* C, size_t* IDX, double* D, size_t M, size_t N, size_t L, size_t K, int sorted)
{
    // matrix to store the distances between all row vectors of matrix Q
    // and all row vectors of matrix C
    double *Dall = (double *)malloc(M * N * sizeof(double));
    if (!Dall)
    {
        fprintf(stderr, "Error allocating memory for distance matrix\n");
        return EXIT_FAILURE;
    }

    // input data indices of nearest neighbors
    size_t *IDXall = (size_t *)malloc(M * N * sizeof(size_t));
    if (!IDXall)
    {
        fprintf(stderr, "Error allocating memory for index matrix\n");
        free(Dall);
        return EXIT_FAILURE;
    }

    // initialize index matrix
    for (size_t i = 0; i < M; i++)
    {
        for (size_t j = 0; j < N; j++)
        {
            IDXall[i * N + j] = j;
        }
    }

    // compute D = -2*Q*C'
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, M, N, L, -2.0, Q, L, C, L, 0.0, Dall, N);

    // stores the squares of the magnitudes of the row vectors of matrix Q
    double *sqrmag_Q = (double *)malloc(M * sizeof(double));
    if (!sqrmag_Q)
    {
        fprintf(stderr, "Error allocating memory for square magnitudes vector\n");
        free(Dall);
        free(IDXall);
        return EXIT_FAILURE;
    }

    // stores the squares of the magnitudes of the row vectors of matrix C
    double *sqrmag_C = (double *)malloc(N * sizeof(double));
    if (!sqrmag_C)
    {
        fprintf(stderr, "Error allocating memory for square magnitudes vector\n");
        free(sqrmag_Q);
        free(Dall);
        free(IDXall);
        return EXIT_FAILURE;
    }

    double tmp;
    // compute the square of magnitudes of the row vectors of matrix Q
    for (size_t i = 0; i < M; i++)
    {
        tmp = 0.0;
        for (size_t j = 0; j < L; j++)
        {
            tmp += Q[i * L + j] * Q[i * L + j]; 
        }
        sqrmag_Q[i] = tmp;
    }

    // compute the square of magnitudes of the row vectors of matrix C
    for (size_t i = 0; i < N; i++)
    {
        tmp = 0.0;
        for (size_t j = 0; j < L ; j++)
        {
            tmp += C[i * L + j] * C[i * L + j]; 
        }
        sqrmag_C[i] = tmp;
    }

    // compute the distance matrix D by applying the formula D = sqrt(C.^2 -2*Q*C' + (Q.^2)')
    for (size_t i = 0; i < M; i++)
    {
        for (size_t j = 0; j < N; j++)
        {
            Dall[i * N + j] = sqrt(Dall[i * N + j] + sqrmag_Q[i] + sqrmag_C[j]);
        }
    }

    if (!sorted)
    {
        // apply Quick Select algorithm for each row of distance matrix
        for (size_t i = 0; i < M; i++)
        {
            qselect(Dall + i * N, IDXall + i * N, 0, N - 1, K);
        }
    }
    else
    {
        // apply Quick Sort algorithm for each row of distance matrix
        for (size_t i = 0; i < M; i++)
        {
            qsort_(Dall + i * N, IDXall + i * N, 0, N - 1);
        }      
    }

    // now copy the first K elements of each row of matrices
    // Dall, IDXall to D and IDX respectivelly
    for (size_t i = 0; i < M; i++)
    {
        for (size_t j = 0; j < K; j++)
        {
            D[i * K + j] = Dall[i * N + j];
            IDX[i * K + j] = IDXall[i * N + j] + 1;
        }
    }

    free(sqrmag_C);
    free(sqrmag_Q);
    free(Dall);
    free(IDXall);
    return EXIT_SUCCESS;
}
