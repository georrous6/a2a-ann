#include "knnsearch.h"
#include <stdio.h>
#include <stdlib.h>
#include <cblas.h>
#include <math.h>


void swap(double *arr, int *idx, int i, int j) 
{
    double dtemp = arr[i];
    arr[i] = arr[j];
    arr[j] = dtemp;
    int lutemp = idx[i];
    idx[i] = idx[j];
    idx[j] = lutemp;
}


int partition(double *arr, int *idx, int l, int r) 
{
    double pivot = arr[r];
    int i = l;
    for (int j = l; j <= r - 1; j++) 
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


void qselect(double *arr, int *idx, int l, int r, int k) 
{
    // Partition the array around the last 
    // element and get the position of the pivot 
    // element in the sorted array
    int index = partition(arr, idx, l, r);

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


void qsort_(double *arr, int *idx, int l, int r) 
{
    if (l < r) 
    {
        // call partition function to find Partition Index
        int index = partition(arr, idx, l, r);

        // Recursively call for left and right
        // half based on Partition Index
        qsort_(arr, idx, l, index - 1);
        qsort_(arr, idx, index + 1, r);
    }
}


int knnsearch_exact(const double* Q, const double* C, int* IDX, double* D, int M, int N, int L, int K, int sorted)
{
    int status = EXIT_FAILURE;
    double *Dall = NULL, *sqrmag_Q = NULL, *sqrmag_C = NULL;
    int *IDXall = NULL;
    if (K <= 0 || K > N)
    {
        fprintf(stderr, "Invalid value for K: %d\n", K);
        return status;
    }

    // matrix to store the distances between all row vectors of matrix Q
    // and all row vectors of matrix C
    Dall = (double *)malloc(M * N * sizeof(double));
    if (!Dall)
    {
        fprintf(stderr, "Error allocating memory for distance matrix\n");
        goto cleanup;
    }

    // input data indices of nearest neighbors
    IDXall = (int *)malloc(M * N * sizeof(int));
    if (!IDXall)
    {
        fprintf(stderr, "Error allocating memory for index matrix\n");
        goto cleanup;
    }

    // initialize index matrix
    for (int i = 0; i < M; i++)
    {
        for (int j = 0; j < N; j++)
        {
            IDXall[i * N + j] = j;
        }
    }

    // compute D = -2*Q*C'
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, M, N, L, -2.0, Q, L, C, L, 0.0, Dall, N);

    // stores the squares of the magnitudes of the row vectors of matrix Q
    sqrmag_Q = (double *)malloc(M * sizeof(double));
    if (!sqrmag_Q)
    {
        fprintf(stderr, "Error allocating memory for square magnitudes vector\n");
        goto cleanup;
    }

    // stores the squares of the magnitudes of the row vectors of matrix C
    sqrmag_C = (double *)malloc(N * sizeof(double));
    if (!sqrmag_C)
    {
        fprintf(stderr, "Error allocating memory for square magnitudes vector\n");
        goto cleanup;
    }

    // compute the square of magnitudes of the row vectors of matrix Q
    for (int i = 0; i < M; i++)
    {
        sqrmag_Q[i] = cblas_ddot(L, Q + i * L, 1, Q + i * L, 1);
    }

    // compute the square of magnitudes of the row vectors of matrix C
    for (int i = 0; i < N; i++)
    {
        sqrmag_C[i] = cblas_ddot(L, C + i * L, 1, C + i * L, 1);
    }

    // compute the distance matrix D by applying the formula D = sqrt(C.^2 -2*Q*C' + (Q.^2)')
    for (int i = 0; i < M; i++)
    {
        for (int j = 0; j < N; j++)
        {
            Dall[i * N + j] = sqrt(Dall[i * N + j] + sqrmag_Q[i] + sqrmag_C[j]);
        }
    }

    if (!sorted)
    {
        // apply Quick Select algorithm for each row of distance matrix
        for (int i = 0; i < M; i++)
        {
            qselect(Dall + i * N, IDXall + i * N, 0, N - 1, K);
        }
    }
    else
    {
        // apply Quick Sort algorithm for each row of distance matrix
        for (int i = 0; i < M; i++)
        {
            qsort_(Dall + i * N, IDXall + i * N, 0, N - 1);
        }      
    }

    // now copy the first K elements of each row of matrices
    // Dall, IDXall to D and IDX respectivelly
    for (int i = 0; i < M; i++)
    {
        for (int j = 0; j < K; j++)
        {
            D[i * K + j] = Dall[i * N + j];
            IDX[i * K + j] = IDXall[i * N + j] + 1;
        }
    }

    status = EXIT_SUCCESS;

    cleanup:
    if (sqrmag_C) free(sqrmag_C);
    if (sqrmag_Q) free(sqrmag_Q);
    if (Dall) free(Dall);
    if (IDXall) free(IDXall);
    return status;
}
