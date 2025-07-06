#include "knnsearch.h"
#include "knnsearch_exact.h"
#include "knnsearch_approx.h"
#include <stdio.h>
#include <stdlib.h>


int knnsearch(const double* Q, const double* C, int** IDX, double** D, const int M, const int N, const int L, int K, const int sorted, int nthreads, int approx)
{
    K = (K > N) ? N : K;  // if K is greater than the corpus size, set K to the coprus size

    *D = (double *)malloc(M * K * sizeof(double));
    if (!(*D))
    {
        fprintf(stderr, "Error allocating memory for matrix D\n");
        return EXIT_FAILURE;
    }

    *IDX = (int *)malloc(M * K * sizeof(int));
    if (!(*IDX))
    {
        fprintf(stderr, "Error allocating memory for matrix IDX\n");
        return EXIT_FAILURE;
    }
    
    if (approx == 1)
    {
        return knnsearch_approx(Q, C, *IDX, *D, M, N, L, K, sorted, nthreads);
    }
        
    return knnsearch_exact(Q, C, *IDX, *D, M, N, L, K, sorted, nthreads);
}