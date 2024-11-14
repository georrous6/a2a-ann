#include "knnsearch.h"
#include "knnsearch_exact.h"
#include "knnsearch_approx.h"
#include <stdlib.h>


int knnsearch(const double* Q, double* C, int* IDX, double* D, const int M, const int N, const int L, int K, const int sorted, int nthreads, int approx)
{
    if (K > N && !sorted)  // K is greater than the coprus size and no sorting is needed, return
    {
        return EXIT_SUCCESS;
    }

    K = (K > N) ? N : K;
    
    if (approx == 1)
    {
        return knnsearch_approx(Q, C, IDX, D, M, N, L, K, sorted, nthreads);
    }
        
    return knnsearch_exact(Q, C, IDX, D, M, N, L, K, sorted, nthreads);
}