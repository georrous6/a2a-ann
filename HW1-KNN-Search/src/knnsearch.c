#include "knnsearch.h"
#include "knnsearch_exact.h"
#include "knnsearch_approx.h"


int knnsearch(const double* Q, double* C, int* IDX, double* D, const int M, const int N, const int L, int K, const int sorted, int nthreads, int approx)
{
    if (approx == 1)
    {
        return knnsearch_approx(Q, C, IDX, D, M, N, L, (K > N) ? N : K, sorted, nthreads);
    }
        
    return knnsearch_exact(Q, C, IDX, D, M, N, L, (K > N) ? N : K, sorted, nthreads);
}