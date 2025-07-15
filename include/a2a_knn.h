#ifndef A2A_KNN_H
#define A2A_KNN_H

#include "a2a_config.h"

#define MIN_QUERIES_PER_BLOCK 1           // Minimum number of queries per block


/**
 * Computes the K-Nearest Neighbors (KNN) between a query matrix Q and a corpus matrix C.
 *
 * This method calculates Euclidean distances between each query vector and all corpus vectors,
 * and returns the indices and distances of the K nearest neighbors for each query.
 * It is designed for high-performance nearest neighbor search on large datasets, with options for
 * multi-threading and memory usage control.
 *
 * @param Q The query matrix of shape (M x L). Must be a row-major array of floating-point numbers.
 * @param C The corpus matrix of shape (N x L). Must be a row-major array of floating-point numbers.
 * @param IDX Output array of shape (M x K). Will be populated with zero-based indices of nearest neighbors.
 * @param D Output array of shape (M x K). Will be populated with distances to nearest neighbors.
 * @param M The number of query vectors (rows in Q).
 * @param N The number of corpus vectors (rows in C).
 * @param L The dimensionality of each vector (number of columns in Q and C).
 * @param K The number of nearest neighbors to retrieve. Must be less than or equal to N.
 * @param sorted If non-zero, the output indices and distances may be sorted in ascending distance order.
 *               (Note: sorting is not enforced internally and may require post-processing.)
 * @param nthreads Number of threads to use for parallel computation.
 *                 If -1, the function automatically chooses based on available CPU cores.
 * @param cblas_nthreads Number of threads for BLAS operations (e.g., OpenBLAS GEMM).
 *                       Recommended to set this to 1 if nthreads is greater than 1.
 * @param max_memory_usage_ratio Fraction of available memory allowed for computation (value between 0 and 1].
 * @param par_type Type of parallelization (PTHREADS, OpenMP or OpenCilk).
 *
 * @return 0 (EXIT_SUCCESS) if the computation was successful; 1 (EXIT_FAILURE) otherwise.
 *
 * Notes:
 * - The function uses Euclidean distance and optimizes calculations using the identity:
 *   ||Q - C||^2 = ||Q||^2 + ||C||^2 - 2 * Q * C^T.
 * - The function splits work into memory-friendly blocks automatically based on
 *   max_memory_usage_ratio.
 * - Multi-threading is supported via pthreads, and matrix multiplications are accelerated via BLAS.
 * - IDX and D must be allocated by the caller before the function is called.
 */
int a2a_knnsearch(const DTYPE* Q, const DTYPE* C, int* IDX, DTYPE* D, const int M, 
    const int N, const int L, const int K, const int sorted, const int nthreads,
    const int cblas_nthreads, const double max_memory_usage_ratio, parallelization_type_t par_type);

#endif // KNNSEARCH_H
