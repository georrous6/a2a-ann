#ifndef KNNSEARCH_H
#define KNNSEARCH_H

/**
 * This is the driver function for finding the exact or approximate k-nearest neighbors 
 * of queries for some corpus data.
 * 
 * @param Q the query points (M x L)
 * @param C the corpus points (N x L)
 * @param IDX the matrix of indices (M x K)
 * @param D the matrix of distances (M x K)
 * @param M the number of rows of Q
 * @param N the number of rows of C
 * @param L the number of columns of Q and C
 * @param K the number of nearest neighbors
 * @param sorted if set to a non-negative value outputs the distances in ascending order
 * @param nthreads the number of threads to use. If set to -1 it automatically uses the appropriate number of threads
 * @param approx if set to 1 it computes the approximate k-nearest neighbors
 * @return 0 on succesfull exit and 1 if an error occured
 * @note The function assumes that Q and C have the same number of columns.
 * @note This function allocates memory for IDX and D matrices, so these matrices
 * should be freed outside the function.
 */
int knnsearch(const double* Q, const double* C, int** IDX, double** D, const int M, const int N, const int L, const int K, const int sorted, int nthreads, const int approx);

#endif