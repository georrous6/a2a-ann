#ifndef KNN_H
#define KNN_H

#include "Queue.h"
#include "template_definitions.h"
#include <stdatomic.h>

#define KNN_MAX_MEMORY_USAGE_RATIO 0.8        // Maximum memory usage ratio
#define KNN_MIN_QUERIES_PER_BLOCK 1           // Minimum number of queries per block

static atomic_int KNN_NUM_THREADS = 1;

void knn_set_num_threads(int n);

int knn_get_num_threads(void);

void knn_set_num_threads_cblas(int n);

int knn_get_num_threads_cblas(void);


/**
 * Computes the exact k-nearest neighbors for each row vector in Q with each
 * row vector in C.
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
 * @return 0 on succesfull exit and 1 if an error occured
 * @note The function assumes that Q and C have the same number of columns.
 * @note The user is responsible to pass IDX and D matrices with the appropriate
 * dimensions
 */
int knnsearch(const DTYPE* Q, const DTYPE* C, int* IDX, DTYPE* D, const int M, const int N, const int L, const int K, const int sorted);

#endif // KNNSEARCH_H
