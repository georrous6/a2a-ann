#ifndef KNNSEARCH_H
#define KNNSEARCH_H

#include "Queue.h"
#include "template_definitions.h"


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
