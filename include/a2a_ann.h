#ifndef A2A_ANN_H
#define A2A_ANN_H

#include "a2a_config.h"


/**
 * Performs an Approximate Nearest Neighbor (ANN) search on a dataset using
 * k-means clustering and parallelized k-NN search within clusters.
 * 
 * The function first partitions the data points into Kc clusters using k-means,
 * then searches for the K nearest neighbors of each point within its cluster.
 * If only one cluster is specified, it falls back to an exact k-NN search.
 * The workload is parallelized over multiple threads.
 * 
 * @param C                       Pointer to the dataset, an array of N points each with L dimensions.
 * @param N                       Number of data points.
 * @param L                       Dimensionality of each data point.
 * @param K                       Number of nearest neighbors to find per point.
 * @param Kc                      Number of clusters to partition the data into.
 * @param IDX                     Output array (size N * K) to store indices of nearest neighbors.
 * @param D                       Output array (size N * K) to store distances to nearest neighbors.
 * @param nthreads                Number of threads to use for parallel computation.
 * @param max_memory_usage_ratio  Maximum allowed memory usage ratio (between 0 and 1).
 * @param par_type                Type of parallelization (PTHREADS, OpenMP or OpenCilk).
 * 
 * @return                        EXIT_SUCCESS (0) on success, or EXIT_FAILURE (non-zero) on error.
 */
int a2a_annsearch(const DTYPE* C, const int N, const int L, const int K, int Kc, 
    int* IDX, DTYPE* D, const int nthreads, const double max_memory_usage_ratio, 
    parallelization_type_t par_type);


#endif