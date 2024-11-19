#ifndef KNNSEARCH_APPROX_H
#define KNNSEARCH_APPROX_H

#define MAX_LEAF_SIZE 1000  // maximum number of points on a leaf


/**
 * Function to create the perpendicular bisector plane between two points
 * 
 * @param p1 the first point
 * @param p2 the second point
 * @param L the dimensions of the points
 * @param threshold a threshold value for spliting the points into two subsets
 * @return the direction vector (p2 - p1) which is perpendicular to the hyperplane
 * @note the direction vector is allocated dynamically therefore, memory deallocation
 * should take place outside this function 
 */
double* perpendicular_bisector(double *p1, double *p2, int L, double *threshold);

/**
 * Swap two points at specific indexes.
 * 
 * @param tree the tree
 * @param points a 1D array of multidimensional points
 * @param L the dimension of the points
 * @param idx1 the index of the first point
 * @param idx2 the index of the second point
 * @note idx1 and idx2 indexes have base the points pointer and not the tree->points
 */
void swap_points(double* Q, int* IDX, const int L, const int idx1, const int idx2);

/**
 * Recursive function for the construction of the tree.
 * 
 * @param tree the tree
 * @param points the points of the current node
 * @param num_points the number of points to be partitioned
 * @param L the dimension of the points
 * @param LEAF_SIZE the maximum number of points to a leaf node
 * @return a leaf or an intermediate node or NULL if an error occured
 */
int ann_recursive(double *C, int *mp,  int *IDX, double *D, const int K, const int index, const int num_points, const int L, const int LEAF_SIZE, const int sorted);


/**
 * Computes the approximate k-nearest neighbors of each point in Q
 * 
 * @param Q the query points (M x L)
 * @param IDX the matrix of indices (M x K)
 * @param D the matrix of distances (M x K)
 * @param M the number of rows of Q
 * @param L the number of columns of Q
 * @param K the number of nearest neighbors
 * @param sorted if set to a non-negative value outputs the distances in ascending order
 * @param nthreads the number of threads to use. 
 * If set to -1 it automatically uses the appropriate number of threads
 * @return 0 on succesfull exit and 1 if an error occured
 * @note The user is responsible to pass IDX and D matrices with the appropriate
 * dimensions
 */
int knnsearch_approx(const double* Q, int* IDX, double* D, const int M, const int L, const int K, const int sorted, int nthreads);


/**
 * Finds the exact k-nearest neighbors. Usefull for small problem size.
 * 
 * @param Q the query points (M x L)
 * @param C the corpus points (N x L)
 * @param IDX the matrix of indices (M x K)
 * @param IDXall the matrix of the initial indices (M x N)
 * @param D the matrix of distances (M x K)
 * @param M the number of rows of Q
 * @param N the number of rows of C
 * @param L the number of columns of Q and C
 * @param K the number of nearest neighbors
 * @param sorted if set to a non-negative value outputs 
 * the distances in ascending order
 * @return 0 on succesfull exit and 1 if an error occured
 * @note The function assumes that Q and C have the same number of columns.
 * @note The function assumes that the IDXall matrix is already initialized 
 * with zero-based indexes.
 */
int knn(const double* Q, const double* C, int *IDX, int* IDXall, double* D, const int M, const int N, const int L, const int K, const int sorted);

#endif
