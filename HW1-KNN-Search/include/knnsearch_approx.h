#ifndef KNNSEARCH_APPROX_H
#define KNNSEARCH_APPROX_H

#define MAX_LEAF_SIZE 1000  // maximum number of points on a leaf


typedef struct Node {
    double *points_ptr;         // pointer to the points of the node
    int num_points;             // number of points of the node
    struct Node *left;          // left node
    struct Node *right;         // right node
    double *direction;          // the direction vector perpendicular to the hyperplane spliting the points into two subsets
    double threshold;           // threshold for partitioning points into two subsets
} Node;


typedef struct AnnoyTree {
    double *points;      // the points of the tree
    int *idx;            // the initial indexes of the points, zero based indexing
    int dimension;       // the dimension of the points
    Node *root;          // the root node of the tree
} AnnoyTree;


/**
 * Function to create the perpendicular bisector plane between two points
 * 
 * @param p1 the first point
 * @param p2 the second point
 * @param dimension the dimensions of the points
 * @param threshold a threshold value for spliting the points into two subsets
 * @return the direction vector (p2 - p1) which is perpendicular to the hyperplane
 * @note the direction vector is allocated dynamically therefore, memory deallocation
 * should take place outside this function 
 */
double* perpendicular_bisector(double *p1, double *p2, int dimension, double *threshold);

/**
 * Swap two points at specific indexes.
 * 
 * @param tree the tree
 * @param points a 1D array of multidimensional points
 * @param dimension the dimension of the points
 * @param idx1 the index of the first point
 * @param idx2 the index of the second point
 * @note idx1 and idx2 indexes have base the points pointer and not the tree->points
 */
void swap_points(AnnoyTree* tree, double* points, int dimension, int idx1, int idx2);

/**
 * Driver function for the construction of the tree.
 * Calls the recursive function build_tree and sets the dimension of the data.
 * 
 * @param points the points to store in the tree
 * @param num_points the number of points to be partitioned
 * @param dimension the dimension of the points
 * @param LEAF_SIZE the maximum number of points to a leaf node
 * @return the Annoy tree or NULL if an error occured
 */
AnnoyTree *AnnoyTree_create(const double *points, int num_points, int dimension, const int LEAF_SIZE);

/**
 * Recursive function for the construction of the tree.
 * 
 * @param tree the tree
 * @param points the points of the current node
 * @param num_points the number of points to be partitioned
 * @param dimension the dimension of the points
 * @param LEAF_SIZE the maximum number of points to a leaf node
 * @return a leaf or an intermediate node or NULL if an error occured
 */
Node* build_tree(AnnoyTree *tree, double *points, int num_points, int dimension, const int LEAF_SIZE);


/**
 * Driver function to deallocate memory used for the tree.
 * Calls the recursive function destroy_tree.
 * 
 * @param tree the tree to destroy
 */
void AnnoyTree_destroy(AnnoyTree* tree);


/**
 * Recursive function that deallocates memory used for the tree.
 * 
 * @param node the root node of the tree
 * @note the points field is not deallocated since it points to the original data
 */
void destroy_tree(Node* node);


/**
 * Recursive function that finds the approximate nearest neighbors of the query point.
 * It returns the indexes of the approximate K-nearest neighbors.
 * 
 * @param tree the tree 
 * @param node the root node of the tree
 * @param point the query point
 * @param neighbors the approximate nearest neighbors
 * @param idx_neighbors the indexes of the approximate nearest neighbors
 * @param dimension the dimansion of the query point
 * @param n_neighbors the number of the approximate nearest neighbors already found
 * @param K the number of the approximate neighbors to find
 */
void getApproxNeighbors(const AnnoyTree* tree, const Node *node, const double *point, double *neighbors, int *idx_neighbors, const int dimension, int *n_neighbors, const int K);


/**
 * Computes the approximate k-nearest neighbors for each row vector in Q with each
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
 * @param nthreads the number of threads to use. 
 * If set to -1 it automatically uses the appropriate number of threads
 * @return 0 on succesfull exit and 1 if an error occured
 * @note The function assumes that Q and C have the same number of columns.
 * @note The user is responsible to pass IDX and D matrices with the appropriate
 * dimensions
 */
int knnsearch_approx(const double* Q, const double* C, int* IDX, double* D, const int M, const int N, const int L, const int K, const int sorted, int nthreads);


/**
 * Finds the exact k-nearest neighbors. Usefull for small problem size.
 * 
 * @param Q the query points (M x L)
 * @param C the corpus points (N x L)
 * @param IDX the matrix of indices (M x K)
 * @param IDXall the matrix of indices of the original data (M x N)
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
int knn(const double* Q, const double* C, int* IDX, int *IDXall, double* D, const int M, const int N, const int L, const int K, const int sorted);

#endif
