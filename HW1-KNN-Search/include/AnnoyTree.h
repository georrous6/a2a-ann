#ifndef ANNOY_TREE_H
#define ANNOY_TREE_H


typedef struct Node {
    double *points;      // points of the node
    int num_points;      // number of points of the node
    struct Node *left;   // left node
    struct Node *right;  // right node
    double *direction;   // the direction vector perpendicular to the hyperplane spliting the points into two subsets
    double threshold;    // threshold for partitioning points into two subsets
} Node;


typedef struct AnnoyTree {
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
 * @param points a 1D array of multidimensional points
 * @param dimension the dimension of the points
 * @param idx1 the index of the first point
 * @param idx2 the index of the second point
 */
void swap_points(double* points, int dimension, int idx1, int idx2);

/**
 * Driver function for the construction of the tree.
 * Calls the recursive function build_tree and sets the dimension of the data.
 * 
 * @param points the points of the current node
 * @param num_points the number of points to be partitioned
 * @param dimension the dimension of the points
 * @param LEAF_SIZE the maximum number of points to a leaf node
 * @return the Annoy tree or NULL if an error occured
 */
AnnoyTree *AnnoyTree_create(double *points, int num_points, int dimension, const int LEAF_SIZE);

/**
 * Recursive function for the construction of the tree.
 * 
 * @param points the points of the current node
 * @param num_points the number of points to be partitioned
 * @param dimension the dimension of the points
 * @param LEAF_SIZE the maximum number of points to a leaf node
 * @return a leaf or an intermediate node or NULL if an error occured
 */
Node* build_tree(double *points, int num_points, int dimension, const int LEAF_SIZE);


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
 * Driver function to find the approximate nearest neighbors of a query point.
 * Calls the recursive function getApproxNeighbors.
 * 
 * @param node the root node of the tree
 * @param point the query point
 * @param dimension the dimansion of the query point
 * @param n_neighbors the number of the approximate nearest neighbors
 * @return a pointer to the approximate nearest neighbors
 */
double *AnnoyTree_getApproxNeighbors(const AnnoyTree* tree, double* point, int dimension, int *n_neighbors);

/**
 * Recursive function that finds the approximate nearest neighbors of the query point.
 * 
 * @param node the root node of the tree
 * @param point the query point
 * @param dimension the dimansion of the query point
 * @param n_neighbors the number of the approximate nearest neighbors
 * @return a pointer to the approximate nearest neighbors
 */
double *getApproxNeighbors(const Node *node, double *point, int dimension, int *n_neighbors);

#endif
