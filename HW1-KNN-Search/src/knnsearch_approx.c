#include "knnsearch_approx.h"
#include "knnsearch_exact.h"
#include "ioutil.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include <cblas.h>


double* perpendicular_bisector(double *p1, double *p2, int dimension, double *threshold) 
{
    double *direction = (double*)malloc(dimension * sizeof(double));
    if (!direction)
    {
        fprintf(stderr, "Error allocating memory for direction vector\n");
        return NULL;
    }

    for (int i = 0; i < dimension; i++) 
    {
        direction[i] = p2[i] - p1[i];
    }

    *threshold = 0.0;

    for (int i = 0; i < dimension; i++) 
    {
        *threshold += 0.5 * (p1[i] + p2[i]) * direction[i];
    }

    return direction;
}


void swap_points(double* points, int dimension, int idx1, int idx2)
{
    double tmp;
    for (int i = 0; i < dimension; i++)
    {
        tmp = points[idx1 * dimension + i];
        points[idx1 * dimension + i] = points[idx2 * dimension + i];
        points[idx2 * dimension + i] = tmp;
    }
}


AnnoyTree *AnnoyTree_create(const double *points, int num_points, int dimension, const int LEAF_SIZE)
{
    AnnoyTree* tree = (AnnoyTree *)malloc(sizeof(AnnoyTree));
    if (!tree)
    {
        fprintf(stderr, "Error allocating memory for Annoy Tree\n");
        return NULL;
    }

     // copy the corpus data into another array
    tree->points = (double *)malloc(sizeof(double) * num_points * dimension);
    if (!tree->points)
    {
        AnnoyTree_destroy(tree);
        fprintf(stderr, "Error allocating memory for the points of the tree\n");
        return NULL;
    }

    memcpy(tree->points, points, num_points * dimension * sizeof(double));
    
    tree->dimension = dimension;
    tree->root = build_tree(tree->points, num_points, tree->dimension, LEAF_SIZE);
    if (!tree->root)
    {
        AnnoyTree_destroy(tree);
        fprintf(stderr, "Failed to create the Annoy tree\n");
        return NULL;
    }
    return tree;
}


Node* build_tree(double *points, int num_points, int dimension, const int LEAF_SIZE) 
{
    Node *node = (Node *)malloc(sizeof(Node));
    if (!node) 
    {
        fprintf(stderr, "Error allocating memory for node\n");
        return NULL;
    }

    if (num_points <= LEAF_SIZE || num_points <= 1)  // create a leaf node
    {
        printf("Created leaf with %d elements\n", num_points);
        node->ptr = points;
        node->num_points = num_points;
        node->left = node->right = NULL;
        node->direction = NULL;
        return node;
    }

    // Create the perpendicular bisector plane from the first two points
    double threshold;
    double *direction = perpendicular_bisector(points, points + dimension, dimension, &threshold);
    if (!direction)
    {
        free(node);
        return NULL;
    }

    int left_idx = 0, right_idx = num_points - 1;

    // partition the points according to their projection on the hyperplane
    while (left_idx <= right_idx) 
    {
        double projection_left = cblas_ddot(dimension, direction, 1, points + left_idx * dimension, 1);
        if (projection_left < threshold) 
        {
            left_idx++;
            continue;
        }

        double projection_right = cblas_ddot(dimension, direction, 1, points + right_idx * dimension, 1);
        if (projection_right >= threshold) 
        {
            right_idx--;
            continue;
        }

        swap_points(points, dimension, left_idx, right_idx);
        left_idx++;
        right_idx--;
    }

    // Recursively build left and right subtrees
    node->direction = direction;
    node->threshold = threshold;
    node->ptr = NULL;  // intermediate nodes do not store points directly
    node->num_points = 0;
    node->left = build_tree(points, left_idx, dimension, LEAF_SIZE);
    node->right = build_tree(points + left_idx * dimension, num_points - left_idx, dimension, LEAF_SIZE);
    if (!node->left || !node->right)
    {
        destroy_tree(node);  // destroy left or right subtrees of the currect node
        return NULL;
    }

    return node;
}


void AnnoyTree_destroy(AnnoyTree* tree)
{
    if (!tree) return;
    if (tree->points) free(tree->points);
    destroy_tree(tree->root);
    free(tree);
}


void destroy_tree(Node* node)
{
    if (!node) return;
    if (node->left) destroy_tree(node->left);
    if (node->right) destroy_tree(node->right);
    if (node->direction) free(node->direction);
    free(node);
}


void getApproxNeighbors(const Node *node, const double *point, double *neighbors, const int dimension, int *n_neighbors, const int K)
{
    if (!node || *n_neighbors >= K) 
    {
        return;
    }

    if (!node->direction)  // leaf node
    {
        printf("Reached leaf node with %d elements\n", node->num_points);
        memcpy(neighbors + *n_neighbors, node->ptr, sizeof(double) * dimension * node->num_points);
        *n_neighbors += node->num_points;
        printf("Exiting leaf node, neighbors remaining to find: %d/%d\n", *n_neighbors, K);
        return;
    }

    // Calculate the projection of the query point to the perpendicular bisector
    double projection = cblas_ddot(dimension, node->direction, 1, point, 1);
    if (projection < node->threshold)  // search in the left subtree
    {
        printf("Not a leaf, going left, neighbors remaining to find %d/%d\n", *n_neighbors, K);
        getApproxNeighbors(node->left, point, neighbors, dimension, n_neighbors, K);
        if (*n_neighbors < K)  // did not found K nearest neighbors, search in the other subtree
        {
            printf("Backtrack to the right, neighbors remaining to find %d/%d\n", *n_neighbors, K);
            getApproxNeighbors(node->right, point, neighbors, dimension, n_neighbors, K);    
        }
    }
    else  // search in the right subtree
    {
        printf("Not a leaf, going right, neighbors remaining to find %d/%d\n", *n_neighbors, K);
        getApproxNeighbors(node->right, point, neighbors, dimension, n_neighbors, K);
        if (*n_neighbors < K)  // did not found K nearest neighbors, search in the other subtree
        {
            printf("Backtrack to the left, neighbors remaining to find %d/%d\n", *n_neighbors, K);
            getApproxNeighbors(node->left, point, neighbors, dimension, n_neighbors, K);
        }
    }
}


int knnsearch_approx(const double* Q, const double* C, int* IDX, double* D, const int M, const int N, const int L, const int K, const int sorted, int nthreads)
{
    // Create an auxiliary array for storing the approximate nearest neighbors
    double *neighbors = (double *)malloc(sizeof(double) * N);
    if (!neighbors)
    {
        fprintf(stderr, "Error allocating memory for the copy of the corpus\n");
        return EXIT_FAILURE;
    }

    // Create the search tree from the corpus
    AnnoyTree* tree = AnnoyTree_create(C, N, L, MAX_LEAF_SIZE);
    if (!tree)
    {
        free(neighbors);
        fprintf(stderr, "Error allocating memory for ANNOY tree\n");
        return EXIT_FAILURE;
    }

    // find the approximate K-nearest neighbors of each query
    int n_neighbors;
    for (int i = 0; i < M; i++)
    {
        n_neighbors = 0;
        getApproxNeighbors(tree->root, Q + i * L, neighbors, L, &n_neighbors, K);
    }

    for (int i = 0; i < n_neighbors; i++)
    {
        for (int j = 0; j < L; j++)
        {
            printf("%lf ", neighbors[i * L + j]);
        }
        printf("\n");
    }

    if (store_matrix((void *)neighbors, "approx_neighbors", n_neighbors, L, "/home/grous/THMMY-AUTH/Semester07/PDS/PDS-HW-2024-25/HW1-KNN-Search/test/approx_tests/my_output.mat", DOUBLE_TYPE))
    {
        printf("Could not save data to mat file\n");
    }

    free(neighbors);
    AnnoyTree_destroy(tree);
    return EXIT_SUCCESS;
}
