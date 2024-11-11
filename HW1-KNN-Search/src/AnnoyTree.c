#include "AnnoyTree.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <cblas.h>


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


AnnoyTree *AnnoyTree_create(double *points, int num_points, int dimension, const int LEAF_SIZE)
{
    AnnoyTree* tree = (AnnoyTree *)malloc(sizeof(AnnoyTree));
    if (!tree)
    {
        fprintf(stderr, "Error allocating memory for Annoy Tree\n");
        return NULL;
    }

    tree->dimension = dimension;
    tree->root = build_tree(points, num_points, tree->dimension, LEAF_SIZE);
    if (!tree->root)
    {
        fprintf(stderr, "Failed to create the tree\n");
        free(tree);
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
        node->points = points;
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
    node->points = NULL;  // intermediate nodes do not store points directly
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


double *AnnoyTree_getApproxNeighbors(const AnnoyTree* tree, double* point, int dimension, int *n_neighbors)
{
    if (tree->dimension != dimension)
    {
        *n_neighbors = 0;
        fprintf(stderr, "Invalid dimension for query point\n");
        return NULL;
    }

    return getApproxNeighbors(tree->root, point, dimension, n_neighbors);
}


double *getApproxNeighbors(const Node *node, double *point, int dimension, int *n_neighbors)
{
    if (!node) 
    {
        *n_neighbors = 0;
        return NULL;
    }

    if (!node->direction)  // leaf node, return its data
    {
        *n_neighbors = node->num_points;
        return node->points;
    }

    // Calculate the projection of the query point to the perpendicular bisector
    double projection = cblas_ddot(dimension, node->direction, 1, point, 1);
    if (projection < node->threshold)  // search in the left subtree
    {
        return getApproxNeighbors(node->left, point, dimension, n_neighbors);
    }
    else  // search in the right subtree
    {
        return getApproxNeighbors(node->right, point, dimension, n_neighbors);
    }
}
