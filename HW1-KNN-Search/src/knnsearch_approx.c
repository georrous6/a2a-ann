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


void swap_points(AnnoyTree *tree, double* points, int dimension, int idx1, int idx2)
{
    // save the initial indexes before swaping
    const int offset1 = (int)(points - tree->points) / dimension + idx1;
    const int offset2 = (int)(points - tree->points) / dimension + idx2;
    int tmp_index = tree->idx[offset1];
    tree->idx[offset1] = tree->idx[offset2];
    tree->idx[offset2] = tmp_index;

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

    tree->idx = (int *)malloc(sizeof(int) * num_points);
    if (!tree->idx)
    {
        AnnoyTree_destroy(tree);
        fprintf(stderr, "Error allocating memory for indexes of points of the tree\n");
        return NULL;    
    }

    for (int i = 0; i < num_points; i++) tree->idx[i] = i;
    
    tree->dimension = dimension;
    tree->root = build_tree(tree, tree->points, num_points, tree->dimension, LEAF_SIZE);
    if (!tree->root)
    {
        AnnoyTree_destroy(tree);
        fprintf(stderr, "Failed to create the Annoy tree\n");
        return NULL;
    }
    return tree;
}


Node* build_tree(AnnoyTree *tree, double *points, int num_points, int dimension, const int LEAF_SIZE) 
{
    Node *node = (Node *)malloc(sizeof(Node));
    if (!node) 
    {
        fprintf(stderr, "Error allocating memory for node\n");
        return NULL;
    }

    if (num_points <= LEAF_SIZE || num_points <= 1)  // create a leaf node
    {
        node->points_ptr = points;
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

        swap_points(tree, points, dimension, left_idx, right_idx);
        left_idx++;
        right_idx--;
    }

    // Recursively build left and right subtrees
    node->direction = direction;
    node->threshold = threshold;
    node->points_ptr = NULL;  // intermediate nodes do not store points directly
    node->num_points = 0;
    node->left = build_tree(tree, points, left_idx, dimension, LEAF_SIZE);
    node->right = build_tree(tree, points + left_idx * dimension, num_points - left_idx, dimension, LEAF_SIZE);
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
    if (tree->idx) free(tree->idx);
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


void getApproxNeighbors(const AnnoyTree* tree, const Node *node, const double *point, double *neighbors, int *idx_neighbors, const int dimension, int *n_neighbors, const int K)
{
    if (!node || *n_neighbors >= K) 
    {
        return;
    }

    if (!node->direction)  // leaf node
    {
        const int idx_offset = (int)(node->points_ptr - tree->points) / dimension;
        memcpy(idx_neighbors + *n_neighbors, tree->idx + idx_offset, sizeof(int) * node->num_points);
        memcpy(neighbors + *n_neighbors * dimension, node->points_ptr, sizeof(double) * dimension * node->num_points);
        *n_neighbors += node->num_points;
        return;
    }

    // Calculate the projection of the query point to the perpendicular bisector
    double projection = cblas_ddot(dimension, node->direction, 1, point, 1);
    if (projection < node->threshold)  // search in the left subtree
    {
        getApproxNeighbors(tree, node->left, point, neighbors, idx_neighbors, dimension, n_neighbors, K);
        if (*n_neighbors < K)  // did not found K nearest neighbors, search in the other subtree
        {
            getApproxNeighbors(tree, node->right, point, neighbors, idx_neighbors, dimension, n_neighbors, K);    
        }
    }
    else  // search in the right subtree
    {
        getApproxNeighbors(tree, node->right, point, neighbors, idx_neighbors, dimension, n_neighbors, K);
        if (*n_neighbors < K)  // did not found K nearest neighbors, search in the other subtree
        {
            getApproxNeighbors(tree, node->left, point, neighbors, idx_neighbors, dimension, n_neighbors, K);
        }
    }
}


int knnsearch_approx(const double* Q, const double* C, int* IDX, double* D, const int M, const int N, const int L, const int K, const int sorted, int nthreads)
{
    // Create an auxiliary array for storing the approximate nearest neighbors
    int *idx_neighbors = (int *)malloc(sizeof(int) * N);
    double *neighbors = (double *)malloc(sizeof(double) * N * L);
    int status = EXIT_FAILURE;

    if (!idx_neighbors || !neighbors)
    {
        fprintf(stderr, "knnsearch_approx: Error allocating memory\n");
        goto cleanup;
    }

    // Create the search tree from the corpus
    AnnoyTree* tree = AnnoyTree_create(C, N, L, MAX_LEAF_SIZE);
    if (!tree)
    {
        fprintf(stderr, "Error allocating memory for ANNOY tree\n");
        goto cleanup;
    }

    // find the approximate K-nearest neighbors of each query
    int n_neighbors;
    for (int i = 0; i < M; i++)
    {
        n_neighbors = 0;
        getApproxNeighbors(tree, tree->root, Q + i * L, neighbors, idx_neighbors, L, &n_neighbors, K);
        if (knn(Q + i * L, neighbors, IDX + i * K, idx_neighbors, D + i * K, 1, n_neighbors, L, K, sorted))
        {
            goto cleanup;
        }
    }

    if (store_matrix((void *)IDX, "IDX_approx", M, K, "/home/grous/THMMY-AUTH/Semester07/PDS/PDS-HW-2024-25/HW1-KNN-Search/test/approx_tests/my_output.mat", INT_TYPE, 'w'))
    {
        printf("Could not save data to mat file\n");
    }

    status = EXIT_SUCCESS;

cleanup:
    if (idx_neighbors) free(idx_neighbors);
    if (neighbors) free(neighbors);
    AnnoyTree_destroy(tree);
    return status;
}

int knn(const double* Q, const double* C, int* IDX, int *IDXall, double* D, const int M, const int N, const int L, const int K, const int sorted)
{
    double *Dall = NULL, *sqrmag_Q = NULL, *sqrmag_C = NULL;
    int status = EXIT_FAILURE;

    Dall = (double *)malloc(M * N * sizeof(double));
    sqrmag_Q = (double *)malloc(M * sizeof(double));
    sqrmag_C = (double *)malloc(N * sizeof(double));

    if (!Dall || !sqrmag_C || !sqrmag_Q)
    {
        fprintf(stderr, "Error allocating memory in knn\n");
        goto cleanup;
    }

    // compute D = -2*Q*C'
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, M, N, L, -2.0, Q, L, C, L, 0.0, Dall, N);

    // compute the square of magnitudes of the row vectors of matrix Q
    for (int i = 0; i < M; i++)
    {
        sqrmag_Q[i] = cblas_ddot(L, Q + i * L, 1, Q + i * L, 1);
    }

    // compute the square of magnitudes of the row vectors of matrix C
    for (int i = 0; i < N; i++)
    {
        sqrmag_C[i] = cblas_ddot(L, C + i * L, 1, C + i * L, 1);
    }

    // compute the distance matrix D by applying the formula D = sqrt(C.^2 -2*Q*C' + (Q.^2)')
    for (int i = 0; i < M; i++)
    {
        for (int j = 0; j < N; j++)
        {
            Dall[i * N + j] = sqrt(Dall[i * N + j] + sqrmag_Q[i] + sqrmag_C[j]);
        }
    }
    
    if (!sorted)
    {
        // apply Quick Select algorithm for each row of distance matrix
        for (int i = 0; i < M; i++)
        {
            qselect(Dall + i * N, IDXall + i * N, 0, N - 1, K);
        }
    }
    else
    {
        // apply Quick Sort algorithm for each row of distance matrix
        for (int i = 0; i < M; i++)
        {
            qsort_(Dall + i * N, IDXall + i * N, 0, N - 1);
        }      
    }

    for (int i = 0; i < M; i++)
    {
        for (int j = 0; j < K; j++)
        {
            D[i * K + j] = Dall[i * N + j];
            IDX[i * K + j] = IDXall[i * N + j] + 1; // 1-based indexing
        }
    }

    status = EXIT_SUCCESS;

cleanup:
    if (Dall) free(Dall);
    if (sqrmag_Q) free(sqrmag_Q);
    if (sqrmag_C) free(sqrmag_C);

    return status;
}
