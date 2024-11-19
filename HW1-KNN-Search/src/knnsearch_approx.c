#include "knnsearch_approx.h"
#include "knnsearch_exact.h"
#include "ioutil.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include <cblas.h>


double* perpendicular_bisector(double *p1, double *p2, int L, double *threshold) 
{
    double *direction = (double*)malloc(L * sizeof(double));
    if (!direction)
    {
        fprintf(stderr, "Error allocating memory for direction vector\n");
        return NULL;
    }

    for (int i = 0; i < L; i++) 
    {
        direction[i] = p2[i] - p1[i];
    }

    *threshold = 0.0;

    for (int i = 0; i < L; i++) 
    {
        *threshold += 0.5 * (p1[i] + p2[i]) * direction[i];
    }

    return direction;
}


void swap_points(double* Q, int *mp, const int L, const int idx1, const int idx2)
{
    int tmp_idx = mp[idx1];
    mp[idx1] = mp[idx2];
    mp[idx2] = tmp_idx;

    double tmp;
    for (int i = 0; i < L; i++)
    {
        tmp = Q[idx1 * L + i];
        Q[idx1 * L + i] = Q[idx2 * L + i];
        Q[idx2 * L + i] = tmp;
    }
}


int ann_recursive(double *Q, int *mp, int *IDX, double *D, const int K, const int index, const int num_points, const int L, const int LEAF_SIZE, const int sorted) 
{
    if (num_points <= LEAF_SIZE || num_points == 1 || num_points <= K)
    {
        printf("Reached leaf at index %d with %d elements. Sorting...\n", index, num_points);
        // Reached a leaf. Find the exact k-nearest neighbors on this region
        int *IDXall = (int *)malloc(sizeof(int) * num_points * num_points);
        if (!IDXall)
        {
            fprintf(stderr, "ann_recursive: Error allocating memory\n");
            return EXIT_FAILURE;
        }

        for (int i = 0; i < num_points; i++)
        {
            for (int j = 0; j < num_points; j++)
            {
                IDXall[i * num_points + j] = mp[index + j];
            }
        }

        const int kk = K > num_points ? num_points : K;
        if (knn(Q + index * L, Q + index * L, IDX + index * K, IDXall, D, num_points, num_points, L, kk, sorted))
        {
            free(IDXall);
            return EXIT_FAILURE;
        }

        free(IDXall);
        return EXIT_SUCCESS;
    }

    // Create the perpendicular bisector plane from the first two points
    double threshold;
    double *direction = perpendicular_bisector(Q + index * L, Q + (index + 1) * L, L, &threshold);
    if (!direction)
    {
        return EXIT_FAILURE;
    }

    int left_idx = index, right_idx = index + num_points - 1;

    // partition the points according to their projection on the hyperplane
    while (left_idx <= right_idx) 
    {
        double projection_left = cblas_ddot(L, direction, 1, Q + left_idx * L, 1);
        if (projection_left < threshold) 
        {
            left_idx++;
            continue;
        }

        double projection_right = cblas_ddot(L, direction, 1, Q + right_idx * L, 1);
        if (projection_right >= threshold) 
        {
            right_idx--;
            continue;
        }

        swap_points(Q, mp, L, left_idx, right_idx);
        left_idx++;
        right_idx--;
    }

    // node->left = build_tree(tree, points, left_idx, L, LEAF_SIZE);
    // node->right = build_tree(tree, points + left_idx * L, num_points - left_idx, L, LEAF_SIZE);
    
    const int num_points_left = left_idx - index;
    const int num_points_right = num_points - left_idx;
    if (num_points_left > 0)
    {
        printf("Creating left leaf at index %d with %d elements\n", left_idx, num_points_left);
        ann_recursive(Q, mp, IDX, D, K, index, num_points_left, L, LEAF_SIZE, sorted);
    }
    if (num_points_right > 0)
    {
        printf("Creating right leaf at index %d with %d elements\n", right_idx, num_points_right);
        ann_recursive(Q, mp, IDX, D, K, left_idx, num_points_right, L, LEAF_SIZE, sorted);
    }

    // double *temp = NULL;
    // // Merge solutions
    // if (num_points_left > 0 && num_points_right > 0)
    // {
    //     temp = (double *)malloc(sizeof(double) * (num_points_left + num_points_right));
    //     if (!temp)
    //     {
    //         fprintf(stderr, "Error allocating memory\n");
    //         return EXIT_FAILURE;
    //     }

    //     for (int i = 0; i < num_points_left; i++)
    // }

    // if (temp) free(temp);
    return EXIT_SUCCESS;
}


int knnsearch_approx(const double* Q, int* IDX, double* D, const int M, const int L, const int K, const int sorted, int nthreads)
{
    // Mapping vector to retrieve the initial Q matrix
    int *mp = (int *)malloc(sizeof(int) * M);

    if (!mp)
    {
        fprintf(stderr, "Error allocating memory for mapping vector\n");
        return EXIT_FAILURE;
    }

    double *copy_Q = (double *)malloc(sizeof(double) * M * L);
    if (!copy_Q)
    {
        fprintf(stderr, "Error allocating memory for the copy of the coprus/queries matrix\n");
        return EXIT_FAILURE;    
    }

    memcpy(copy_Q, Q, sizeof(double) * M * L);

    for (int i = 0; i < M ; i++) mp[i] = i;
    if (ann_recursive(copy_Q, mp, IDX, D, K, 0, M, L, MAX_LEAF_SIZE, sorted))
    {
        free(mp);
        free(copy_Q);
        return EXIT_FAILURE;
    }

    double temp_D;
    int temp_IDX;
    for (int i = 0; i < M; i++)
    {
        if (i != mp[i])
        {
            for (int j = 0; j < K; j++)
            {
                temp_D = D[i * K + j];
                D[i * K + j] = D[mp[i] * K + j];
                D[mp[i] * K + j] = temp_D;

                temp_IDX = IDX[i * K + j];
                IDX[i * K + j] = IDX[mp[i] * K + j];
                IDX[mp[i] * K + j] = temp_IDX;
            }
        }
    }

    free(copy_Q);
    free(mp);
    return EXIT_SUCCESS;
}

int knn(const double* Q, const double* C, int *IDX, int* IDXall, double* D, const int M, const int N, const int L, const int K, const int sorted)
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
            Dall[i * N + j] += sqrmag_Q[i] + sqrmag_C[j];
        }
    }
    
    // apply Quick Select algorithm for each row of distance matrix
    for (int i = 0; i < M; i++)
    {
        qselect(Dall + i * N, IDXall + i * N, 0, N - 1, K);
    }
 

    for (int i = 0; i < M; i++)
    {
        for (int j = 0; j < K; j++)
        {
            D[i * K + j] = sqrt(Dall[i * N + j]);
            IDX[i * K + j] = IDXall[i * N + j]; // zero-based indexing
        }

        // sort each row of the distance matrix
        if (sorted)
        {
            qsort_(D + i * K, IDX + i * K, 0, K - 1);
        }
    }

    status = EXIT_SUCCESS;

cleanup:
    if (Dall) free(Dall);
    if (sqrmag_Q) free(sqrmag_Q);
    if (sqrmag_C) free(sqrmag_C);

    return status;
}
