#include <stdlib.h>
#include "a2a_ann.h"

typedef struct {
    int* indices;
    int count;
} ClusterIndex;

static DTYPE distance_squared(const DTYPE* a, const DTYPE* b, int L) {
    DTYPE dist = ZERO;
    for (int i = 0; i < L; ++i) {
        DTYPE diff = a[i] - b[i];
        dist += diff * diff;
    }
    return dist;
}


static int build_cluster_index(const int* assignments, const int* counts, const int N, 
    const int Kc, ClusterIndex* cluster_index) {

    for (int k = 0; k < Kc; ++k) {
        cluster_index[k].indices = malloc(sizeof(int) * counts[k]);
        if (!cluster_index[k].indices) return EXIT_FAILURE;
        cluster_index[k].count = 0;
    }
    for (int i = 0; i < N; ++i) {
        int cid = assignments[i];
        cluster_index[cid].indices[cluster_index[cid].count++] = i;
    }

    return EXIT_SUCCESS;
}


static int kmeans(const DTYPE* data, int N, int L, int K, 
    DTYPE* centroids, int* assignments, int* counts, int max_iter) {

    int* chosen = calloc(N, sizeof(int));
    if (!chosen) return EXIT_FAILURE;
    int selected = 0;

    // Initialize centroids by randomly selecting K points from data
    while (selected < K) {
        int r = rand() % N;
        if (!chosen[r]) {
            memcpy(centroids + selected * L, data + r * L, L * sizeof(DTYPE));
            chosen[r] = 1;
            selected++;
        }
    }
    free(chosen);

    DTYPE* new_centroids = calloc(K * L, sizeof(DTYPE));
    if (!new_centroids) return EXIT_FAILURE;
    for (int i = 0; i < N; ++i) assignments[i] = -1;  // force at least one change

    // Main k-means loop
    for (int iter = 0; iter < max_iter; ++iter) {

        int changed = 0; // Track if any assignment changed

        // Loop over all points to assign them to the nearest centroid
        for (int i = 0; i < N; ++i) {
            DTYPE best_dist = INF;
            int best_k = -1;

            // Loop over all centroids to find the closest one
            for (int k = 0; k < K; ++k) {
                DTYPE d = distance_squared(data + i * L, centroids + k * L, L);
                if (d < best_dist) {
                    best_dist = d;
                    best_k = k;
                }
            }

            // If the best cluster is different from the previous assignment,
            // mark it as changed and update the assignment.
            if (best_k != assignments[i]) changed = 1;

            assignments[i] = best_k;
        }

        if (!changed) break;  // No change in assignments, exit early

        // Reset centroid accumulators
        memset(counts, 0, sizeof(int) * K);
        for (int i = 0; i < K * L; ++i) new_centroids[i] = ZERO;

        // For each cluster k, recompute its centroid as the mean of 
        // all points assigned to it.
        for (int i = 0; i < N; ++i) {
            int k = assignments[i];
            for (int j = 0; j < L; ++j)
                new_centroids[k * L + j] += data[i * L + j];
            counts[k]++;
        }

        for (int k = 0; k < K; ++k) {
            // If the cluster has points, average them; otherwise, reinitialize
            // the centroid randomly from the data.
            if (counts[k] > 0) {
                for (int j = 0; j < L; ++j)
                    new_centroids[k * L + j] /= counts[k];
            } else {
                int r = rand() % N;
                memcpy(new_centroids + k * L, data + r * L, sizeof(DTYPE) * L);
            }
        }

        memcpy(centroids, new_centroids, sizeof(DTYPE) * K * L);
    }

    free(new_centroids);
    return EXIT_SUCCESS;
}


int a2a_annsearch(const DTYPE* C, const int N, const int L, const int K, 
    const int Kc, int* IDX, DTYPE* D, const int max_iter) {

    int status = EXIT_FAILURE;

    // Allocate local memory for assignments and centroids
    int* assignments = malloc(sizeof(int) * N);
    DTYPE* centroids = malloc(sizeof(DTYPE) * Kc * L);
    int* counts = calloc(Kc, sizeof(int));

    if (!assignments || !centroids || !counts) goto cleanup;

    // Step 1: k-means clustering
    if (kmeans(C, N, L, Kc, centroids, assignments, counts, max_iter)) goto cleanup;

    // Step 2: build cluster point index
    ClusterIndex* cluster_index = malloc(sizeof(ClusterIndex) * Kc);
    if (!cluster_index) goto cleanup;
    for (int k = 0; k < Kc; k++) cluster_index[k].indices = NULL;

    if (build_cluster_index(assignments, counts, N, Kc, cluster_index)) goto cleanup;

    // Step 3: run knnsearch per-cluster
    for (int cid = 0; cid < Kc; ++cid) {
        int cluster_size = cluster_index[cid].count;
        if (cluster_size == 0) continue;

        int* indices = cluster_index[cid].indices;

        // Allocate memory for the submatrix and indices
        DTYPE* C_sub = malloc(sizeof(DTYPE) * cluster_size * L);
        int* idx_sub = malloc(sizeof(int) * cluster_size * (K + 1));
        DTYPE* dist_sub = malloc(sizeof(DTYPE) * cluster_size * (K + 1));
        if (!C_sub || !idx_sub || !dist_sub) {
            if (C_sub) free(C_sub);
            if (idx_sub) free(idx_sub);
            if (dist_sub) free(dist_sub);
            goto cleanup;
        }

        // Construct a submatrix of C for the current cluster
        for (int i = 0; i < cluster_size; ++i) {
            int orig_idx = indices[i];
            for (int l = 0; l < L; ++l)
                C_sub[i * L + l] = C[orig_idx * L + l];
        }

        // Find K nearest neighbors in the submatrix
        if (knnsearch(C_sub, C_sub, idx_sub, dist_sub,
                  cluster_size, cluster_size, L, K + 1, 1)) goto cleanup;  // +1 to skip self

        // Fill the output matrices IDX and D
        for (int i = 0; i < cluster_size; ++i) {
            int orig_i = indices[i];
            int out_k = 0;
            for (int k = 0; k < K + 1; ++k) {
                int local_j = idx_sub[i * (K + 1) + k];
                if (local_j == i) continue; // skip self
                IDX[orig_i * K + out_k] = indices[local_j];
                D[orig_i * K + out_k] = dist_sub[i * (K + 1) + k];
                if (++out_k >= K) break;
            }
        }

        free(C_sub);
        free(idx_sub);
        free(dist_sub);
    }

    status = EXIT_SUCCESS;

cleanup:

    // Cleanup
    if (cluster_index) {
        for (int i = 0; i < Kc; ++i) 
            if (cluster_index[i].indices) free(cluster_index[i].indices);
        free(cluster_index);
    }
    if (assignments) free(assignments);
    if (centroids) free(centroids);
    if (counts) free(counts);

    return status;
}
