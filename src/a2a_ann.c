#include <stdlib.h>
#include <string.h>
#include "knn.h"
#include <pthread.h>
#include <sys/sysinfo.h>
#include <unistd.h>
#include "a2a_ann.h"


typedef struct {
    int* indices;
    int count;
} ClusterIndex;

// Structure to sort clusters by size
typedef struct {
    int id;
    int size;
} ClusterSizeEntry;


typedef struct annTask {
    int* cluster_ids;             // Cluster indices assigned to this task
    int num_clusters;             // Number of clusters in this task
    ClusterIndex* cluster_index;  // Cluster index for this task
    int L;                        // Dimension of the data points
    int K;                        // Number of nearest neighbors to find
    const DTYPE* C;               // Original data matrix
    DTYPE* D;                     // Output distance matrix
    int* IDX;                     // Output index matrix
} annTask;


void ann_set_num_threads(int n) {

    const long num_cores = sysconf(_SC_NPROCESSORS_ONLN);  // Number of online processors
    if (num_cores < 1) {
        perror("sysconf\n");
        return;
    }

    n = n < 1 ? (int)num_cores : n;
    printf("Setting number of threads to %d\n", n);

    atomic_store(&ANN_NUM_THREADS, n);
}


int ann_get_num_threads(void) {
    return atomic_load(&ANN_NUM_THREADS);
}


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


static void *annTaskExec(void *arg) {

    annTask * task = (annTask *)arg;
    int num_clusters = task->num_clusters;
    int *cluster_ids = task->cluster_ids;
    ClusterIndex* cluster_index = task->cluster_index;
    int* IDX = task->IDX;
    DTYPE* D = task->D;
    const DTYPE* C = task->C;
    int L = task->L;
    int K = task->K;


    for (int c = 0; c < num_clusters; ++c) {
        const int cid = cluster_ids[c];
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
            pthread_exit(NULL);
        }

        // Construct a submatrix of C for the current cluster
        for (int i = 0; i < cluster_size; ++i) {
            int orig_idx = indices[i];
            for (int l = 0; l < L; ++l)
                C_sub[i * L + l] = C[orig_idx * L + l];
        }

        // Find K nearest neighbors in the submatrix
        if (knnsearch(C_sub, C_sub, idx_sub, dist_sub, cluster_size, cluster_size, L, K + 1, 1)) {
            free(C_sub);
            free(idx_sub);
            free(dist_sub);
            pthread_exit(NULL);
        }

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

    pthread_exit(NULL);
}


// Comparator for sorting clusters by descending size
static int compare_cluster_sizes(const void* a, const void* b) {
    const ClusterSizeEntry* ca = (const ClusterSizeEntry*)a;
    const ClusterSizeEntry* cb = (const ClusterSizeEntry*)b;
    return cb->size - ca->size; // descending
}

static int distribute_clusters_by_size(int Kc, int nthreads, ClusterIndex* cluster_index, annTask* tasks) {
    int* thread_load = calloc(nthreads, sizeof(int));
    if (!thread_load) return EXIT_FAILURE;

    ClusterSizeEntry* entries = malloc(sizeof(ClusterSizeEntry) * Kc);
    if (!entries) {
        free(thread_load);
        return EXIT_FAILURE;
    }

    // Fill cluster ID + size entries
    for (int i = 0; i < Kc; ++i) {
        entries[i].id = i;
        entries[i].size = cluster_index[i].count;
    }

    // Sort clusters by size descending
    qsort(entries, Kc, sizeof(ClusterSizeEntry), compare_cluster_sizes);

    // Initialize tasks
    for (int i = 0; i < nthreads; ++i) {
        tasks[i].cluster_ids = (int *)malloc(sizeof(int) * Kc); // max size
        if (!tasks[i].cluster_ids) {
            for (int j = 0; j < i; ++j) free(tasks[j].cluster_ids);
            free(thread_load);
            free(entries);
            return EXIT_FAILURE;
        }
        tasks[i].num_clusters = 0;
    }

    // Greedy bin-packing
    for (int i = 0; i < Kc; ++i) {
        int cid = entries[i].id;
        int min_thread = 0;
        for (int t = 1; t < nthreads; ++t) {
            if (thread_load[t] < thread_load[min_thread])
                min_thread = t;
        }
        tasks[min_thread].cluster_ids[tasks[min_thread].num_clusters++] = cid;
        thread_load[min_thread] += entries[i].size;
    }

    free(thread_load);
    free(entries);
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

    const int nthreads = ann_get_num_threads();
    const int NTHREADS = Kc > nthreads ? nthreads : Kc; 
    
    pthread_t *threads = (pthread_t *)malloc(sizeof(pthread_t) * NTHREADS);
    if (!threads) goto cleanup;
    annTask* tasks = (annTask *)malloc(sizeof(annTask) * NTHREADS);
    if (!tasks) goto cleanup;

    // Distribute clusters among threads
    if (distribute_clusters_by_size(Kc, NTHREADS, cluster_index, tasks)) goto cleanup;

    // Set OpenBLAS and KNN to single-threaded mode
    knn_set_num_threads_cblas(1);
    knn_set_num_threads(1);
    for (int i = 0; i < NTHREADS; ++i) {
        tasks[i].cluster_index = cluster_index;
        tasks[i].L = L;
        tasks[i].K = K;
        tasks[i].C = C;
        tasks[i].D = D;
        tasks[i].IDX = IDX;

        if (pthread_create(&threads[i], NULL, annTaskExec, (void *)&tasks[i])) {
            fprintf(stderr, "Error creating thread %d\n", i);
            for (int j = 0; j < i; ++j) {
                pthread_join(threads[j], NULL);
            }
            goto cleanup;
        }
    }

    for (int i = 0; i < NTHREADS; ++i) {
        if (pthread_join(threads[i], NULL)) {
            fprintf(stderr, "Error joining thread %d\n", i);
            goto cleanup;
        }
    }

    status = EXIT_SUCCESS;

cleanup:

    // Cleanup
    if (cluster_index) {
        for (int i = 0; i < Kc; ++i) 
            if (cluster_index[i].indices) free(cluster_index[i].indices);
        free(cluster_index);
    }
    if (tasks) {
        for (int i = 0; i < NTHREADS; ++i) {
            if (tasks[i].cluster_ids) free(tasks[i].cluster_ids);
        }
        free(tasks);
    }
    if (assignments) free(assignments);
    if (centroids) free(centroids);
    if (counts) free(counts);

    return status;
}
