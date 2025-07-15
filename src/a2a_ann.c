#include <stdlib.h>
#include <string.h>
#include "a2a_knn.h"
#include <pthread.h>
#include <sys/sysinfo.h>
#include <unistd.h>
#include <stdatomic.h>
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
    int* cluster_ids;                    // Cluster indices assigned to this task
    int num_clusters;                    // Number of clusters in this task
    ClusterIndex* cluster_index;         // Cluster index for this task
    int L;                               // Dimension of the data points
    int K;                               // Number of nearest neighbors to find
    const DTYPE* C;                      // Original data matrix
    DTYPE* D;                            // Output distance matrix
    int* IDX;                            // Output index matrix
    int N;                               // Total number of data points
    double max_memory_usage_ratio;       // Maximum memory usage ratio
} annTask;


static DTYPE distance_squared(const DTYPE* a, const DTYPE* b, int L) {
    DTYPE dist = SUFFIX(0.0);
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


static int kmeans(const DTYPE* data, const int N, const int L, const int K, int *Kc, 
    int **assignments, int **counts, const int nthreads, const double max_memory_usage_ratio) {

    *assignments = NULL;
    *counts = NULL;

    DTYPE *centroids = NULL, *queries = NULL, *D = NULL;
    int *chosen = NULL, *IDX, *queries_map = NULL, *valid_clusters = NULL;
    int *tmp_assignments = NULL, *tmp_counts = NULL;
    int status = EXIT_FAILURE;

    centroids = (DTYPE *)malloc((*Kc) * L * sizeof(DTYPE));
    queries = (DTYPE *)malloc((N - (*Kc)) * L * sizeof(DTYPE));
    chosen = (int *)calloc(N, sizeof(int));
    queries_map = (int *)malloc((N - (*Kc)) * sizeof(int));
    IDX = (int *)malloc((N - (*Kc)) * sizeof(int));
    D = (DTYPE *)malloc((N - (*Kc)) * (*Kc) * sizeof(DTYPE));
    valid_clusters = (int *)malloc((*Kc) * sizeof(int));
    tmp_assignments = (int *)malloc(N * sizeof(int));
    tmp_counts = (int *)malloc((*Kc) * sizeof(int));

    if (!chosen || !centroids || !queries || !queries_map || !IDX || 
        !valid_clusters || !tmp_assignments || !tmp_counts || !D) {
        fprintf(stderr, "Error allocating memory for k-means clustering\n");
        goto cleanup;
    }

    // Initialize centroids by randomly selecting K points from data
    int centroid_idx = 0;
    memset(tmp_counts, 0, (*Kc) * sizeof(int));  // Set counts to zero
    while (centroid_idx < *Kc) {
        int r = rand() % N;
        if (!chosen[r]) {
            memcpy(centroids + centroid_idx * L, data + r * L, L * sizeof(DTYPE));
            chosen[r] = 1;
            tmp_counts[centroid_idx]++;
            tmp_assignments[r] = centroid_idx++;
        }
    }

    // Initialize queries by copying the non-chosen points
    int query_idx = 0;
    for (int i = 0; i < N; i++) {
        if (!chosen[i]) {
            memcpy(queries + query_idx * L, data + i * L, L * sizeof(DTYPE));
            queries_map[query_idx++] = i;  // map query index to original index
        }
    }
    free(chosen); chosen = NULL;

    // Assign each query to the nearest centroid
    if (a2a_knnsearch(queries, centroids, IDX, D, N - (*Kc), *Kc, L, 1, 0, 
    nthreads, 1, max_memory_usage_ratio)) goto cleanup;

    // Map the indices back to the original data points
    for (int i = 0; i < N - (*Kc); i++) {
        int query_original_idx = queries_map[i];
        int cluster_index = IDX[i];
        tmp_counts[cluster_index]++;
        tmp_assignments[query_original_idx] = cluster_index;
    }
    free(IDX); IDX = NULL;

    // Compute the new centroids by averaging the assigned points
    memset(centroids, 0, (*Kc) * L * sizeof(DTYPE));
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < L; j++) {
            centroids[tmp_assignments[i] * L + j] += data[i * L + j];
        }
    }

    for (int i = 0; i < *Kc; i++) {
        for (int j = 0; j < L; j++) {
            centroids[i * L + j] /= tmp_counts[i];
        }
    }

    // Merge clusters that have size smaller than K to the closest centroid to them
    memset(valid_clusters, 1, (*Kc) * sizeof(int));  // Set all clusters as valid initially
    int Kc_new = *Kc;
    while (1) {
        int invalid_cluster_index = -1;
        // Find the first invalid cluster (size < K)
        for (int i = 0; i < *Kc; i++) {
            if (tmp_counts[i] < K && valid_clusters[i]) {
                invalid_cluster_index = i;
                break;
            }
        }

        if (invalid_cluster_index == -1) break;
        valid_clusters[invalid_cluster_index] = 0;  // Mark as invalid

        int closest_cluster_index = -1;  // Index of the closest valid cluster
        DTYPE min_dist = INF;

        // Find the closest valid cluster to the invalid one
        for (int i = 0; i < *Kc; i++) {
            if (valid_clusters[i] && i != invalid_cluster_index) {
                DTYPE dist = distance_squared(centroids + invalid_cluster_index * L, centroids + i * L, L);
                if (dist < min_dist) {  // If the distance is very small, merge
                    closest_cluster_index = i;
                    min_dist = dist;
                }
            }
        }

        // This should never happen since I check if N / Kc > K
        // Thus there will always be at least one valid cluster
        if (closest_cluster_index == -1) {
            DEBUG_PRINT("ANN: No valid cluster found to merge with");
            goto cleanup;
        }
        DEBUG_PRINT("ANN: Merging cluster %d -> %d\n", invalid_cluster_index, closest_cluster_index);

        tmp_counts[closest_cluster_index] += tmp_counts[invalid_cluster_index];

        // Recompute the centroid of the closest cluster
        memset(centroids + closest_cluster_index * L, 0, L * sizeof(DTYPE));  // Reset the closest centroid
        for (int i = 0; i < N; i++) {
            if (tmp_assignments[i] == invalid_cluster_index) {
                tmp_assignments[i] = closest_cluster_index;
            }

            // now add all points that are assigned to the closest cluster
            if (tmp_assignments[i] == closest_cluster_index) {
                for (int j = 0; j < L; j++) {
                    centroids[closest_cluster_index * L + j] += data[i * L + j];
                }
            }
        }

        for (int j = 0; j < L; j++) {
            centroids[closest_cluster_index * L + j] /= tmp_counts[closest_cluster_index];
        }

        Kc_new--;  // Reduce the number of clusters
    }

    *assignments = (int *)malloc(N * sizeof(int));
    *counts = (int *)malloc(Kc_new * sizeof(int));
    if (!(*assignments) || !(*counts)) {
        fprintf(stderr, "Error allocating memory for k-means clustering\n");
        goto cleanup;
    }

    // Reassign the assignments to the new clusters
    int cluster_index = 0;
    for (int i = 0; i < *Kc; i++) {
        if (valid_clusters[i]) {
            for (int j = 0; j < N; j++) {
                if (tmp_assignments[j] == i) {
                    (*assignments)[j] = cluster_index;
                }
            }
            (*counts)[cluster_index++] = tmp_counts[i];
        }
    }
    *Kc = Kc_new;
    DEBUG_PRINT("ANN: K-means clustering completed with %d clusters\n", *Kc);

    status = EXIT_SUCCESS;

cleanup:
    if (queries) free(queries);
    if (queries_map) free(queries_map);
    if (IDX) free(IDX);
    if (D) free(D);
    if (chosen) free(chosen);
    if (centroids) free(centroids);
    if (valid_clusters) free(valid_clusters);
    if (tmp_assignments) free(tmp_assignments);
    if (tmp_counts) free(tmp_counts);
    if (status != EXIT_SUCCESS) {
        if (*assignments) free(*assignments);
        if (*counts) free(*counts);
        *assignments = NULL;
        *counts = NULL;
    }

    return status;
}


static void *annTaskExec(void *arg) {

    annTask * task = (annTask *)arg;
    const int num_clusters = task->num_clusters;
    int *cluster_ids = task->cluster_ids;
    ClusterIndex* cluster_index = task->cluster_index;
    int* IDX = task->IDX;
    DTYPE* D = task->D;
    const DTYPE* C = task->C;
    const int L = task->L;
    const int K = task->K;
    const int N = task->N;
    const double max_memory_usage_ratio = task->max_memory_usage_ratio;

    DTYPE *C_sub = NULL, *dist_sub = NULL;
    int *idx_sub = NULL;

    // Compute the total number of points across all clusters for the current thread
    int total_thread_points = 0;
    for (int c = 0; c < num_clusters; ++c) {
        const int cid = cluster_ids[c];
        total_thread_points += cluster_index[cid].count;
    }

    int *retval = (int *)malloc(sizeof(int));
    if (!retval) return NULL;
    *retval = EXIT_SUCCESS;

    DEBUG_PRINT("\nANN: Running thread %lu with %d assigned clusters and %d points in total \n", pthread_self(), num_clusters, total_thread_points);

    for (int c = 0; c < num_clusters; ++c) {
        const int cid = cluster_ids[c];
        int cluster_size = cluster_index[cid].count;
        DEBUG_PRINT("\nANN: Solving cluster %d with %d points\n", cid, cluster_size);
        
        // This should never happen
        DEBUG_ASSERT(cluster_size > 0, "ANN: Cluster size must be greater than 0\n");

        int* indices = cluster_index[cid].indices;

        // Allocate memory for the submatrix and indices
        C_sub = (DTYPE *)malloc(sizeof(DTYPE) * cluster_size * L);
        dist_sub = (DTYPE *)malloc(sizeof(DTYPE) * cluster_size * (K + 1));
        idx_sub = (int *)malloc(sizeof(int) * cluster_size * (K + 1));
        if (!C_sub || !idx_sub || !dist_sub) {
            if (C_sub) free(C_sub);
            if (idx_sub) free(idx_sub);
            if (dist_sub) free(dist_sub);
            free(retval);
            return NULL;
        }

        // Construct a submatrix of C for the current cluster
        for (int i = 0; i < cluster_size; ++i) {
            int orig_idx = indices[i];
            for (int l = 0; l < L; ++l)
                C_sub[i * L + l] = C[orig_idx * L + l];
        }

        // Find K nearest neighbors in the submatrix
        const double memory_usage_ratio = max_memory_usage_ratio * (double)total_thread_points / (double)N;
        if (a2a_knnsearch(C_sub, C_sub, idx_sub, dist_sub, cluster_size, cluster_size, L, K + 1, 0, 1, 1, memory_usage_ratio)) {
            free(C_sub);
            free(idx_sub);
            free(dist_sub);
            free(retval);
            return NULL;
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

    return (void *)retval;
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

    ClusterSizeEntry* entries = (ClusterSizeEntry *)malloc(sizeof(ClusterSizeEntry) * Kc);
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


static int check_input_args_ann(const DTYPE* C, const int N, const int L, const int K, 
    int Kc, int* IDX, DTYPE* D, const int nthreads, 
    const double max_memory_usage_ratio) {

    if (!C || N <= 0 || L <= 0 || K <= 0 || Kc <= 0 || !IDX || !D) {
        fprintf(stderr, "Invalid input parameters for ANN search\n");
        return EXIT_FAILURE;
    }
    if (Kc > N) {
        fprintf(stderr, "Number of clusters cannot exceed number of data points\n");
        return EXIT_FAILURE;
    }
    if (N / Kc <= K) {
        fprintf(stderr, "Number of clusters is too small for the given K\n");
        return EXIT_FAILURE;
    }
    if (nthreads < 1) {
        fprintf(stderr, "Number of threads must be at least 1\n");
        return EXIT_FAILURE;
    }
    if (max_memory_usage_ratio <= 0.0 || max_memory_usage_ratio > 1.0) {
        fprintf(stderr, "Invalid memory usage ratio: %f\n", max_memory_usage_ratio);
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}


static int pthreads_parallelization(annTask* tasks, int nthreads) {

    int status = EXIT_FAILURE;
    pthread_t *threads = (pthread_t *)malloc(sizeof(pthread_t) * nthreads);
    if (!threads) return EXIT_FAILURE;

    for (int i = 0; i < nthreads; ++i) {
        
        if (pthread_create(&threads[i], NULL, annTaskExec, (void *)&tasks[i])) {
            fprintf(stderr, "Error creating thread %d\n", i);
            for (int j = 0; j < i; ++j) {
                pthread_join(threads[j], NULL);
            }
            goto cleanup;
        }
    }

    for (int i = 0; i < nthreads; ++i) {
        void *retval = NULL;
        if (pthread_join(threads[i], &retval)) {
            fprintf(stderr, "Error joining thread %d\n", i);
            goto cleanup;
        }

        if (retval == NULL) {
            fprintf(stderr, "Thread %d returned NULL\n", i);
            goto cleanup;
        }
        int *thread_status = (int *)retval;
        if (*thread_status != EXIT_SUCCESS) {
            fprintf(stderr, "Thread %d failed with status %d\n", i, *thread_status);
            free(retval);
            goto cleanup;
        }
    }

    status = EXIT_SUCCESS;

    cleanup:
    free(threads);
    return status;
}


static int openmp_parallelization(annTask* tasks, int nthreads) {
    #ifndef USE_OPENCILK
        int status = EXIT_SUCCESS;
        #pragma omp parallel num_threads(nthreads)
        {
            int tid = omp_get_thread_num();
            annTask *task = &tasks[tid];
            void *retval = annTaskExec((void *)task);

            if (retval == NULL) {
                #pragma omp critical
                {
                    fprintf(stderr, "Error executing task in OpenMP thread %d\n", tid);
                    status = EXIT_FAILURE;
                }
            } else {
                int *thread_status = (int *)retval;
                if (*thread_status != EXIT_SUCCESS) {
                    #pragma omp critical
                    {
                        fprintf(stderr, "Task in OpenMP thread %d failed with status %d\n", tid, *thread_status);
                        status = EXIT_FAILURE;
                    }
                }
                free(retval);
            }
        }

        return status;
    #else
        fprintf(stderr, "OpenMP is not enabled in this build\n");
        return EXIT_FAILURE;
    #endif
}


static int opencilk_parallelization(annTask* tasks, int nthreads) {
    #ifdef USE_OPENCILK
        atomic_int status = ATOMIC_VAR_INIT(EXIT_SUCCESS);

        cilk_for (int i = 0; i < nthreads; ++i) {
            annTask *task = &tasks[i];
            void *retval = annTaskExec((void *)task);

            if (retval == NULL) {
                fprintf(stderr, "Error executing task in OpenCilk thread %d\n", i);
                atomic_store(&status, EXIT_FAILURE);
            } else {
                int *thread_status = (int *)retval;
                if (*thread_status != EXIT_SUCCESS) {
                    fprintf(stderr, "Task in OpenCilk thread %d failed with status %d\n", i, *thread_status);
                    atomic_store(&status, EXIT_FAILURE);
                }
                free(retval);
            }
        }

        return atomic_load(&status);
    #else
        fprintf(stderr, "OpenCilk is not enabled in this build\n");
        return EXIT_FAILURE;
    #endif
}


int a2a_annsearch(const DTYPE* C, const int N, const int L, const int K, 
    int Kc, int* IDX, DTYPE* D, const int nthreads,
    const double max_memory_usage_ratio, parallelization_type_t par_type) {

    if (check_input_args_ann(C, N, L, K, Kc, IDX, D, nthreads, max_memory_usage_ratio)) {
        return EXIT_FAILURE;
    }

    int status = EXIT_FAILURE;
    int *assignments = NULL, *counts = NULL;
    ClusterIndex* cluster_index = NULL;
    pthread_t *threads = NULL;
    annTask* tasks = NULL;

    if (Kc == 1) {
        // Fall back to exact solution
        if (a2a_knnsearch(C, C, IDX, D, N, N, L, K, 0, nthreads, 1, max_memory_usage_ratio)) {
            return EXIT_FAILURE;
        }
        return EXIT_SUCCESS;
    }

    // Step 1: k-means clustering
    if (kmeans(C, N, L, K, &Kc, &assignments, &counts, nthreads, max_memory_usage_ratio)) goto cleanup;

    // Step 2: build cluster point index
    cluster_index = (ClusterIndex *)malloc(sizeof(ClusterIndex) * Kc);
    if (!cluster_index) goto cleanup;
    for (int k = 0; k < Kc; k++) cluster_index[k].indices = NULL;

    if (build_cluster_index(assignments, counts, N, Kc, cluster_index)) goto cleanup;
    
    threads = (pthread_t *)malloc(sizeof(pthread_t) * nthreads);
    if (!threads) goto cleanup;
    tasks = (annTask *)malloc(sizeof(annTask) * nthreads);
    if (!tasks) goto cleanup;

    // Distribute clusters among threads
    if (distribute_clusters_by_size(Kc, nthreads, cluster_index, tasks)) goto cleanup;

    // Initialize tasks
    for (int i = 0; i < nthreads; ++i) {
        tasks[i].cluster_index = cluster_index;
        tasks[i].L = L;
        tasks[i].K = K;
        tasks[i].C = C;
        tasks[i].D = D;
        tasks[i].IDX = IDX;
        tasks[i].N = N;
        tasks[i].max_memory_usage_ratio = max_memory_usage_ratio;
    }

    switch(par_type) {
        case PAR_PTHREADS:
            if (pthreads_parallelization(tasks, nthreads)) goto cleanup;
            break;
        case PAR_OPENMP:
            if (openmp_parallelization(tasks, nthreads)) goto cleanup;
            break;
        case PAR_OPENCILK:
            if (opencilk_parallelization(tasks, nthreads)) goto cleanup;
            break;
        default:
            fprintf(stderr, "Unknown parallelization type\n");
            goto cleanup;
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
        for (int i = 0; i < nthreads; ++i) {
            if (tasks[i].cluster_ids) free(tasks[i].cluster_ids);
        }
        free(tasks);
    }
    if (threads) free(threads);
    if (assignments) free(assignments);
    if (counts) free(counts);

    return status;
}
