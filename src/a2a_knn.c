#include "a2a_knn.h"
#include "a2a_queue.h"
#include <sys/sysinfo.h>
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>


static pthread_mutex_t mutexQueue;        // Mutex for the tasks Queue
static pthread_cond_t condQueue;          // Condition variable for Queue
static pthread_cond_t condTasksComplete;  // Condition variable to signal task completion for a block
static int isActive;                      // Flag for threads to exit
static int runningTasks;                  // Holds the number of running tasks


/**
 * Thread Task for the exact K-Nearest Neighbors problem
 */
typedef struct knnTask {
    const DTYPE *C;
    const DTYPE *Q;
    DTYPE *D_all_block;
    int *IDX_all_block;
    const DTYPE *sqrmag_C;
    const DTYPE *sqrmag_Q_block;
    const int K;
    const int N;
    const int L;
    const int QUERIES_NUM_THREAD;     // Number of queries for the task to proccess
    const int q_index;                // Index of the first query to be proccessed
    const int q_index_thread;         // Index of the query to be proccesed inside a thread
} knnTask;


static int get_num_threads(int nthreads, const int MAX_QUERIES_MEMORY, const int cblas_nthreads) {

    // Get the number of online processors
    long num_cores = sysconf(_SC_NPROCESSORS_ONLN);
    if (num_cores < 1) {
        perror("sysconf\n");
        num_cores = 1;  // Fallback to 1 if sysconf fails
    }

    // If the number of threads is -1 find automatically the appropriate number of threads,
    nthreads = nthreads < 1 ? (int)num_cores : nthreads;

    // If the number of queries per block is less than the minimum, set nthreads to 1
    nthreads = MAX_QUERIES_MEMORY / nthreads < MIN_QUERIES_PER_BLOCK ? 1 : nthreads;

   nthreads > 1 ? openblas_set_num_threads(1) : openblas_set_num_threads(cblas_nthreads);

    return nthreads;
}


static size_t get_available_memory_bytes() {
    size_t available_memory = 0UL;
    struct sysinfo info;
    if (sysinfo(&info) == 0) {
        available_memory = info.freeram * info.mem_unit;  // Multiply by unit size
    }
    return available_memory;
}



static void swap(DTYPE *arr, int *idx, int i, int j) 
{
    DTYPE dtemp = arr[i];
    arr[i] = arr[j];
    arr[j] = dtemp;
    int lutemp = idx[i];
    idx[i] = idx[j];
    idx[j] = lutemp;
}


static int partition(DTYPE *arr, int *idx, int l, int r) {
    DTYPE pivot = arr[r];
    int i = l;
    for (int j = l; j <= r - 1; j++) {
        if (arr[j] <= pivot) {
            swap(arr, idx, i, j);
            i++;
        }
    }
    swap(arr, idx, i, r);
    return i;
}


static void qselect(DTYPE *arr, int *idx, int l, int r, int k) {
    // Partition the array around the last 
    // element and get the position of the pivot 
    // element in the sorted array
    int index = partition(arr, idx, l, r);

    // If position is the same as k.
    if (index - l == k - 1)
        return;

    // If position is more, recur for the left subarray.
    if (index - l > k - 1) {
        qselect(arr, idx, l, index - 1, k);
        return;
    }

    // Else recur for the right subarray.
    qselect(arr, idx, index + 1, r, k - index + l - 1);
}


static void qsort_(DTYPE *arr, int *idx, int l, int r) {
    if (l < r) {
        // call partition function to find Partition Index
        int index = partition(arr, idx, l, r);

        // Recursively call for left and right
        // half based on Partition Index
        qsort_(arr, idx, l, index - 1);
        qsort_(arr, idx, index + 1, r);
    }
}


static int alloc_memory(DTYPE **D_all_block, int **IDX_all_block, DTYPE **sqrmag_Q_block, DTYPE **sqrmag_C, 
    const int M, const int N, int *MAX_QUERIES_MEMORY, const double max_memory_usage_ratio) {
    size_t available_memory = get_available_memory_bytes();
    size_t max_allocable_memory = (size_t)(available_memory * max_memory_usage_ratio);

    *MAX_QUERIES_MEMORY = M; 
    size_t required_memory = (size_t)(*MAX_QUERIES_MEMORY) * (size_t)N * sizeof(int) +
                                    (size_t)(*MAX_QUERIES_MEMORY) * (size_t)N * sizeof(DTYPE) +
                                    (size_t)(*MAX_QUERIES_MEMORY) * sizeof(DTYPE) +
                                    (size_t)N * sizeof(DTYPE);
    
    if (required_memory > max_allocable_memory) {
        *MAX_QUERIES_MEMORY = (max_allocable_memory - (size_t)N * sizeof(DTYPE)) / 
                            ((size_t)N * sizeof(int) + (size_t)N * sizeof(DTYPE) + sizeof(DTYPE));

        DEBUG_PRINT("KNN: Too large distance matrix. Max queries per block: %d. Using %.2lf%% of available memory\n", *MAX_QUERIES_MEMORY, max_memory_usage_ratio * 100.0);
    }

    if (*MAX_QUERIES_MEMORY < 1) {
        fprintf(stderr, "Error: Insufficient memory for minimum block size.\n");
        return EXIT_FAILURE;
    }


    *IDX_all_block = (int *)malloc((size_t)(*MAX_QUERIES_MEMORY) * (size_t)N * sizeof(int));
    *D_all_block = (DTYPE *)malloc((size_t)(*MAX_QUERIES_MEMORY) * (size_t)N * sizeof(DTYPE));
    *sqrmag_Q_block = (DTYPE *)malloc((size_t)(*MAX_QUERIES_MEMORY) * sizeof(DTYPE));
    *sqrmag_C = (DTYPE *)malloc((size_t)N * sizeof(DTYPE));

    if ((*IDX_all_block) && (*D_all_block) && (*sqrmag_C) && (*sqrmag_Q_block)) {
        return EXIT_SUCCESS;
    }

    if (*sqrmag_C) free(*sqrmag_C);
    if (*sqrmag_Q_block) free(*sqrmag_Q_block);
    if (*D_all_block) free(*D_all_block);
    if (*IDX_all_block) free(*IDX_all_block);

    return EXIT_FAILURE;
}


static void knnTaskExec(const knnTask *task) {
    //DEBUG_PRINT("KNN: Thread %lu executes task with %d queries...\n", pthread_self(), task->QUERIES_NUM_THREAD);
    DTYPE *D_all_block = task->D_all_block;
    int *IDX_all_block = task->IDX_all_block;
    const DTYPE *sqrmag_C = task->sqrmag_C;
    const DTYPE *sqrmag_Q_block = task->sqrmag_Q_block;
    const DTYPE *C = task->C;
    const DTYPE *Q = task->Q;
    const int QUERIES_NUM_THREAD = task->QUERIES_NUM_THREAD;
    const int N = task->N;
    const int K = task->K;
    const int L = task->L;
    const int q_index = task->q_index;
    const int q_index_thread = task->q_index_thread;

    // compute D = -2*Q*C'
    GEMM(CblasRowMajor, CblasNoTrans, CblasTrans, QUERIES_NUM_THREAD, N, L, SUFFIX(-2.0), Q + q_index * L, L, C, L, SUFFIX(0.0), D_all_block + q_index_thread * N, N);

    // compute the distance matrix D by applying the formula D = sqrt(C.^2 -2*Q*C' + (Q.^2)')
    for (int i = 0; i < QUERIES_NUM_THREAD; i++) {
        for (int j = 0; j < N; j++) {
            D_all_block[(i + q_index_thread) * N + j] += sqrmag_Q_block[i + q_index_thread] + sqrmag_C[j];
        }
    }

    // apply Quick Select algorithm for each row of distance matrix
    for (int i = 0; i < QUERIES_NUM_THREAD; i++) {
        qselect(D_all_block + (i + q_index_thread) * N, IDX_all_block + (i + q_index_thread) * N, 0, N - 1, K);
    }

    //DEBUG_PRINT("KNN: Thread %lu finished task with %d queries...\n", pthread_self(), task->QUERIES_NUM_THREAD);
}


static void *knnThreadStart(void *pool) {
    a2a_Queue* queue = (a2a_Queue *)pool;
    knnTask task;

    while (isActive) {
        pthread_mutex_lock(&mutexQueue);
        while (a2a_QueueIsEmpty(queue) && isActive) {
            //DEBUG_PRINT("KNN: Thread %lu waiting...\n", pthread_self());
            pthread_cond_wait(&condQueue, &mutexQueue);
        }
        //DEBUG_PRINT("KNN: Thread %lu woke up...\n", pthread_self());

        if (!isActive)  // Check again after waiting to exit if flag has changed
        {
            pthread_mutex_unlock(&mutexQueue);
            //DEBUG_PRINT("KNN: Thread %lu exiting...\n", pthread_self());
            break;
        }

        a2a_QueueDequeue(queue, (void *)&task);
                
        pthread_mutex_unlock(&mutexQueue);
        
        knnTaskExec(&task);

        pthread_mutex_lock(&mutexQueue);
        runningTasks--;
        //DEBUG_PRINT("KNN: Running tasks: %d\n", runningTasks);
        if (runningTasks == 0) {
            // Signal main thread that all tasks for the current block are done
            //DEBUG_PRINT("KNN: All tasks for current block completed.\n");
            pthread_cond_signal(&condTasksComplete);
        }
        pthread_mutex_unlock(&mutexQueue);
    }

    return NULL;
}


static int check_input_args_knn(const DTYPE* Q, const DTYPE* C, int* IDX, DTYPE* D, 
    const int M, const int N, const int L, const int K, const double cblas_nthreads, 
    const double max_memory_usage_ratio) {
    if (!Q || !C || !IDX || !D) {
        fprintf(stderr, "Error: Null pointer passed to a2a_knnsearch.\n");
        return EXIT_FAILURE;
    }
    if (M <= 0 || N <= 0 || L <= 0 || K <= 0) {
        fprintf(stderr, "Error: Invalid dimensions for a2a_knnsearch (M=%d, N=%d, L=%d, K=%d).\n", M, N, L, K);
        return EXIT_FAILURE;
    }
    if (K > N) {
        fprintf(stderr, "Error: K must be less than or equal to the number of corpus points (K=%d, N=%d).\n", K, N);
        return EXIT_FAILURE;
    }
    if (cblas_nthreads < 1) {
        fprintf(stderr, "Error: Invalid number of OpenBLAS threads (%d).\n", (int)cblas_nthreads);
        return EXIT_FAILURE;
    }
    if (max_memory_usage_ratio <= 0 || max_memory_usage_ratio > 1) {
        fprintf(stderr, "Error: Invalid max memory usage ratio (%f). Must be in (0, 1].\n", max_memory_usage_ratio);
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}


int a2a_knnsearch(const DTYPE* Q, const DTYPE* C, int* IDX, DTYPE* D, const int M, 
    const int N, const int L, const int K, const int sorted, const int nthreads,
    const int cblas_nthreads, const double max_memory_usage_ratio) {

    if (check_input_args_knn(Q, C, IDX, D, M, N, L, K, cblas_nthreads, max_memory_usage_ratio)) {
        return EXIT_FAILURE;
    
    }
    isActive = 1;
    runningTasks = 0;
    DTYPE *D_all_block = NULL, *sqrmag_Q_block = NULL, *sqrmag_C = NULL;
    int *IDX_all_block = NULL;
    int MAX_QUERIES_MEMORY;  // The maximum number of queries that can be stored in memory
    int status = EXIT_FAILURE;

    // Allocate the appropriate amount of memory for the matrices and compute the
    // maximum number of queries that can be proccessed
    if (alloc_memory(&D_all_block, &IDX_all_block, &sqrmag_Q_block, &sqrmag_C, M, N, &MAX_QUERIES_MEMORY, max_memory_usage_ratio)) {
        fprintf(stderr, "knnsearch: Error allocating memory\n");
        return status;
    }

    const int NTHREADS = get_num_threads(nthreads, MAX_QUERIES_MEMORY, cblas_nthreads);

    DEBUG_PRINT("KNN: Running on %d threads (OpenBLAS threads: %d)\n", NTHREADS, openblas_get_num_threads());

    pthread_t* threads = NULL;
    pthread_attr_t attr;
    a2a_Queue tasksQueue;

    // Create the threads if multithreading is desired
    if (NTHREADS > 1) {
        a2a_QueueInit(&tasksQueue, sizeof(knnTask));
        pthread_attr_init(&attr);
        pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);
        pthread_mutex_init(&mutexQueue, NULL);
        pthread_cond_init(&condQueue, NULL);
        pthread_cond_init(&condTasksComplete, NULL);

        threads = (pthread_t *)malloc(sizeof(pthread_t) * NTHREADS);
        if (!threads) {
            fprintf(stderr, "Error allocating memory for threads\n");
            goto cleanup;
        }

        for (int t = 0; t < NTHREADS; t++) {
            if (pthread_create(&threads[t], NULL, knnThreadStart, (void *)&tasksQueue)) {
                fprintf(stderr, "Error creating thread %d\n", t + 1);
                goto cleanup;
            }
        }
    }

    // Pre compute the square of magnitudes of the row vectors of matrix C
    // since it is shared accross threads
    for (int i = 0; i < N; i++) {
        sqrmag_C[i] = DOT(L, C + i * L, 1, C + i * L, 1);
    }


    // Iterate through each block of queries
    int q_index = 0;
    while (q_index < M) {
        // number of queries per block
        const int QUERIES_NUM_BLOCK = (M - q_index) > MAX_QUERIES_MEMORY ? MAX_QUERIES_MEMORY : (M - q_index);

        // initialize index matrix
        for (int i = 0; i < QUERIES_NUM_BLOCK; i++) {
            for (int j = 0; j < N; j++) {
                IDX_all_block[i * N + j] = j;
            }
        }

        // Compute the square of magnitudes of the row vectors of matrix Q for this block
        for (int i = 0; i < QUERIES_NUM_BLOCK; i++) {
            sqrmag_Q_block[i] = DOT(L, Q + (i + q_index) * L, 1, Q + (i + q_index) * L, 1);
        }
        
        DEBUG_PRINT("KNN: Processing block with %d queries (using %.2lf%% of available memory)\n", QUERIES_NUM_BLOCK, max_memory_usage_ratio * 100.0);
        if (NTHREADS == 1) { // no multithreading
            knnTask task = {
                .C = C, 
                .Q = Q, 
                .D_all_block = D_all_block,
                .IDX_all_block = IDX_all_block,
                .QUERIES_NUM_THREAD = QUERIES_NUM_BLOCK,
                .sqrmag_C = sqrmag_C,
                .sqrmag_Q_block = sqrmag_Q_block,
                .N = N,
                .L = L,
                .K = K,
                .q_index = q_index,
                .q_index_thread = 0
            };
            knnTaskExec(&task);
        }
        else { // multithreading
            // Create the tasks and add them to the queue
            if (QUERIES_NUM_BLOCK / NTHREADS == 0) { // create only one task
                knnTask task = {
                    .C = C, 
                    .Q = Q, 
                    .D_all_block = D_all_block,
                    .IDX_all_block = IDX_all_block,
                    .QUERIES_NUM_THREAD = QUERIES_NUM_BLOCK,
                    .sqrmag_C = sqrmag_C,
                    .sqrmag_Q_block = sqrmag_Q_block,
                    .N = N,
                    .L = L,
                    .K = K,
                    .q_index = q_index,
                    .q_index_thread = 0
                };
                a2a_QueueEnqueue(&tasksQueue, (void *)&task);  // add task to the queue
                runningTasks++;
            }
            else { // split workload accross all the threads
                int remainder = QUERIES_NUM_BLOCK % NTHREADS;
                int QUERIES_NUM_THREAD = 0;  // number of queries being proccesed on each thread
                int q_index_thread = 0;  // indexing of the queries in each block of queries
                for (int t = 0; t < NTHREADS; t++) {
                    QUERIES_NUM_THREAD = QUERIES_NUM_BLOCK / NTHREADS;
                    if (remainder > 0) {
                        remainder--;
                        QUERIES_NUM_THREAD++;
                    }

                    knnTask task = {
                        .C = C, 
                        .Q = Q, 
                        .D_all_block = D_all_block,
                        .IDX_all_block = IDX_all_block,
                        .QUERIES_NUM_THREAD = QUERIES_NUM_THREAD,
                        .sqrmag_C = sqrmag_C,
                        .sqrmag_Q_block = sqrmag_Q_block,
                        .N = N,
                        .L = L,
                        .K = K,
                        .q_index = q_index_thread + q_index,
                        .q_index_thread = q_index_thread
                    };
                    a2a_QueueEnqueue(&tasksQueue, (void *)&task);  // add task to the queue
                    runningTasks++;
                    q_index_thread += QUERIES_NUM_THREAD;
                }
            }

            pthread_cond_broadcast(&condQueue);  // Wake up all threads to assign them the tasks
                
            // Wait for all tasks in the current block to finish
            pthread_mutex_lock(&mutexQueue);
            //DEBUG_PRINT("KNN: Waiting for %d tasks to complete...\n", runningTasks);
            while (runningTasks > 0) {
                pthread_cond_wait(&condTasksComplete, &mutexQueue);
            }
            pthread_mutex_unlock(&mutexQueue);
        }

        // now copy the first K elements of each row of matrices
        // D_all_block, IDX_all_block to D and IDX respectivelly
        for (int i = 0; i < QUERIES_NUM_BLOCK; i++) {
            for (int j = 0; j < K; j++) {
                D[(q_index + i) * K + j] = SQRT(D_all_block[i * N + j]);
                IDX[(q_index + i) * K + j] = IDX_all_block[i * N + j];  // zero-based indexing
            }

            // sort each row of the distance matrix
            if (sorted) {
                qsort_(D + (q_index + i) * K, IDX + (q_index + i) * K, 0, K - 1);
            }
        }

        q_index += QUERIES_NUM_BLOCK;  // move to the next block of queries
    }

    if (NTHREADS > 1) {
        isActive = 0;  // Set termination flag for threads
        pthread_cond_broadcast(&condQueue);  // Wake up all threads to allow them to exit

        for (int t = 0; t < NTHREADS; t++) {
            if (pthread_join(threads[t], NULL)) {
                fprintf(stderr, "Failed to join thread %d\n", t);
                goto cleanup;
            }
        }
    }

    status = EXIT_SUCCESS;

cleanup:
    if (NTHREADS > 1) {
        a2a_QueueDestroy(&tasksQueue);
        pthread_attr_destroy(&attr);
        pthread_mutex_destroy(&mutexQueue);
        pthread_cond_destroy(&condQueue);
        pthread_cond_destroy(&condTasksComplete);
        if (threads) free(threads);
    }
    free(sqrmag_C);
    free(sqrmag_Q_block);
    free(D_all_block);
    free(IDX_all_block);
    return status;
}
