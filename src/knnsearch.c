#include "knnsearch.h"
#include "ann_config.h"
#include "Queue.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <pthread.h>
#include <cblas.h>


pthread_mutex_t mutexQueue;
pthread_cond_t condQueue;
pthread_cond_t condTasksComplete;
int isActive;
int runningTasks;


void swap(DTYPE *arr, int *idx, int i, int j) 
{
    DTYPE dtemp = arr[i];
    arr[i] = arr[j];
    arr[j] = dtemp;
    int lutemp = idx[i];
    idx[i] = idx[j];
    idx[j] = lutemp;
}


int partition(DTYPE *arr, int *idx, int l, int r) 
{
    DTYPE pivot = arr[r];
    int i = l;
    for (int j = l; j <= r - 1; j++) 
    {
        if (arr[j] <= pivot) 
        {
            swap(arr, idx, i, j);
            i++;
        }
    }
    swap(arr, idx, i, r);
    return i;
}


void qselect(DTYPE *arr, int *idx, int l, int r, int k) 
{
    // Partition the array around the last 
    // element and get the position of the pivot 
    // element in the sorted array
    int index = partition(arr, idx, l, r);

    // If position is the same as k.
    if (index - l == k - 1)
        return;

    // If position is more, recur for the left subarray.
    if (index - l > k - 1)
    {
        qselect(arr, idx, l, index - 1, k);
        return;
    }

    // Else recur for the right subarray.
    qselect(arr, idx, index + 1, r, k - index + l - 1);
}


void qsort_(DTYPE *arr, int *idx, int l, int r) 
{
    if (l < r) 
    {
        // call partition function to find Partition Index
        int index = partition(arr, idx, l, r);

        // Recursively call for left and right
        // half based on Partition Index
        qsort_(arr, idx, l, index - 1);
        qsort_(arr, idx, index + 1, r);
    }
}


void knnTaskExec(const knnTask *task)
{
    //printf("Thread executes task with %d queries...\n", task->QUERIES_NUM);
    DTYPE *Dall = task->Dall;
    int *IDXall = task->IDXall;
    const DTYPE *sqrmag_C = task->sqrmag_C;
    const DTYPE *sqrmag_Q = task->sqrmag_Q;
    const DTYPE *C = task->C;
    const DTYPE *Q = task->Q;
    const int QUERIES_NUM = task->QUERIES_NUM;
    const int N = task->N;
    const int K = task->K;
    const int L = task->L;
    const int q_index = task->q_index;
    const int q_index_thread = task->q_index_thread;

    // compute D = -2*Q*C'
    GEMM(CblasRowMajor, CblasNoTrans, CblasTrans, QUERIES_NUM, N, L, -2.0, Q + q_index * L, L, C, L, 0.0, Dall + q_index_thread * N, N);

    // compute the distance matrix D by applying the formula D = sqrt(C.^2 -2*Q*C' + (Q.^2)')
    for (int i = 0; i < QUERIES_NUM; i++)
    {
        for (int j = 0; j < N; j++)
        {
            Dall[(i + q_index_thread) * N + j] += sqrmag_Q[i + q_index] + sqrmag_C[j];
        }
    }

    // apply Quick Select algorithm for each row of distance matrix
    for (int i = 0; i < QUERIES_NUM; i++)
    {
        qselect(Dall + (i + q_index_thread) * N, IDXall + (i + q_index_thread) * N, 0, N - 1, K);
    }

    //printf("Thread finished task with %d queries.\n", task->QUERIES_NUM);
}


void *knnThreadStart(void *pool)
{
    Queue* queue = (Queue *)pool;
    knnTask task;

    while (isActive)
    {
        pthread_mutex_lock(&mutexQueue);
        while (Queue_isEmpty(queue) && isActive)
        {
            pthread_cond_wait(&condQueue, &mutexQueue);
        }

        if (!isActive)  // Check again after waiting to exit if flag has changed
        {
            pthread_mutex_unlock(&mutexQueue);
            break;
        }

        Queue_dequeue(queue, (void *)&task);
                
        pthread_mutex_unlock(&mutexQueue);
        
        knnTaskExec(&task);

        pthread_mutex_lock(&mutexQueue);
        runningTasks--;
        //printf("Running tasks: %d\n", runningTasks);
        if (runningTasks == 0)
        {
            // Signal main thread that all tasks for the current block are done
            //printf("All tasks for current block completed.\n");
            pthread_cond_signal(&condTasksComplete);
        }
        pthread_mutex_unlock(&mutexQueue);
    }

    return NULL;
}


int alloc_memory(DTYPE **Dall, int **IDXall, DTYPE **sqrmag_Q, DTYPE **sqrmag_C, const int M, const int N, int *MBLOCK_MAX_SIZE)
{
    unsigned long available_memory = get_available_memory_bytes();
    unsigned long max_allocable_memory = (unsigned long)(available_memory * MAX_MEMORY_USAGE_RATIO);

    *MBLOCK_MAX_SIZE = M; 
    unsigned long required_memory = (*MBLOCK_MAX_SIZE) * N * sizeof(int) +
                                    (*MBLOCK_MAX_SIZE) * N * sizeof(DTYPE) +
                                    (*MBLOCK_MAX_SIZE) * sizeof(DTYPE) +
                                    N * sizeof(DTYPE);
    
    if (required_memory > max_allocable_memory)
    {
        *MBLOCK_MAX_SIZE = (max_allocable_memory - N * sizeof(DTYPE)) / 
                            (N * sizeof(int) + N * sizeof(DTYPE) + sizeof(DTYPE));

        printf("Too large distance matrix. Max queries per block: %d\n", *MBLOCK_MAX_SIZE);
    }

    if (*MBLOCK_MAX_SIZE < 1) 
    {
        fprintf(stderr, "Error: Insufficient memory for minimum block size.\n");
        return EXIT_FAILURE;
    }



    *IDXall = (int *)malloc((*MBLOCK_MAX_SIZE) * N * sizeof(int));
    *Dall = (DTYPE *)malloc((*MBLOCK_MAX_SIZE) * N * sizeof(DTYPE));
    *sqrmag_Q = (DTYPE *)malloc((*MBLOCK_MAX_SIZE) * sizeof(DTYPE));
    *sqrmag_C = (DTYPE *)malloc(N * sizeof(DTYPE));

    if ((*IDXall) && (*Dall) && (*sqrmag_C) && (*sqrmag_Q))
    {
        return EXIT_SUCCESS;
    }

    if (*sqrmag_C) free(*sqrmag_C);
    if (*sqrmag_Q) free(*sqrmag_Q);
    if (*Dall) free(*Dall);
    if (*IDXall) free(*IDXall);

    return EXIT_FAILURE;
}


int knnsearch(const DTYPE* Q, const DTYPE* C, int* IDX, DTYPE* D, const int M, const int N, const int L, const int K, const int sorted)
{
    isActive = 1;
    runningTasks = 0;
    DTYPE *Dall = NULL, *sqrmag_Q = NULL, *sqrmag_C = NULL;
    int *IDXall = NULL;
    int MAX_QUERIES_MEMORY;  // The maximum number of queries that can be stored in memory
    int status = EXIT_FAILURE;
    if (K <= 0)
    {
        fprintf(stderr, "Invalid value for K: %d\n", K);
        return status;
    }

    // Allocate the appropriate amount of memory for the matrices and compute the
    // maximum number of queries that can be proccessed
    if (alloc_memory(&Dall, &IDXall, &sqrmag_Q, &sqrmag_C, M, N, &MAX_QUERIES_MEMORY))
    {
        fprintf(stderr, "knnsearch: Error allocating memory\n");
        return status;
    }

    // if the number of threads is -1 find automatically the appropriate number of threads, 
    // otherwise use the number of threads the user passed explicitly
    int nthreads = ann_get_num_threads();
    if (MAX_QUERIES_MEMORY / nthreads < MIN_QUERIES_PER_BLOCK) {
        ann_set_num_threads(1);
    }
    const int NTHREADS = ann_get_num_threads();

    pthread_t* threads = NULL;
    pthread_attr_t attr;
    Queue tasksQueue;

    // Create the threads if multithreading is desired
    if (NTHREADS > 1)
    {
        Queue_init(&tasksQueue, sizeof(knnTask));
        pthread_attr_init(&attr);
        pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);
        pthread_mutex_init(&mutexQueue, NULL);
        pthread_cond_init(&condQueue, NULL);
        pthread_cond_init(&condTasksComplete, NULL);

        threads = (pthread_t *)malloc(sizeof(pthread_t) * NTHREADS);
        if (!threads)
        {
            fprintf(stderr, "Error allocating memory for threads\n");
            goto cleanup;
        }

        for (int t = 0; t < NTHREADS; t++)
        {
            if (pthread_create(&threads[t], NULL, knnThreadStart, (void *)&tasksQueue))
            {
                fprintf(stderr, "Error creating thread %d\n", t + 1);
                goto cleanup;
            }
        }
    }


    // Iterate through each block of queries
    int q_index = 0;
    while (q_index < M) 
    {
        const int QUERIES_NUM = (M - q_index) > MAX_QUERIES_MEMORY ? MAX_QUERIES_MEMORY : (M - q_index);

        // initialize index matrix
        for (int i = 0; i < QUERIES_NUM; i++)
        {
            for (int j = 0; j < N; j++)
            {
                IDXall[i * N + j] = j;
            }
        }

        // Pre compute the square of magnitudes of the row vectors of matrix C
        // since it is shared accross threads
        for (int i = 0; i < N; i++)
        {
            sqrmag_C[i] = DOT(L, C + i * L, 1, C + i * L, 1);
        }

        // Pre compute the square of magnitudes of the row vectors of matrix Q
        for (int i = 0; i < QUERIES_NUM; i++)
        {
            sqrmag_Q[i] = DOT(L, Q + (i + q_index) * L, 1, Q + (i + q_index) * L, 1);
        }
        
        printf("\nThreads: %d (OpenBLAS threads: %d)\n", NTHREADS, openblas_get_num_threads());
        if (NTHREADS == 1)  // no multithreading
        {
            knnTask task = {
                .C = C, 
                .Q = Q, 
                .Dall = Dall,
                .IDXall = IDXall,
                .QUERIES_NUM = QUERIES_NUM,
                .sqrmag_C = sqrmag_C,
                .sqrmag_Q = sqrmag_Q,
                .N = N,
                .L = L,
                .K = K,
                .q_index = q_index,
                .q_index_thread = 0
            };
            knnTaskExec(&task);
        }
        else  // multithreading
        {
            // Create the tasks and add them to the queue
            if (QUERIES_NUM / NTHREADS == 0)  // create only one task
            {
                knnTask task = {
                    .C = C, 
                    .Q = Q, 
                    .Dall = Dall,
                    .IDXall = IDXall,
                    .QUERIES_NUM = QUERIES_NUM,
                    .sqrmag_C = sqrmag_C,
                    .sqrmag_Q = sqrmag_Q,
                    .N = N,
                    .L = L,
                    .K = K,
                    .q_index = q_index,
                    .q_index_thread = 0
                };
                Queue_enqueue(&tasksQueue, (void *)&task);  // add task to the queue
                runningTasks++;
            }
            else  // split workload accross all the threads
            {
                int remainder = QUERIES_NUM % NTHREADS;
                int QUERIES_NUM_THREAD = 0;  // number of queries being proccesed on each thread
                int q_index_thread = 0;  // indexing of the queries in each block of queries
                for (int t = 0; t < NTHREADS; t++)
                {
                    QUERIES_NUM_THREAD = QUERIES_NUM / NTHREADS;
                    if (remainder > 0)
                    {
                        remainder--;
                        QUERIES_NUM_THREAD++;
                    }

                    knnTask task = {
                        .C = C, 
                        .Q = Q, 
                        .Dall = Dall,
                        .IDXall = IDXall,
                        .QUERIES_NUM = QUERIES_NUM_THREAD,
                        .sqrmag_C = sqrmag_C,
                        .sqrmag_Q = sqrmag_Q,
                        .N = N,
                        .L = L,
                        .K = K,
                        .q_index = q_index_thread + q_index,
                        .q_index_thread = q_index_thread
                    };
                    Queue_enqueue(&tasksQueue, (void *)&task);  // add task to the queue
                    runningTasks++;
                    q_index_thread += QUERIES_NUM_THREAD;
                }
            }

            pthread_cond_broadcast(&condQueue);  // Wake up all threads to assign them the tasks
                
            // Wait for all tasks in the current block to finish
            pthread_mutex_lock(&mutexQueue);
            //printf("Waiting for %d tasks to complete...\n", runningTasks);
            while (runningTasks > 0)
            {
                pthread_cond_wait(&condTasksComplete, &mutexQueue);
            }
            pthread_mutex_unlock(&mutexQueue);
        }

        // now copy the first K elements of each row of matrices
        // Dall, IDXall to D and IDX respectivelly
        for (int i = 0; i < QUERIES_NUM; i++)
        {
            for (int j = 0; j < K; j++)
            {
                D[(q_index + i) * K + j] = SQRT(Dall[i * N + j]);
                IDX[(q_index + i) * K + j] = IDXall[i * N + j];  // zero-based indexing
            }

            // sort each row of the distance matrix
            if (sorted)
            {
                qsort_(D + (q_index + i) * K, IDX + (q_index + i) * K, 0, K - 1);
            }
        }

        q_index += QUERIES_NUM;
    }

    if (NTHREADS > 1)
    {
        isActive = 0;  // Set termination flag for threads
        pthread_cond_broadcast(&condQueue);  // Wake up all threads to allow them to exit

        for (int t = 0; t < NTHREADS; t++)
        {
            if (pthread_join(threads[t], NULL))
            {
                fprintf(stderr, "Failed to join thread %d\n", t);
                goto cleanup;
            }
        }
    }

    status = EXIT_SUCCESS;

cleanup:
    if (NTHREADS > 1)
    {
        Queue_destroy(&tasksQueue);
        pthread_attr_destroy(&attr);
        pthread_mutex_destroy(&mutexQueue);
        pthread_cond_destroy(&condQueue);
        pthread_cond_destroy(&condTasksComplete);
        if (threads) free(threads);
    }
    free(sqrmag_C);
    free(sqrmag_Q);
    free(Dall);
    free(IDXall);
    return status;
}
