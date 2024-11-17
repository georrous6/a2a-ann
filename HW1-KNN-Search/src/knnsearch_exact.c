#include "knnsearch_exact.h"
#include "Queue.h"
#include <stdio.h>
#include <stdlib.h>
#include <cblas.h>
#include <math.h>
#include <unistd.h>
#include <sys/sysinfo.h>
#include <pthread.h>


pthread_mutex_t mutexQueue;
pthread_cond_t condQueue;
pthread_cond_t condTasksComplete;
int isActive;
int runningTasks;


void swap(double *arr, int *idx, int i, int j) 
{
    double dtemp = arr[i];
    arr[i] = arr[j];
    arr[j] = dtemp;
    int lutemp = idx[i];
    idx[i] = idx[j];
    idx[j] = lutemp;
}


int partition(double *arr, int *idx, int l, int r) 
{
    double pivot = arr[r];
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


void qselect(double *arr, int *idx, int l, int r, int k) 
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


void qsort_(double *arr, int *idx, int l, int r) 
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


int get_num_threads(int MBLOCK_MAX_SIZE, int N)
{
    const long num_cores = sysconf(_SC_NPROCESSORS_ONLN);  // Number of online processors
    if (num_cores < 1) 
    {
        perror("sysconf\n");
        return 1;
    }

    int queries_per_block = MBLOCK_MAX_SIZE / (int)num_cores;

    return queries_per_block >= MIN_THREAD_QUERIES_SIZE && N >= MIN_THREAD_CORPUS_SIZE ? (int)num_cores : 1;
}


unsigned long get_available_memory_bytes() 
{
    unsigned long available_memory = 0;
    struct sysinfo info;
    if (sysinfo(&info) == 0) 
    {
        available_memory = info.freeram * info.mem_unit;  // Multiply by unit size
    }
    return available_memory;
}


int alloc_memory(double **Dall, int **IDXall, double **sqrmag_Q, double **sqrmag_C, const int M, const int N, int *MBLOCK_MAX_SIZE)
{
    unsigned long available_memory = get_available_memory_bytes();
    unsigned long max_allocable_memory = (unsigned long)(available_memory * MAX_MEMORY_USAGE_RATIO);

    *MBLOCK_MAX_SIZE = M; 
    unsigned long required_memory = (*MBLOCK_MAX_SIZE) * N * sizeof(int) +
                                    (*MBLOCK_MAX_SIZE) * N * sizeof(double) +
                                    (*MBLOCK_MAX_SIZE) * sizeof(double) +
                                    N * sizeof(double);
    
    if (required_memory > max_allocable_memory)
    {
        *MBLOCK_MAX_SIZE = (max_allocable_memory - N * sizeof(double)) / 
                            (N * sizeof(int) + N * sizeof(double) + sizeof(double));
    }

    if (*MBLOCK_MAX_SIZE < 1) 
    {
        fprintf(stderr, "Error: Insufficient memory for minimum block size.\n");
        return EXIT_FAILURE;
    }



    *IDXall = (int *)malloc((*MBLOCK_MAX_SIZE) * N * sizeof(int));
    *Dall = (double *)malloc((*MBLOCK_MAX_SIZE) * N * sizeof(double));
    *sqrmag_Q = (double *)malloc((*MBLOCK_MAX_SIZE) * sizeof(double));
    *sqrmag_C = (double *)malloc(N * sizeof(double));

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


void executeKNNExactTask(const KNNExactTask *task)
{
    double *Dall = task->Dall;
    int *IDXall = task->IDXall;
    int MBLOCK_THREAD_SIZE = task->MBLOCK_THREAD_SIZE;
    int N = task->N;
    int K = task->K;
    int sorted = task->sorted;
    if (!sorted)
    {
        // apply Quick Select algorithm for each row of distance matrix
        for (int i = 0; i < MBLOCK_THREAD_SIZE; i++)
        {
            qselect(Dall + i * N, IDXall + i * N, 0, N - 1, K);
        }
    }
    else
    {
        // apply Quick Sort algorithm for each row of distance matrix
        for (int i = 0; i < MBLOCK_THREAD_SIZE; i++)
        {
            qsort_(Dall + i * N, IDXall + i * N, 0, N - 1);
        }      
    }
}


int knnsearch_exact(const double* Q, const double* C, int* IDX, double* D, const int M, const int N, const int L, const int K, const int sorted, int nthreads)
{
    isActive = 1;
    runningTasks = 0;
    double *Dall = NULL, *sqrmag_Q = NULL, *sqrmag_C = NULL;
    int *IDXall = NULL;
    int MBLOCK_MAX_SIZE, NBLOCK_MAX_SIZE;
    int status = EXIT_FAILURE;
    if (K <= 0)
    {
        fprintf(stderr, "Invalid value for K: %d\n", K);
        return status;
    }

    // Allocate the appropriate amount of memory for the matrices and compute the
    // maximum coprus size that fits in memory
    if (alloc_memory(&Dall, &IDXall, &sqrmag_Q, &sqrmag_C, M, N, &MBLOCK_MAX_SIZE))
    {
        fprintf(stderr, "knnsearch_exact: Error allocating memory\n");
        return status;
    }

    // if the number of threads is -1 find automatically the appropriate number of threads, 
    // otherwise use the number of threads the user passed explicitly
    nthreads = nthreads == -1 ? get_num_threads(MBLOCK_MAX_SIZE, N) : nthreads;

    pthread_t* threads = NULL;
    pthread_attr_t attr;
    Queue tasksQueue;

    // Create the threads if multithreading is desired
    if (nthreads > 1)
    {
        Queue_init(&tasksQueue, sizeof(KNNExactTask));
        pthread_attr_init(&attr);
        pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);
        pthread_mutex_init(&mutexQueue, NULL);
        pthread_cond_init(&condQueue, NULL);
        pthread_cond_init(&condTasksComplete, NULL);

        threads = (pthread_t *)malloc(sizeof(pthread_t) * nthreads);
        if (!threads)
        {
            fprintf(stderr, "Error allocating memory for threads\n");
            goto cleanup;
        }

        for (int t = 0; t < nthreads; t++)
        {
            if (pthread_create(&threads[t], NULL, startKNNExactThread, (void *)&tasksQueue))
            {
                fprintf(stderr, "Error creating thread %d\n", t + 1);
                goto cleanup;
            }
        }
    }


    for (int M_INDEX = 0; M_INDEX * MBLOCK_MAX_SIZE < M; M_INDEX++) 
    {
        const int MBLOCK_SIZE = (M - M_INDEX * MBLOCK_MAX_SIZE) > MBLOCK_MAX_SIZE ? MBLOCK_MAX_SIZE : (M - M_INDEX * MBLOCK_MAX_SIZE);
        const int Q_OFFSET = M_INDEX * MBLOCK_MAX_SIZE * L;
        const int D_OFFSET = M_INDEX * MBLOCK_MAX_SIZE * K;

        // initialize index matrix
        for (int i = 0; i < MBLOCK_SIZE; i++)
        {
            for (int j = 0; j < N; j++)
            {
                IDXall[i * N + j] = j;
            }
        }


        // compute D = -2*Q*C'
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, MBLOCK_SIZE, N, L, -2.0, Q + Q_OFFSET, L, C, L, 0.0, Dall, N);

        // compute the square of magnitudes of the row vectors of matrix Q
        for (int i = 0; i < MBLOCK_SIZE; i++)
        {
            sqrmag_Q[i] = cblas_ddot(L, Q + i * L + Q_OFFSET, 1, Q + i * L + Q_OFFSET, 1);
        }

        // compute the square of magnitudes of the row vectors of matrix C
        for (int i = 0; i < N; i++)
        {
            sqrmag_C[i] = cblas_ddot(L, C + i * L, 1, C + i * L, 1);
        }

        // compute the distance matrix D by applying the formula D = sqrt(C.^2 -2*Q*C' + (Q.^2)')
        for (int i = 0; i < MBLOCK_SIZE; i++)
        {
            for (int j = 0; j < N; j++)
            {
                Dall[i * N + j] = sqrt(Dall[i * N + j] + sqrmag_Q[i] + sqrmag_C[j]);
            }
        }
        
        if (nthreads == 1)  // no multithreading
        {
            KNNExactTask task = (KNNExactTask){Dall, IDXall, K, N, MBLOCK_SIZE, sorted};
            executeKNNExactTask(&task);
        }
        else  // multithreading
        {
            // Create the tasks and add them to the queue
            if (MBLOCK_SIZE / nthreads == 0)  // create only one task
            {
                KNNExactTask task = (KNNExactTask){Dall, IDXall, K, N, MBLOCK_SIZE, sorted};
                Queue_enqueue(&tasksQueue, (void *)&task);  // add task to the queue
                runningTasks++;
            }
            else  // split workload accross all the threads
            {
                int remainder = MBLOCK_SIZE % nthreads;
                int MBLOCK_THREAD_SIZE = 0, D_THREAD_OFFSET = 0;
                for (int t = 0; t < nthreads; t++)
                {
                    D_THREAD_OFFSET += MBLOCK_THREAD_SIZE * N;
                    MBLOCK_THREAD_SIZE = MBLOCK_SIZE / nthreads;
                    if (remainder > 0)
                    {
                        remainder--;
                        MBLOCK_THREAD_SIZE++;
                    }

                    KNNExactTask task = (KNNExactTask){Dall + D_THREAD_OFFSET, IDXall + D_THREAD_OFFSET, K, N, MBLOCK_THREAD_SIZE, sorted};
                    Queue_enqueue(&tasksQueue, (void *)&task);  // add task to the queue
                    runningTasks++;
                }
            }

            pthread_cond_broadcast(&condQueue);  // Wake up all threads to assign them the tasks
                
            // Wait for all tasks in the current block to finish
            pthread_mutex_lock(&mutexQueue);
            while (runningTasks > 0)
            {
                pthread_cond_wait(&condTasksComplete, &mutexQueue);
            }
            pthread_mutex_unlock(&mutexQueue);
        }

        // now copy the first K elements of each row of matrices
        // Dall, IDXall to D and IDX respectivelly
        for (int i = 0; i < MBLOCK_SIZE; i++)
        {
            for (int j = 0; j < K; j++)
            {
                D[D_OFFSET + i * K + j] = Dall[i * N + j];
                IDX[D_OFFSET + i * K + j] = IDXall[i * N + j] + 1; // 1-based indexing
            }
        }
    }

    if (nthreads > 1)
    {
        isActive = 0;  // Set termination flag for threads
        pthread_cond_broadcast(&condQueue);  // Wake up all threads to allow them to exit

        for (int t = 0; t < nthreads; t++)
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
    if (nthreads > 1)
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


void *startKNNExactThread(void *pool)
{
    Queue* queue = (Queue *)pool;
    KNNExactTask task;

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
        
        executeKNNExactTask(&task);

        pthread_mutex_lock(&mutexQueue);
        runningTasks--;
        if (runningTasks == 0)
        {
            // Signal main thread that all tasks for the current block are done
            pthread_cond_signal(&condTasksComplete);
        }
        pthread_mutex_unlock(&mutexQueue);
    }

    return NULL;
}

