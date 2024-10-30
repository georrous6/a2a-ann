#include "knnsearch.h"
#include <stdio.h>
#include <stdlib.h>
#include <cblas.h>
#include <math.h>
#include <time.h>
#include <unistd.h>
#include <sys/sysinfo.h>
#include <pthread.h>


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


int get_thread_count(int MBLOCK_SIZE, int N)
{
    const long num_cores = sysconf(_SC_NPROCESSORS_ONLN);  // Number of online processors
    if (num_cores < 1) 
    {
        perror("sysconf\n");
        return 1;
    }

    if (MBLOCK_SIZE < 1) return 1;
    if (MBLOCK_SIZE <= (int)num_cores) return MBLOCK_SIZE;
    return (int)num_cores;
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
    int i = 0;
    *MBLOCK_MAX_SIZE = M;

    while (i < ALLOC_MAX_ITERS && *MBLOCK_MAX_SIZE > 0)
    {
        unsigned long required_memory = (*MBLOCK_MAX_SIZE) * N * sizeof(int) +
                                        (*MBLOCK_MAX_SIZE) * N * sizeof(double) +
                                        (*MBLOCK_MAX_SIZE) * sizeof(double) +
                                        N * sizeof(double);


        if (required_memory > max_allocable_memory) 
        {
            // If memory exceeds limit, halve the block size and retry
            *MBLOCK_MAX_SIZE = (*MBLOCK_MAX_SIZE) / 2 + 1;
            i++;
            continue;
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

        // adjust the block size and retry
        *MBLOCK_MAX_SIZE = (*MBLOCK_MAX_SIZE) / 2 + 1;
        i++;
    }

    return EXIT_FAILURE;
}


void *qfunc(void *args)
{
    double *Dall = ((THREAD_ARGS *) args)->Dall;
    int *IDXall = ((THREAD_ARGS *) args)->IDXall;
    int MBLOCK_THREAD_SIZE = ((THREAD_ARGS *) args)->MBLOCK_THREAD_SIZE;
    int N = ((THREAD_ARGS *) args)->N;
    int K = ((THREAD_ARGS *) args)->K;
    int sorted = ((THREAD_ARGS *) args)->sorted;
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

    pthread_exit(NULL);
}


int knnsearch_exact(const double* Q, const double* C, int* IDX, double* D, const int M, const int N, const int L, int K, const int sorted)
{
    double *Dall = NULL, *sqrmag_Q = NULL, *sqrmag_C = NULL;
    int *IDXall = NULL;
    int MBLOCK_MAX_SIZE, NBLOCK_MAX_SIZE;
    int status = EXIT_FAILURE;
    if (K <= 0)
    {
        fprintf(stderr, "Invalid value for K: %d\n", K);
        return status;
    }

    K = K > N ? N : K;  // The k-nearest points are greater than the corpus size

    clock_t start = clock();

    if (alloc_memory(&Dall, &IDXall, &sqrmag_Q, &sqrmag_C, M, N, &MBLOCK_MAX_SIZE))
    {
        fprintf(stderr, "knnsearch_exact: Error allocating memory\n");
        return status;
    }

    clock_t end = clock();
    //printf("Ellapsed time (memory allocation): %lf\n", ((double) (end - start)) / CLOCKS_PER_SEC);
    printf("MBLOCK_MAX_SIZE: %d\n", MBLOCK_MAX_SIZE);


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
        start = clock();
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, MBLOCK_SIZE, N, L, -2.0, Q + Q_OFFSET, L, C, L, 0.0, Dall, N);
        end = clock();
        //printf("Ellapsed time (matrix multiplication): %lf\n", ((double) (end - start)) / CLOCKS_PER_SEC);

        start = clock();
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
        end = clock();
        //printf("Ellapsed time (distance calculation): %lf\n", ((double) (end - start)) / CLOCKS_PER_SEC);

        start = clock();
        const int NTHREADS = get_thread_count(MBLOCK_SIZE, N);
        if (NTHREADS == 1)  // no multithreading
        {
            if (!sorted)
            {
                // apply Quick Select algorithm for each row of distance matrix
                for (int i = 0; i < MBLOCK_SIZE; i++)
                {
                    qselect(Dall + i * N, IDXall + i * N, 0, N - 1, K);
                }
            }
            else
            {
                // apply Quick Sort algorithm for each row of distance matrix
                for (int i = 0; i < MBLOCK_SIZE; i++)
                {
                    qsort_(Dall + i * N, IDXall + i * N, 0, N - 1);
                }      
            }
        }
        else  // multithreading
        {
            printf("Threads No: %d\n", NTHREADS);
            pthread_attr_t attr;
            pthread_t *threads = (pthread_t *)malloc(sizeof(pthread_t) * NTHREADS);
            if (!threads)
            {
                fprintf(stderr, "Error allocating memory for threads\n");
                goto cleanup;
            }

            THREAD_ARGS *thread_args = malloc(NTHREADS * sizeof(THREAD_ARGS));
            if (!thread_args) 
            {
                fprintf(stderr, "Error allocating memory for thread arguments\n");
                free(threads);
                goto cleanup;
            }

            pthread_attr_init(&attr);
            pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);

            int remainder = MBLOCK_SIZE % NTHREADS;
            int MBLOCK_THREAD_SIZE = 0, D_THREAD_OFFSET = 0, rc;
            for (int t = 0; t < NTHREADS; t++)
            {
                D_THREAD_OFFSET += MBLOCK_THREAD_SIZE * N;
                MBLOCK_THREAD_SIZE = MBLOCK_SIZE / NTHREADS;
                if (remainder > 0)
                {
                    remainder--;
                    MBLOCK_THREAD_SIZE++;
                }

                thread_args[t] = (THREAD_ARGS){Dall + D_THREAD_OFFSET, IDXall + D_THREAD_OFFSET, K, N, MBLOCK_THREAD_SIZE, sorted};
                rc = pthread_create(&threads[t], NULL, qfunc, &thread_args[t]);
                if (rc)
                {
                    fprintf(stderr, "ERROR; return code from pthread_create() is %d\n", rc);
                    free(threads);
                    free(thread_args);
                    goto cleanup;
                }
            }

        	pthread_attr_destroy(&attr);
	        for(int t = 0; t < NTHREADS; t++) 
	        {
    	        rc = pthread_join(threads[t], NULL);
    	        if (rc) 
		        {
        	        printf("ERROR; return code from pthread_join() is %d\n", rc);
        	        free(threads);
                    free(thread_args);
                    goto cleanup;
                }
            }

            free(thread_args);
            free(threads);
        }
        end = clock();
        //printf("Ellapsed time (Quick Select): %lf\n", ((double) (end - start)) / CLOCKS_PER_SEC);

        // now copy the first K elements of each row of matrices
        // Dall, IDXall to D and IDX respectivelly
        start = clock();
        for (int i = 0; i < MBLOCK_SIZE; i++)
        {
            for (int j = 0; j < K; j++)
            {
                D[D_OFFSET + i * K + j] = Dall[i * N + j];
                IDX[D_OFFSET + i * K + j] = IDXall[i * N + j] + 1; // 1-based indexing
            }
        }
        end = clock();
        //printf("Ellapsed time (final initialization): %lf\n", ((double) (end - start)) / CLOCKS_PER_SEC);
    }

    status = EXIT_SUCCESS;

cleanup:
    free(sqrmag_C);
    free(sqrmag_Q);
    free(Dall);
    free(IDXall);
    return status;
}
