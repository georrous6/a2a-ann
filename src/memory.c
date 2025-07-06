#include "memory.h"
#include <sys/sysinfo.h>
#include <unistd.h>
#include <stdlib.h>
#include <stdio.h>

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

        printf("Too large distance matrix. Max queries per block: %d\n", *MBLOCK_MAX_SIZE);
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