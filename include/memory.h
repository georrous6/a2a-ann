#ifndef MEMORY_H
#define MEMORY_H

#define MAX_MEMORY_USAGE_RATIO 0.8        // Use up to 80% of available memory
#define MIN_THREAD_CORPUS_SIZE 50         // Minimum number of corpus points per thread task
#define MIN_THREAD_QUERIES_SIZE 2         // Minimum number of queries per thread task

/**
 * Allocates the appropriate memory for the matrices of the exact solution according 
 * to the system's available resources. This function repeatedly splits the corpus data
 * into blocks until it fits in the memory without exausting the system's resources.
 * 
 * @param Dall the distances matrix between all the corpus and the queries points
 * @param IDXall the indexes matrix between all the corpus and the queries points
 * @param sqrmag_Q a vector with the square magnitudes of all the query points
 * @param sqrmag_C a vector with the square magnitudes of all the coprus points
 * @param M the number of queries
 * @param N the number of corpus data
 * @param MBLOCK_SIZE the maximum number of the queries that the program is able to handle
 * @return EXIT_SUCCESS if the allocation was successfull and EXIT_FAILURE if the maximum 
 * number of iterations is reached, defined by ALLOC_MAX_ITERS
 */
int alloc_memory(double **Dall, int **IDXall, double **sqrmag_Q, double **sqrmag_C, const int M, const int N, int *MBLOCK_SIZE);


/**
 * Returns the system's available memory in bytes.
 */
unsigned long get_available_memory_bytes();


/**
 * Computes the optimal number of threads according to the maximum
 * number of queries per block and the size of the corpus.
 * 
 * @param MBLOCK_MAX_SIZE the maximum number of queries per block
 * @param N the number of corpus points
 * @return the number of threads to use
 */
int get_num_threads(int MBLOCK_MAX_SIZE, int N);

#endif // MEMORY_H
