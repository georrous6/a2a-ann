#ifndef KNNSEARCH_H
#define KNNSEARCH_H

#include "Queue.h"
#include "template_definitions.h"
#include <pthread.h>

extern pthread_mutex_t mutexQueue;        // Mutex for the tasks Queue
extern pthread_cond_t condQueue;          // Condition variable for Queue
extern pthread_cond_t condTasksComplete;  // Condition variable to signal task completion for a block
extern int isActive;                      // Flag for threads to exit
extern int runningTasks;                  // Holds the number of running tasks


/**
 * Thread Task for the exact K-Nearest Neighbors problem
 */
typedef struct knnTask {
    const DTYPE *C;
    const DTYPE *Q;
    DTYPE *Dall;
    int *IDXall;
    const DTYPE *sqrmag_C;
    const DTYPE *sqrmag_Q;
    const int K;
    const int N;
    const int L;
    const int QUERIES_NUM;     // Number of queries for the task to proccess
    const int q_index;         // Index of the first query to be proccessed
    const int q_index_thread;  // Index of the query to be proccesed inside a thread
} knnTask;


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
int alloc_memory(DTYPE **Dall, int **IDXall, DTYPE **sqrmag_Q, DTYPE **sqrmag_C, const int M, const int N, int *MBLOCK_SIZE);


/**
 * Function to swap elements from two arrays.
 * 
 * @param arr data array
 * @param idx the index array
 * @param i the index of the first element to be swapped
 * @param j the index of the second element to be swapped
 */
void swap(DTYPE *arr, int *idx, int i, int j);


/**
 * Standard partition process of Quick Select.
 * It considers the last element as pivot and moves
 * all smaller elements to the left of it and greater
 * elements to the right.
 * 
 * @param arr the data array. The partitioning will be made with respect to its elements.
 * @param idx the index array
 * @param l the leftmost index of the array
 * @param r the rightmost index of the array
 * @return the index of the pivot element
 */
int partition(DTYPE *arr, int *idx, int l, int r);


/**
 * Apply Quick Select algorithm to an array while maintaining
 * the original index of the data
 * 
 * @param arr the data array
 * @param idx the indices array
 * @param l the leftmost index of the array
 * @param r the rightmost index of the array
 * @param k the index of the kth smallest element. Indexing starts from 1
 */
void qselect(DTYPE *arr, int *idx, int l, int r, int k);


/**
 * Apply Quick Sort algorithm to an array while maintaining
 * the original index of the data
 * 
 * @param arr the data array
 * @param idx the indices array
 * @param l the leftmost index of the array
 * @param r the rightmost index of the array
 */
void qsort_(DTYPE *arr, int *idx, int l, int r);


/**
 * Function for executing the task for the exact solution of KNN.
 * 
 * @param task the task to execute
 */
void knnTaskExec(const knnTask *task);

/**
 * The function running on the threads for the exact solution.
 * 
 * @param pool the thread pool for the exact solution.
 */
void *knnThreadStart(void *pool);


/**
 * Computes the exact k-nearest neighbors for each row vector in Q with each
 * row vector in C.
 * 
 * @param Q the query points (M x L)
 * @param C the corpus points (N x L)
 * @param IDX the matrix of indices (M x K)
 * @param D the matrix of distances (M x K)
 * @param M the number of rows of Q
 * @param N the number of rows of C
 * @param L the number of columns of Q and C
 * @param K the number of nearest neighbors
 * @param sorted if set to a non-negative value outputs the distances in ascending order
 * @return 0 on succesfull exit and 1 if an error occured
 * @note The function assumes that Q and C have the same number of columns.
 * @note The user is responsible to pass IDX and D matrices with the appropriate
 * dimensions
 */
int knnsearch(const DTYPE* Q, const DTYPE* C, int* IDX, DTYPE* D, const int M, const int N, const int L, const int K, const int sorted);

#endif // KNNSEARCH_H
