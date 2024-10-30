#ifndef KNNSEARCH_H
#define KNNSEARCH_H

#include <stddef.h>

#define ALLOC_MAX_ITERS 10
#define MAX_MEMORY_USAGE_RATIO 0.8 // Use up to 80% of available memory
#define MIN_THREAD_CORPUS_SIZE 50


typedef struct THREAD_ARGS {
    double *Dall;
    int *IDXall;
    int K;
    int N;
    int MBLOCK_THREAD_SIZE;
    int sorted;
} THREAD_ARGS;


/**
 * Function to swap elements from two arrays.
 * 
 * @param arr data array
 * @param idx the index array
 * @param i the index of the first element to be swapped
 * @param j the index of the second element to be swapped
 */
void swap(double *arr, int *idx, int i, int j);


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
int partition(double *arr, int *idx, int l, int r);

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
void qselect(double *arr, int *idx, int l, int r, int k);


/**
 * Apply Quick Sort algorithm to an array while maintaining
 * the original index of the data
 * 
 * @param arr the data array
 * @param idx the indices array
 * @param l the leftmost index of the array
 * @param r the rightmost index of the array
 */
void qsort_(double *arr, int *idx, int l, int r);


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
 * @param sorted if set to a non-negative value outputs 
 * the distances in ascending order
 * @return 0 on succesfull exit and 1 if an error occured
 * @note The function assumes that Q and C have the same number of columns.
 * @note The user is responsible to pass IDX and D matrices with the appropriate
 * dimensions
 */
int knnsearch_exact(const double* Q, const double* C, int* IDX, double* D, const int M, const int N, const int L, const int K, const int sorted);


int alloc_memory(double **Dall, int **IDXall, double **sqrmag_Q, double **sqrmag_C, const int M, const int N, int *MBLOCK_SIZE);


unsigned long get_available_memory_bytes();


int get_thread_count(int MBLOCK_SIZE, int N);

#endif