#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <string.h>
#include <cblas.h>
#include <matio.h>
#include <math.h>
#include <limits.h>
#include <stddef.h>


/**
 * Function to swap elements from two arrays.
 * 
 * @param arr data array
 * @param idx the index array
 * @param i the index of the first element to be swapped
 * @param j the index of the second element to be swapped
 */
void swap(double *arr, size_t *idx, size_t i, size_t j) 
{
    double dtemp = arr[i];
    arr[i] = arr[j];
    arr[j] = dtemp;
    size_t lutemp = idx[i];
    idx[i] = idx[j];
    idx[j] = lutemp;
}


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
size_t partition(double *arr, size_t *idx, size_t l, size_t r) 
{
    double pivot = arr[r];
    size_t i = l;
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

/**
 * Apply Quick Select algorithm to an array containing pairs of value-index,
 * where the index is the initial index of the element in the array.
 * 
 * @param arr an array containing pairs of value-index
 * @param l the leftmost index of the array
 * @param r the rightmost index of the array
 * @param k the index of the kth smallest element incremented by 1
 */
void qselect(double *arr, size_t *idx, size_t l, size_t r, size_t k) 
{
    // Partition the array around the last 
    // element and get the position of the pivot 
    // element in the sorted array
    size_t index = partition(arr, idx, l, r);

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


/**
 * Loads a matrix from a .mat file.
 * 
 * @param filename the name of the .mat file
 * @param matname the name of the matrix to be loaded
 * @param rows stores the number of rows of the matrix
 * @param cols stores the number of columns of the matrix
 * @return a pointer to the dynamically allocated array of data
 * or NULL if an error occured
 * @note memory deallocation should take place outside the function
 */
double* load_matrix(const char *filename, const char* matname, size_t* rows, size_t* cols)
{
    mat_t *matfp;
    matvar_t *matvar;
    double *data = NULL;

    // Open the .mat file for reading
    matfp = Mat_Open(filename, MAT_ACC_RDONLY);
    if (!matfp) 
    {
        fprintf(stderr, "Error opening MAT file: %s\n", strerror(errno));
        return NULL;
    }

    // Read the matrix from the .mat file
    matvar = Mat_VarRead(matfp, matname);
    if (!matvar) 
    {
        fprintf(stderr, "Error reading variable '%s' from MAT file.\n", matname);
        Mat_Close(matfp);
        return NULL;
    }

    // Check if the variable is a 2D double matrix
    if (matvar->rank == 2 && matvar->data_type == MAT_T_DOUBLE) 
    {
        // Store dimensions
        *rows = matvar->dims[0];
        *cols = matvar->dims[1];

        // Allocate memory to copy matrix data
        data = (double *)malloc((*rows) * (*cols) * sizeof(double));
        if (!data) 
        {
            fprintf(stderr, "Error allocating memory for matrix data.\n");
            Mat_VarFree(matvar);
            Mat_Close(matfp);
            return NULL;
        }

        // Copy matrix data with transposition
        for (size_t i = 0; i < *rows; i++) 
        {
            for (size_t j = 0; j < *cols; j++) 
            {
                data[i * (*cols) + j] = ((double*)matvar->data)[j * (*rows) + i];
            }
        }
    } 
    else 
    {
        fprintf(stderr, "The variable '%s' is not a 2D double matrix.\n", matname);
        Mat_VarFree(matvar);
        Mat_Close(matfp);
        return NULL;
    }

    // Clean up
    Mat_VarFree(matvar);
    Mat_Close(matfp);

    return data;
}


/**
 * Prints the matrix to standard output
 * 
 * @param mat the matrix
 * @param name the name of the matrix
 * @param rows the number of rows of the matrix
 * @param cols the number of columns of the matrix
 */
void print_matrix(const double* mat, const char* name, size_t rows, size_t cols)
{
    printf("\n%s:\n", name);
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++) 
        {
            printf("%lf ", mat[i * cols + j]);
        }
        printf("\n");
    }
}


/**
 * Computes the k-nearest neighbors for each row vector in Q with each
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
 * @return 0 on succesfull exit and 1 if an error occured
 * @note The function assumes that Q and C have the same number of columns.
 * @note The user is responsible to pass IDX and D matrices with the appropriate
 * dimensions
 */
int knnsearch(const double* Q, const double* C, size_t* IDX, double* D, size_t M, size_t N, size_t L, size_t K)
{
    // matrix to store the distances between all row vectors of matrix Q
    // and all row vectors of matrix C
    double *Dall = (double *)malloc(M * N * sizeof(double));
    if (!Dall)
    {
        fprintf(stderr, "Error allocating memory for distance matrix\n");
        return EXIT_FAILURE;
    }

    // input data indices of nearest neighbors
    size_t *IDXall = (size_t *)malloc(M * N * sizeof(size_t));
    if (!IDXall)
    {
        fprintf(stderr, "Error allocating memory for index matrix\n");
        free(Dall);
        return EXIT_FAILURE;
    }

    // initialize index matrix
    for (size_t i = 0; i < M; i++)
    {
        for (size_t j = 0; j < N; j++)
        {
            IDXall[i * N + j] = j;
        }
    }

    // compute D = -2*Q*C'
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, M, N, L, -2.0, Q, L, C, L, 0.0, Dall, N);

    // stores the squares of the magnitudes of the row vectors of matrix Q
    double *sqrmag_Q = (double *)malloc(M * sizeof(double));
    if (!sqrmag_Q)
    {
        fprintf(stderr, "Error allocating memory for square magnitudes vector\n");
        free(Dall);
        free(IDXall);
        return EXIT_FAILURE;
    }

    // stores the squares of the magnitudes of the row vectors of matrix C
    double *sqrmag_C = (double *)malloc(N * sizeof(double));
    if (!sqrmag_C)
    {
        fprintf(stderr, "Error allocating memory for square magnitudes vector\n");
        free(sqrmag_Q);
        free(Dall);
        free(IDXall);
        return EXIT_FAILURE;
    }

    double tmp;
    // compute the square of magnitudes of the row vectors of matrix Q
    for (size_t i = 0; i < M; i++)
    {
        tmp = 0.0;
        for (size_t j = 0; j < L; j++)
        {
            tmp += Q[i * L + j] * Q[i * L + j]; 
        }
        sqrmag_Q[i] = tmp;
    }

    // compute the square of magnitudes of the row vectors of matrix C
    for (size_t i = 0; i < N; i++)
    {
        tmp = 0.0;
        for (size_t j = 0; j < L ; j++)
        {
            tmp += C[i * L + j] * C[i * L + j]; 
        }
        sqrmag_C[i] = tmp;
    }

    // compute the distance matrix D by applying the formula D = sqrt(C.^2 -2*Q*C' + (Q.^2)')
    for (size_t i = 0; i < M; i++)
    {
        for (size_t j = 0; j < N; j++)
        {
            Dall[i * N + j] = sqrt(Dall[i * N + j] + sqrmag_Q[i] + sqrmag_C[j]);
        }
    }

    // apply Quick Select algorithm for each row of distance matrix
    for (size_t i = 0; i < M; i++)
    {
        qselect(Dall + i * N, IDXall + i * N, 0, N - 1, K);
    }

    // now copy the first K elements of each row of matrices
    // Dall, IDXall to D and IDX respectivelly
    for (int i = 0; i < M; i++)
    {
        for (int j = 0; j < K; j++)
        {
            D[i * K + j] = Dall[i * N + j];
            IDX[i * K + j] = IDXall[i * N + j];
        }
    }

    free(sqrmag_C);
    free(sqrmag_Q);
    free(Dall);
    free(IDXall);
    return EXIT_SUCCESS;
}

/**
 * Converts a string to unsigned long.
 * 
 * @param value the evaluation of the string
 * @param str the string to be evaluated
 * @return 0 if the evaluation was successfull and 1 otherwise
 */
int str2size_t(size_t* value, const char *str) 
{
    char *endptr;
    errno = 0;  // Clear errno before calling strtoul

    unsigned long ul = strtoul(str, &endptr, 10);

    // Error handling
    if (errno == ERANGE && ul == ULONG_MAX) 
    {
        printf("Overflow occurred, the value of K is too large.\n");
        return EXIT_FAILURE;  // Return max size_t value to indicate overflow
    }
    if (endptr == str || *endptr != '\0') 
    {
        printf("Invalid input for K parameter.\n");
        return EXIT_FAILURE;  // Indicate conversion failure
    }

    *value = (size_t)ul;
    return EXIT_SUCCESS;
}


int main(int argc, char *argv[])
{
    if (argc != 5)
    {
        fprintf(stderr, "Invalid number of arguments. \nExpected input format is <INPUT_DATA.mat> <CORPUS_MAT> <QUERIES_MAT> <K>\n");
        return EXIT_FAILURE;
    }

    // check the validity of value K
    size_t K;
    if (str2size_t(&K, argv[4]))
    {
        return EXIT_FAILURE;
    }
    
    size_t CM, CN, QM, QN;
    const char* filename = argv[1];
    const char* corpus = argv[2];
    const char* queries = argv[3];
    double* C = load_matrix(filename, corpus, &CM, &CN);
    if (!C)
        return EXIT_FAILURE;

    double* Q = load_matrix(filename, queries, &QM, &QN);
    if (!Q)
    {
        free(C);
        return EXIT_FAILURE;
    }

    if (CN != QN)
    {
        fprintf(stderr, "Invalid dimensions for corpus and queries data\n");
        free(C);
        free(Q);
        return EXIT_FAILURE;
    }

    if (K > CM)
    {
        fprintf(stderr, "Invalid K value; must be smaller or equal to the corpus size\n");
        free(C);
        free(Q);
        return EXIT_FAILURE;
    }

    double *D = (double *)malloc(QM * K * sizeof(double));
    if (!D)
    {
        fprintf(stderr, "Error allocating memory for matrix D\n");
        free(C);
        free(Q);
        return EXIT_FAILURE;
    }

    size_t *IDX = (size_t *)malloc(QM * K * sizeof(size_t));
    if (!IDX)
    {
        fprintf(stderr, "Error allocating memory for matrix IDX\n");
        free(C);
        free(Q);
        free(D);
        return EXIT_FAILURE;
    }

    if (knnsearch(Q, C, IDX, D, QM, CM, QN, K))
    {
        free(C);
        free(Q);
        free(D);
        free(IDX);
        return EXIT_FAILURE;
    }

    print_matrix(C, corpus, CM, CN);
    print_matrix(Q, queries, QM, QN);
    print_matrix(D, "D", QM, K);
    //print_matrix(IDX, "IDX", QM, K);

    free(C);
    free(Q);
    free(D);
    free(IDX);
    return EXIT_SUCCESS;

}