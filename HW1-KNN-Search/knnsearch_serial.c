#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <string.h>
#include <time.h>
#include <cblas.h>
#include <matio.h>
#include <math.h>

#define DATA_DIM 2
#define CORPUS_DATA_SIZE 20
#define QUERY_DATA_SIZE 1


int store_random_data_to_file(const char* filename, const char* matname, size_t rows, size_t cols)
{
    double *data;
    mat_t *matfp = NULL;
    matvar_t *matvar;

    data = (double *)malloc(rows * cols * sizeof(double));
    if (data == NULL) 
    {
        fprintf(stderr, "Error allocating memory\n");
        return EXIT_FAILURE;
    }

    // Generate random doubles and fill the matrix
    for (int i = 0; i < rows * cols; i++) 
    {
        data[i] = (double)rand() / RAND_MAX; // Random double between 0 and 1
    }

    // Try opening the MAT file in read/write mode (append mode)
    matfp = Mat_Open(filename, MAT_ACC_RDWR);
    if (!matfp)
    {
        // If file doesn't exist, create a new one
        matfp = Mat_CreateVer(filename, NULL, MAT_FT_MAT5); // or MAT_FT_MAT4 / MAT_FT_MAT73
        if (!matfp)
        {
            fprintf(stderr, "Error creating MAT file %s. %s\n", filename, strerror(errno));
            free(data);
            return EXIT_FAILURE;
        }
    }

    // Create a MAT variable for the matrix
    size_t dim2d[2] = { rows, cols };
    matvar = Mat_VarCreate(matname, MAT_C_DOUBLE, MAT_T_DOUBLE, 2, dim2d, data, 0);
    if (!matvar) 
    {
        fprintf(stderr, "Error creating MAT variable %s\n", matname);
        Mat_Close(matfp);
        free(data);
        return EXIT_FAILURE;
    }

    // Write the variable to the .mat file
    if (Mat_VarWrite(matfp, matvar, MAT_COMPRESSION_NONE)) 
    {
        fprintf(stderr, "Error writing MAT variable %s\n", matname);
        Mat_VarFree(matvar);
        Mat_Close(matfp);
        free(data);
        return EXIT_FAILURE;
    }

    // Free resources
    Mat_VarFree(matvar);
    Mat_Close(matfp);
    free(data);

    return EXIT_SUCCESS;
}


double* load_matrix_from_file(const char *filename, const char* matname, size_t* rows, size_t* cols)
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

        // Copy matrix data
        memcpy(data, matvar->data, (*rows) * (*cols) * sizeof(double));
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


void print_matrix(const double* mat, size_t rows, size_t cols)
{
    for (int i = 0; i < rows * cols; i++)
    {
        if (i % cols == 0)
            printf("\n");
        printf("%lf ", mat[i]);
    }
}


int knnsearch(const double* Q, const double* C, double* D, int M, int N, int K, int k)
{
    // compute D = -2*Q*C'
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, M, N, K, -2.0, Q, K, C, K, 0.0, D, N);

    // stores the squares of the magnitudes of the row vectors of matrix Q
    double *sqrmag_Q = (double *)malloc(M * sizeof(double));
    if (!sqrmag_Q)
    {
        fprintf(stderr, "Error allocating memory\n");
        return EXIT_FAILURE;
    }

    // stores the squares of the magnitudes of the row vectors of matrix C
    double *sqrmag_C = (double *)malloc(N * sizeof(double));
    if (!sqrmag_C)
    {
        fprintf(stderr, "Error allocating memory\n");
        return EXIT_FAILURE;
    }

    double tmp;
    for (int i = 0; i < M; i++)
    {
        tmp = 0.0;
        for (int j = 0; j < K ; j++)
        {
            tmp += Q[i * K + j]; 
        }
        sqrmag_Q[i] = tmp;
    }

    for (int i = 0; i < N; i++)
    {
        tmp = 0.0;
        for (int j = 0; j < K ; j++)
        {
            tmp += C[i * K + j]; 
        }
        sqrmag_C[i] = tmp;
    }

    for (int i = 0; i < M; i++)
    {
        for (int j = 0; j < N; j++)
        {
            D[i * N + j] = sqrt(D[i * N + j] + sqrmag_Q[i] + sqrmag_C[j]);
        }
    }

    return EXIT_SUCCESS;
}

int main(int argc, char *argv[])
{
    srand(time(NULL));

    // if (store_random_data_to_file("data.mat", "corpus", CORPUS_DATA_SIZE, DATA_DIM))
    //     return EXIT_FAILURE;
    // if (store_random_data_to_file("data.mat", "queries", QUERY_DATA_SIZE, DATA_DIM))
    //     return EXIT_FAILURE;

    size_t CM, CN, QM, QN;
    double* C = load_matrix_from_file("data.mat", "corpus", &CM, &CN);
    if (!C)
        return EXIT_FAILURE;

    double* Q = load_matrix_from_file("data.mat", "queries", &QM, &QN);
    if (!Q)
    {
        free(C);
        return EXIT_FAILURE;
    }

    if (CN != QN)
    {
        fprintf(stderr, "Invalid dimensions for corpus and query data\n");
        free(C);
        free(Q);
        return EXIT_FAILURE;
    }

    double *D = (double *)malloc(QM * CM * sizeof(double));
    if (!D)
    {
        fprintf(stderr, "Error allocating memory\n");
        free(C);
        free(Q);
        return EXIT_FAILURE;
    }

    knnsearch(Q, C, D, QM, CM, QN, 1);

    printf("\n\nCoprus:");
    print_matrix(C, CM, CN);
    printf("\n\nQueries:");
    print_matrix(Q, QM, QN);
    printf("\n\nDistances:");
    print_matrix(D, QM, CM);
    printf("\n");

    free(C);
    free(Q);
    free(D);
    return EXIT_SUCCESS;

}