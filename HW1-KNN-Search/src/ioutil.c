#include "ioutil.h"
#include <stdio.h>
#include <matio.h>
#include <errno.h>
#include <limits.h>
#include <stdint.h>
#include <string.h>
#include <getopt.h> 


void* load_matrix(const char *filename, const char* matname, size_t* rows, size_t* cols)
{
    mat_t *matfp;
    matvar_t *matvar;
    void *data = NULL;

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
                ((double *)data)[i * (*cols) + j] = ((double*)matvar->data)[j * (*rows) + i];
            }
        }
    }
    else if (matvar->rank == 2 && matvar->data_type == MAT_T_UINT64)
    {
        // Store dimensions
        *rows = matvar->dims[0];
        *cols = matvar->dims[1];

        // Allocate memory for a size_t (uint64_t) matrix
        data = (size_t *)malloc((*rows) * (*cols) * sizeof(size_t));
        if (!data) 
        {
            fprintf(stderr, "Error allocating memory for size_t matrix data.\n");
            Mat_VarFree(matvar);
            Mat_Close(matfp);
            return NULL;
        }

        // Copy matrix data with transposition
        for (size_t i = 0; i < *rows; i++) 
        {
            for (size_t j = 0; j < *cols; j++) 
            {
                ((size_t*)data)[i * (*cols) + j] = ((uint64_t*)matvar->data)[j * (*rows) + i];
            }
        }
    }
    else 
    {
        fprintf(stderr, "The variable '%s' is not a 2D size_t or double matrix.\n", matname);
        Mat_VarFree(matvar);
        Mat_Close(matfp);
        return NULL;
    }

    // Clean up
    Mat_VarFree(matvar);
    Mat_Close(matfp);

    return data;
}


int store_matrix(const void* mat, const char* matname, size_t rows, size_t cols, const char *filename, MATRIX_TYPE type)
{
    if (!mat) 
    {
        fprintf(stderr, "Error: Matrix data is NULL.\n");
        return EXIT_FAILURE;
    }

    mat_t* matfp = Mat_CreateVer(filename, NULL, MAT_FT_MAT5);
    if (!matfp) 
    {
        fprintf(stderr, "Error creating MAT file \'%s\'. %s\n", filename, strerror(errno));
        return EXIT_FAILURE;
    }

    size_t dims[2] = {rows, cols};
    matvar_t *matvar = NULL;
    
    if (type == DOUBLE_TYPE)
    {
        matvar = Mat_VarCreate(matname, MAT_C_DOUBLE, MAT_T_DOUBLE, 2, dims, mat, 0);
    }
    else if (type == SIZE_T_TYPE)
    {
        // Convert size_t array to uint64_t for MATLAB compatibility
        uint64_t *data_uint64 = (uint64_t *)malloc(rows * cols * sizeof(uint64_t));
        if (!data_uint64) 
        {
            fprintf(stderr, "Error allocating for uint64_t matrix.\n");
            Mat_Close(matfp);
            return EXIT_FAILURE;
        }

        for (size_t i = 0; i < rows * cols; i++) 
        {
            data_uint64[i] = (uint64_t)((size_t *)mat)[i];
        }
        matvar = Mat_VarCreate(matname, MAT_C_UINT64, MAT_T_UINT64, 2, dims, data_uint64, 0);
        free(data_uint64);  // Free temporary uint64 array
    }
    else
    {
        fprintf(stderr, "Error: Unsupported matrix type.\n");
        Mat_Close(matfp);
        return EXIT_FAILURE;
    }
    
    if (!matvar)
    {
        fprintf(stderr, "Error creating MAT variable %s\n", matname);
        Mat_Close(matfp);
        return EXIT_FAILURE;
    }

    if (Mat_VarWrite(matfp, matvar, MAT_COMPRESSION_NONE))
    {
        fprintf(stderr, "Error writing variable \'%s\' to \'%s\'\n", matname, filename);
        Mat_VarFree(matvar);
        Mat_Close(matfp);
        return EXIT_FAILURE;
    }

    Mat_VarFree(matvar);
    Mat_Close(matfp);
    return EXIT_SUCCESS;
}


void print_matrix(const void* mat, const char* name, size_t rows, size_t cols, MATRIX_TYPE type)
{
    printf("\n%s:\n", name);
    for (size_t i = 0; i < rows; i++)
    {
        for (size_t j = 0; j < cols; j++) 
        {
            if (type == DOUBLE_TYPE)
            {
                printf("%lf ", ((double *)mat)[i * cols + j]);
            }
            else if (type == SIZE_T_TYPE)
            {
                printf("%zu ", ((size_t *)mat)[i * cols + j]); // %zu is the format specifier for size_t
            }
        }
        printf("\n");
    }
}


/**
 * Converts a string to size_t.
 * 
 * @param value the evaluation of the string
 * @param str the string to be evaluated
 * @return EXIT_SUCCESS if the evaluation was successfull and EXIT_FAILURE otherwise
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


void print_usage(const char *program_name) 
{
    fprintf(stderr, "Usage: %s <filename> <CNAME> <QNAME> <K> [-s] [-o output_file] [-jN]\n", program_name);
    fprintf(stderr, "Options:\n");
    fprintf(stderr, "  -s, --sorted          Use sorted data\n");
    fprintf(stderr, "  -o, --output <file>   Specify output file (default is stdout)\n");
    fprintf(stderr, "  -jN                   Number of threads (N should be an integer)\n");
}

// Function to parse command-line arguments
int parse_arguments(int argc, char *argv[], Options *opts, const char **filename, const char **CNAME, const char **QNAME, size_t *K) 
{
    int opt;
    
    // Initialize options
    opts->sorted = 0;  // Default: not sorted
    opts->output_filename = NULL; // Default: no output filename
    opts->num_threads = 1; // Default: 1 thread

    // Parse optional arguments
    const char *optstring = "so:j:";
    const struct option long_options[] = {
        {"sorted", no_argument, NULL, 's'},
        {"output", required_argument, NULL, 'o'},
        {"threads", required_argument, NULL, 'j'},
        {NULL, 0, NULL, 0}
    };

    while ((opt = getopt_long(argc, argv, optstring, long_options, NULL)) != -1) 
    {
        switch (opt) 
        {
            case 's':
                opts->sorted = 1; // Set the sorted flag
                break;
            case 'o':
                opts->output_filename = strdup(optarg); // Allocate memory for output filename
                if (!opts->output_filename)
                {
                    fprintf(stderr, "Error allocating memory for output filename\n");
                    return EXIT_FAILURE;
                }
                break;
            case 'j':
                opts->num_threads = atoi(optarg); // Convert N from string to int
                if (opts->num_threads < 1) 
                {
                    fprintf(stderr, "Invalid number of threads: %d\n", opts->num_threads);
                    return EXIT_FAILURE; // Return error
                }
                break;
            case '?':
                print_usage(argv[0]);
                return EXIT_FAILURE; // Return error
        }
    }

    // Check for the correct number of positional arguments
    if (argc - optind != 4) {
        fprintf(stderr, "Expected 4 positional arguments: <filename> <CNAME> <QNAME> <K>\n");
        print_usage(argv[0]);
        return EXIT_FAILURE; // Return error
    }

    // Positional arguments
    *filename = argv[optind];
    *CNAME = argv[optind + 1];
    *QNAME = argv[optind + 2];
    if (str2size_t(K, argv[optind + 3]) || *K == 0) // Convert K to size_t
    {
        fprintf(stderr, "The value for K must be positive\n");
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS; // Successful parsing
}
