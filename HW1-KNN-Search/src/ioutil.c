#include "ioutil.h"
#include "mat.h"
#include "hdf5.h"
#include <stdio.h>
#include <matio.h>
#include <errno.h>
#include <limits.h>
#include <stdint.h>
#include <string.h>
#include <getopt.h> 
#include <dirent.h>
#include <sys/types.h>


void* load_matrix(const char *filename, const char* matname, int* rows, int* cols)
{
    if (has_extension(filename, ".mat"))
    {
        return readMATFile(filename, matname, rows, cols);
    }
    else if (has_extension(filename, ".hdf5"))
    {
        return readHDF5File(filename, matname, rows, cols);
    }

    return NULL;
}


void *readHDF5File(const char *file_name, const char *dataset_name, int *rows, int *cols) 
{
    hid_t file_id, dataset_id, datatype_id, space_id; // IDs for file, dataset, datatype, and dataspace
    H5T_class_t type_class; // To store the type class (integer, float, etc.)
    hsize_t dims[2]; // For 2D datasets
    herr_t status;
    void *data = NULL;

    // Open the HDF5 file in read-only mode
    file_id = H5Fopen(file_name, H5F_ACC_RDONLY, H5P_DEFAULT);
    if (file_id < 0) 
    {
        fprintf(stderr, "Error: Unable to open file %s\n", file_name);
        return NULL;
    }

    // Open the dataset
    dataset_id = H5Dopen(file_id, dataset_name, H5P_DEFAULT);
    if (dataset_id < 0) 
    {
        fprintf(stderr, "Error: Unable to open dataset %s\n", dataset_name);
        H5Fclose(file_id);
        return NULL;
    }

    // Get the dataspace of the dataset
    space_id = H5Dget_space(dataset_id);
    if (space_id < 0) 
    {
        fprintf(stderr, "Error: Unable to get dataspace for dataset %s\n", dataset_name);
        H5Dclose(dataset_id);
        H5Fclose(file_id);
        return NULL;
    }

    // Verify that the dataset is 2D
    int ndims = H5Sget_simple_extent_ndims(space_id);
    if (ndims != 2) 
    {
        fprintf(stderr, "Error: Dataset %s is not 2D\n", dataset_name);
        H5Sclose(space_id);
        H5Dclose(dataset_id);
        H5Fclose(file_id);
        return NULL;
    }

    // Get the dimensions of the dataset
    H5Sget_simple_extent_dims(space_id, dims, NULL);
    printf("Dataset dimensions: %llu x %llu\n", dims[0], dims[1]);
    *rows = dims[0];
    *cols = dims[1];

    // Get the datatype of the dataset
    datatype_id = H5Dget_type(dataset_id);
    if (datatype_id < 0) 
    {
        fprintf(stderr, "Error: Unable to get datatype of dataset %s\n", dataset_name);
        H5Sclose(space_id);
        H5Dclose(dataset_id);
        H5Fclose(file_id);
        return NULL;
    }

    // Get the class of the datatype (integer, float, etc.)
    type_class = H5Tget_class(datatype_id);

    // Check if the type is supported (either integer or float)
    if (type_class == H5T_INTEGER) 
    {
        printf("Detected data type: Integer\n");
        // Allocate memory for the data
        size_t total_elements = dims[0] * dims[1];
        data = (int *)malloc(total_elements * sizeof(int));  // Allocate memory for integers
        if (!data)
        {
            fprintf(stderr, "Error allocating memory for int matrix\n");
            H5Tclose(datatype_id);
            H5Sclose(space_id);
            H5Dclose(dataset_id);
            H5Fclose(file_id);
            return NULL;
        }

        // Read the data into the array
        status = H5Dread(dataset_id, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, data);
        if (status < 0) 
        {
            fprintf(stderr, "Error: Unable to read integer dataset %s\n", dataset_name);
            free(data);
            H5Tclose(datatype_id);
            H5Sclose(space_id);
            H5Dclose(dataset_id);
            H5Fclose(file_id);
            return NULL;
        }
    } 
    else if (type_class == H5T_FLOAT) 
    {
        printf("Detected data type: Float\n");
        // Allocate memory for the data
        size_t total_elements = dims[0] * dims[1];
        float *temp = (float *)malloc(total_elements * sizeof(float));
        data = (double *)malloc(total_elements * sizeof(double));
        if (!temp || !data)
        {
            fprintf(stderr, "Error allocating memory for float/double matrix\n");
            H5Tclose(datatype_id);
            H5Sclose(space_id);
            H5Dclose(dataset_id);
            H5Fclose(file_id);
            if (temp) free(temp);
            if (data) free(data);
            return NULL;
        }

        // Read the data into the array
        status = H5Dread(dataset_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, temp);
        if (status < 0) 
        {
            fprintf(stderr, "Error: Unable to read float dataset %s\n", dataset_name);
            free(temp);
            free(data);
            H5Tclose(datatype_id);
            H5Sclose(space_id);
            H5Dclose(dataset_id);
            H5Fclose(file_id);
            return NULL;
        }

        // Convert from float to double
        for (size_t i = 0; i < total_elements; i++) 
        {
            ((double *)data)[i] = (double)temp[i];
        }

        free(temp);
    } 
    else 
    {
        fprintf(stderr, "Error: Unsupported data type. Only integers and floats are supported.\n");
        H5Tclose(datatype_id);
        H5Sclose(space_id);
        H5Dclose(dataset_id);
        H5Fclose(file_id);
        return NULL;
    }

    // Clean up
    H5Tclose(datatype_id);
    H5Sclose(space_id);
    H5Dclose(dataset_id);
    H5Fclose(file_id);
    return data;
}


void *readMATFile(const char *filename, const char* matname, int* rows, int* cols)
{
    MATFile *matfp;
    mxArray *matvar;
    void *data = NULL;

    // Open the MAT file
    matfp = matOpen(filename, "r");
    if (!matfp) 
    {
        fprintf(stderr, "Error opening MAT file '%s'.\n", filename);
        return NULL;
    }

    // Read the variable from the MAT file
    matvar = matGetVariable(matfp, matname);
    if (!matvar) 
    {
        fprintf(stderr, "Error reading variable '%s' from MAT file.\n", matname);
        matClose(matfp);
        return NULL;
    }

    // Check if the variable is a 2D numeric matrix
    if (mxGetNumberOfDimensions(matvar) == 2)
    {
        *rows = mxGetM(matvar);
        *cols = mxGetN(matvar);

        if (mxIsDouble(matvar))
        {
            // Allocate memory for the double matrix
            data = (double *)malloc((*rows) * (*cols) * sizeof(double));
            if (!data) 
            {
                fprintf(stderr, "Error allocating memory for matrix data.\n");
                mxDestroyArray(matvar);
                matClose(matfp);
                return NULL;
            }

            // Copy and transpose data from column-major (MATLAB) to row-major (C)
            double *matData = mxGetPr(matvar);
            for (int i = 0; i < *rows; i++) 
            {
                for (int j = 0; j < *cols; j++) 
                {
                    ((double *)data)[i * (*cols) + j] = matData[j * (*rows) + i];
                }
            }
        }
        else if (mxIsInt32(matvar))
        {
            // Allocate memory for the int matrix
            data = (int *)malloc((*rows) * (*cols) * sizeof(int));
            if (!data) 
            {
                fprintf(stderr, "Error allocating memory for int matrix data.\n");
                mxDestroyArray(matvar);
                matClose(matfp);
                return NULL;
            }

            // Copy and transpose data from column-major (MATLAB) to row-major (C)
            int *matData = (int *)mxGetData(matvar);
            for (int i = 0; i < *rows; i++) 
            {
                for (int j = 0; j < *cols; j++) 
                {
                    ((int *)data)[i * (*cols) + j] = matData[j * (*rows) + i];
                }
            }
        }
        else
        {
            fprintf(stderr, "The variable '%s' is not a supported numeric type (double or int32).\n", matname);
            mxDestroyArray(matvar);
            matClose(matfp);
            return NULL;
        }
    }
    else
    {
        fprintf(stderr, "The variable '%s' is not a 2D matrix.\n", matname);
        mxDestroyArray(matvar);
        matClose(matfp);
        return NULL;
    }

    // Clean up
    mxDestroyArray(matvar);
    matClose(matfp);

    return data;
}


int store_matrix(const void* mat, const char* matname, int rows, int cols, const char *filename, MATRIX_TYPE type, const char mode)
{
    if (!mat) 
    {
        fprintf(stderr, "Error: Matrix data is NULL.\n");
        return EXIT_FAILURE;
    }

    // Open or create the MAT file
    MATFile *matfp = NULL;
    if (mode == 'w') 
    {
        matfp = matOpen(filename, "w");
    }
    else if (mode == 'a') 
    {
        matfp = matOpen(filename, "u");
        if (!matfp) 
        {
            // File does not exist, create a new one
            matfp = matOpen(filename, "w");
            if (!matfp) 
            {
                fprintf(stderr, "Error: Unable to create MAT file '%s'.\n", filename);
                return EXIT_FAILURE;
            }
        }
    }
    else 
    {
        fprintf(stderr, "Error: Unsupported file mode. Use 'w' for write or 'a' for append.\n");
        return EXIT_FAILURE;
    }

    if (!matfp) 
    {
        fprintf(stderr, "Error opening MAT file '%s'.\n", filename);
        return EXIT_FAILURE;
    }

    // Create a MATLAB array for the matrix
    mxArray *mx_matrix = NULL;
    if (type == DOUBLE_TYPE) // Assume 0 corresponds to double
    {
        mx_matrix = mxCreateDoubleMatrix(rows, cols, mxREAL);
        if (!mx_matrix) 
        {
            fprintf(stderr, "Error creating MATLAB matrix '%s'.\n", matname);
            matClose(matfp);
            return EXIT_FAILURE;
        }

            // Transpose and copy data into the mxArray (convert row-major to column-major)
            double *mxData = mxGetPr(mx_matrix);
            const double *inputData = (const double *)mat;
            for (int i = 0; i < rows; i++) 
            {
                for (int j = 0; j < cols; j++) 
                {
                    mxData[j * rows + i] = inputData[i * cols + j];
                }
            }
    }
    else if (type == INT_TYPE) // Assume 1 corresponds to int32
    {
        mx_matrix = mxCreateNumericMatrix(rows, cols, mxINT32_CLASS, mxREAL);
        if (!mx_matrix) 
        {
            fprintf(stderr, "Error creating MATLAB matrix '%s'.\n", matname);
            matClose(matfp);
            return EXIT_FAILURE;
        }

        // Transpose and copy data into the mxArray (convert row-major to column-major)
        int *mxData = (int *)mxGetData(mx_matrix);
        const int *inputData = (const int *)mat;
        for (int i = 0; i < rows; i++) 
        {
            for (int j = 0; j < cols; j++) 
            {
                mxData[j * rows + i] = inputData[i * cols + j];
            }
        }
    }
    else 
    {
        fprintf(stderr, "Error: Unsupported matrix type.\n");
        matClose(matfp);
        return EXIT_FAILURE;
    }

    // Write the matrix to the MAT file
    if (matPutVariable(matfp, matname, mx_matrix) != 0) 
    {
        fprintf(stderr, "Error writing variable '%s' to '%s'.\n", matname, filename);
        mxDestroyArray(mx_matrix);
        matClose(matfp);
        return EXIT_FAILURE;
    }

    // Clean up
    mxDestroyArray(mx_matrix);
    matClose(matfp);

    return EXIT_SUCCESS;
}


void print_matrix(const void* mat, const char* name, int rows, int cols, MATRIX_TYPE type)
{
    printf("\n%s:\n", name);
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++) 
        {
            if (type == DOUBLE_TYPE)
            {
                printf("%lf ", ((double *)mat)[i * cols + j]);
            }
            else if (type == INT_TYPE)
            {
                printf("%d ", ((int *)mat)[i * cols + j]); // %zu is the format specifier for int
            }
        }
        printf("\n");
    }
}


void print_usage(const char *program_name) 
{
    fprintf(stderr, "Usage: %s <filename>.mat <C> <Q> <K> <IDX> <D> [-s] [-a] [-v] [-o <output_file>.mat] [-jN]\n", program_name);
    fprintf(stderr, "C        The coprus matrix\n");
    fprintf(stderr, "Q        The queries matrix\n");
    fprintf(stderr, "K        The matrix specifying the number of neighbors to search for\n");
    fprintf(stderr, "IDX      The indexes matrix\n");
    fprintf(stderr, "D        The distances matrix\n");
    fprintf(stderr, "Options:\n");
    fprintf(stderr, "  -s, --sorted          Use sorted data\n");
    fprintf(stderr, "  -a, --approx          For approximate solution\n");
    fprintf(stderr, "  -v, --verbose         Display information about the proccess\n");
    fprintf(stderr, "  -o, --output <file>   Specify output file (default is stdout)\n");
    fprintf(stderr, "  -jN                   Number of threads (N should be an integer)\n");
}


int parse_arguments(int argc, char *argv[], Options *opts, const char **filename, const char **C_NAME, const char **Q_NAME, const char **K_NAME, const char** IDX_NAME, const char **D_NAME) 
{
    int opt;
    
    // Initialize options
    opts->sorted = 0;              // Default: not sorted
    opts->approx = 0;              // Default: find the exact solution
    opts->verbose = 0;             // Default: do not diplay information
    opts->output_filename = NULL;  // Default: no output filename
    opts->num_threads = -1;        // Default: automatically determine the number of threads

    // Parse optional arguments
    const char *optstring = "savo:j:";
    const struct option long_options[] = {
        {"sorted", no_argument, NULL, 's'},
        {"approx", no_argument, NULL, 'a'},
        {"verbose", no_argument, NULL, 'v'},
        {"output", required_argument, NULL, 'o'},
        {"threads", required_argument, NULL, 'j'},
        {NULL, 0, NULL, 0}
    };

    while ((opt = getopt_long(argc, argv, optstring, long_options, NULL)) != -1) 
    {
        switch (opt) 
        {
            case 's':
                opts->sorted = 1;  // Set the sorted flag
                break;
            case 'a':
                opts->approx = 1;  // Set the approximate solution flag
                break;
            case 'v':
                opts->verbose = 1;  // Set the verbose flag
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
    if (argc - optind != 6) 
    {
        fprintf(stderr, "Expected 6 positional arguments, provided %d\n", argc - optind);
        print_usage(argv[0]);
        return EXIT_FAILURE; // Return error
    }

    // Positional arguments
    *filename = argv[optind];
    *C_NAME = argv[optind + 1];
    *Q_NAME = argv[optind + 2];
    *K_NAME = argv[optind + 3];
    *IDX_NAME = argv[optind + 4];
    *D_NAME = argv[optind + 5];


    return EXIT_SUCCESS; // Successful parsing
}


int has_extension(const char *filename, const char *extension) 
{
    size_t len_filename = strlen(filename);
    size_t len_extension = strlen(extension);

    // Check if the file length is at least as long as the extension
    if (len_filename < len_extension) 
    {
        return 0;
    }

    // Compare the end of filename with the extension
    return strcmp(filename + len_filename - len_extension, extension) == 0;
}


int compare_file_paths(const void *a, const void *b) 
{
    return strcmp(*(const char **)a, *(const char **)b);
}


char **get_file_paths(const char *directory_path, const char *extension, size_t *file_count, int sorted)
{
    char **file_paths = NULL;
    *file_count = 0;

    DIR *dir = opendir(directory_path);
    if (dir == NULL) 
    {
        fprintf(stderr, "Could not open directory \'%s\'\n", directory_path);
        return NULL;
    }


    struct dirent *entry;
    // first pass, count the number of files with the appropriate extension
    int cnt = 0;
    while ((entry = readdir(dir)) != NULL) 
    {
        // Skip the current and parent directory entries
        if (entry->d_name[0] == '.' && (entry->d_name[1] == '\0' || (entry->d_name[1] == '.' && entry->d_name[2] == '\0'))) 
        {
            continue;
        }

        // file is not for testing
        if (!has_extension(entry->d_name, extension))
        {
            continue;
        }

        cnt++;
    }

    // allocate memory for for the array of file paths
    file_paths = (char **)malloc(sizeof(char *) * cnt);
    if (!file_paths)
    {
        closedir(dir);
        return NULL;
    }


    // second pass, store the file paths
    cnt = 0;
    rewinddir(dir);
    const size_t MAX_PATH = 1024;
    char file_path[MAX_PATH];
    while ((entry = readdir(dir)) != NULL) 
    {
        // Skip the current and parent directory entries
        if (entry->d_name[0] == '.' && (entry->d_name[1] == '\0' || (entry->d_name[1] == '.' && entry->d_name[2] == '\0'))) 
        {
            continue;
        }

        // file is not for testing
        if (!has_extension(entry->d_name, extension))
        {
            continue;
        }

        snprintf(file_path, sizeof(file_path), "%s/%s", directory_path, entry->d_name);
        file_paths[cnt] = strdup(file_path);
        if (!file_paths[cnt])
        {
            fprintf(stderr, "Error allocating memory\n");
            for (int i = 0; i < cnt; i++)
            {
                free(file_paths[i]);
            }

            free(file_paths);
            closedir(dir);
            return NULL;
        }

        cnt++;
    }

    closedir(dir);

    // Sort file paths
    if (sorted)
        qsort(file_paths, cnt, sizeof(char *), compare_file_paths);

    *file_count = cnt;
    return file_paths;
}
