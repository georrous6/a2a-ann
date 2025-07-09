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
    hid_t file_id, dataset_id, datatype_id, space_id;
    H5T_class_t type_class;
    size_t type_size;
    hsize_t dims[2];
    herr_t status;
    void *data = NULL;

    // Open file
    file_id = H5Fopen(file_name, H5F_ACC_RDONLY, H5P_DEFAULT);
    if (file_id < 0) {
        fprintf(stderr, "Error: Unable to open file %s\n", file_name);
        return NULL;
    }

    // Open dataset
    dataset_id = H5Dopen(file_id, dataset_name, H5P_DEFAULT);
    if (dataset_id < 0) {
        fprintf(stderr, "Error: Unable to open dataset %s\n", dataset_name);
        H5Fclose(file_id);
        return NULL;
    }

    // Get dataspace
    space_id = H5Dget_space(dataset_id);
    if (space_id < 0) {
        fprintf(stderr, "Error: Unable to get dataspace for dataset %s\n", dataset_name);
        H5Dclose(dataset_id);
        H5Fclose(file_id);
        return NULL;
    }

    // Ensure it's a 2D dataset
    int ndims = H5Sget_simple_extent_ndims(space_id);
    if (ndims != 2) {
        fprintf(stderr, "Error: Dataset %s is not 2D\n", dataset_name);
        H5Sclose(space_id);
        H5Dclose(dataset_id);
        H5Fclose(file_id);
        return NULL;
    }

    // Get dimensions
    H5Sget_simple_extent_dims(space_id, dims, NULL);
    printf("Dataset dimensions: %llu x %llu\n", dims[0], dims[1]);
    *rows = (int)dims[0];
    *cols = (int)dims[1];
    size_t total_elements = dims[0] * dims[1];

    // Get datatype
    datatype_id = H5Dget_type(dataset_id);
    if (datatype_id < 0) {
        fprintf(stderr, "Error: Unable to get datatype of dataset %s\n", dataset_name);
        H5Sclose(space_id);
        H5Dclose(dataset_id);
        H5Fclose(file_id);
        return NULL;
    }

    type_class = H5Tget_class(datatype_id);
    type_size = H5Tget_size(datatype_id);

    if (type_class == H5T_INTEGER && type_size == sizeof(int)) {
        printf("Detected data type: int\n");
        data = malloc(total_elements * sizeof(int));
        if (!data) {
            fprintf(stderr, "Error allocating memory for int matrix\n");
        } else {
            status = H5Dread(dataset_id, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, data);
            if (status < 0) {
                fprintf(stderr, "Error: Unable to read int dataset %s\n", dataset_name);
                free(data);
                data = NULL;
            }
        }
    } else if (type_class == H5T_FLOAT && type_size == sizeof(float)) {
        printf("Detected data type: float\n");
        data = malloc(total_elements * sizeof(float));
        if (!data) {
            fprintf(stderr, "Error allocating memory for float matrix\n");
        } else {
            status = H5Dread(dataset_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, data);
            if (status < 0) {
                fprintf(stderr, "Error: Unable to read float dataset %s\n", dataset_name);
                free(data);
                data = NULL;
            }
        }
    } else if (type_class == H5T_FLOAT && type_size == sizeof(double)) {
        printf("Detected data type: double\n");
        data = malloc(total_elements * sizeof(double));
        if (!data) {
            fprintf(stderr, "Error allocating memory for double matrix\n");
        } else {
            status = H5Dread(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, data);
            if (status < 0) {
                fprintf(stderr, "Error: Unable to read double dataset %s\n", dataset_name);
                free(data);
                data = NULL;
            }
        }
    } else {
        fprintf(stderr, "Error: Unsupported data type (class=%d, size=%zu).\n", type_class, type_size);
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

    // Check for 2D numeric matrix
    if (mxGetNumberOfDimensions(matvar) == 2)
    {
        *rows = (int)mxGetM(matvar);
        *cols = (int)mxGetN(matvar);
        size_t total_elements = (*rows) * (*cols);

        if (mxIsDouble(matvar))
        {
            printf("Detected data type: double\n");
            data = malloc(total_elements * sizeof(double));
            if (!data) 
            {
                fprintf(stderr, "Error allocating memory for double matrix data.\n");
                mxDestroyArray(matvar);
                matClose(matfp);
                return NULL;
            }

            double *matData = mxGetPr(matvar);
            for (int i = 0; i < *rows; i++) 
            {
                for (int j = 0; j < *cols; j++) 
                {
                    ((double *)data)[i * (*cols) + j] = matData[j * (*rows) + i];
                }
            }
        }
        else if (mxIsSingle(matvar))
        {
            printf("Detected data type: float (single)\n");
            data = malloc(total_elements * sizeof(float));
            if (!data) 
            {
                fprintf(stderr, "Error allocating memory for float matrix data.\n");
                mxDestroyArray(matvar);
                matClose(matfp);
                return NULL;
            }

            float *matData = (float *)mxGetData(matvar);
            for (int i = 0; i < *rows; i++) 
            {
                for (int j = 0; j < *cols; j++) 
                {
                    ((float *)data)[i * (*cols) + j] = matData[j * (*rows) + i];
                }
            }
        }
        else if (mxIsInt32(matvar))
        {
            printf("Detected data type: int32\n");
            data = malloc(total_elements * sizeof(int));
            if (!data) 
            {
                fprintf(stderr, "Error allocating memory for int32 matrix data.\n");
                mxDestroyArray(matvar);
                matClose(matfp);
                return NULL;
            }

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
            fprintf(stderr, "Unsupported data type for variable '%s'. Only double, float (single), or int32 supported.\n", matname);
            mxDestroyArray(matvar);
            matClose(matfp);
            return NULL;
        }
    }
    else
    {
        fprintf(stderr, "Variable '%s' is not a 2D matrix.\n", matname);
        mxDestroyArray(matvar);
        matClose(matfp);
        return NULL;
    }

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
        fprintf(stderr, "Error: Unsupported file mode. Use 'w' or 'a'.\n");
        return EXIT_FAILURE;
    }

    if (!matfp) 
    {
        fprintf(stderr, "Error opening MAT file '%s'.\n", filename);
        return EXIT_FAILURE;
    }

    mxArray *mx_matrix = NULL;

    if (type == DOUBLE_TYPE) 
    {
        mx_matrix = mxCreateDoubleMatrix(rows, cols, mxREAL);
        if (!mx_matrix) 
        {
            fprintf(stderr, "Error creating MATLAB double matrix '%s'.\n", matname);
            matClose(matfp);
            return EXIT_FAILURE;
        }

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
    else if (type == INT_TYPE) 
    {
        mx_matrix = mxCreateNumericMatrix(rows, cols, mxINT32_CLASS, mxREAL);
        if (!mx_matrix) 
        {
            fprintf(stderr, "Error creating MATLAB int32 matrix '%s'.\n", matname);
            matClose(matfp);
            return EXIT_FAILURE;
        }

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
    else if (type == FLOAT_TYPE) 
    {
        mx_matrix = mxCreateNumericMatrix(rows, cols, mxSINGLE_CLASS, mxREAL);
        if (!mx_matrix) 
        {
            fprintf(stderr, "Error creating MATLAB float (single) matrix '%s'.\n", matname);
            matClose(matfp);
            return EXIT_FAILURE;
        }

        float *mxData = (float *)mxGetData(mx_matrix);
        const float *inputData = (const float *)mat;
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

    // Write the matrix to the file
    if (matPutVariable(matfp, matname, mx_matrix) != 0) 
    {
        fprintf(stderr, "Error writing variable '%s' to '%s'.\n", matname, filename);
        mxDestroyArray(mx_matrix);
        matClose(matfp);
        return EXIT_FAILURE;
    }

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
                printf("%lf ", ((const double *)mat)[i * cols + j]);
            }
            else if (type == INT_TYPE)
            {
                printf("%d ", ((const int *)mat)[i * cols + j]);
            }
            else if (type == FLOAT_TYPE)
            {
                printf("%f ", ((const float *)mat)[i * cols + j]);
            }
        }
        printf("\n");
    }
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
