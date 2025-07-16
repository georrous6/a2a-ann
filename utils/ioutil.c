#include "ioutil.h"
#include "hdf5.h"
#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <limits.h>
#include <stdint.h>
#include <string.h>
#include <getopt.h> 
#include <dirent.h>
#include <sys/types.h>


void *load_hdf5(const char *file_name, const char *dataset_name, int *rows, int *cols) 
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


int store_hdf5(const void* mat, const char* matname, int rows, int cols, const char *filename, MATRIX_TYPE type, const char mode) {
    if (!mat) {
        fprintf(stderr, "Error: Matrix data is NULL.\n");
        return EXIT_FAILURE;
    }

    hid_t file_id;
    if (mode == 'w') {
        file_id = H5Fcreate(filename, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    } else if (mode == 'a') {
        file_id = H5Fopen(filename, H5F_ACC_RDWR, H5P_DEFAULT);
        if (file_id < 0) {
            fprintf(stderr, "File '%s' does not exist in append mode. Creating it...\n", filename);
            file_id = H5Fcreate(filename, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
        }
    } else {
        fprintf(stderr, "Error: Unsupported file mode. Use 'w' or 'a'.\n");
        return EXIT_FAILURE;
    }

    if (file_id < 0) {
        fprintf(stderr, "Error opening or creating HDF5 file '%s'.\n", filename);
        return EXIT_FAILURE;
    }

    hsize_t dims[2] = { rows, cols };
    hid_t dataspace_id = H5Screate_simple(2, dims, NULL);
    if (dataspace_id < 0) {
        fprintf(stderr, "Error creating dataspace.\n");
        H5Fclose(file_id);
        return EXIT_FAILURE;
    }

    hid_t dtype;
    size_t type_size;
    switch (type) {
        case DOUBLE_TYPE:
            dtype = H5T_NATIVE_DOUBLE;
            type_size = sizeof(double);
            break;
        case FLOAT_TYPE:
            dtype = H5T_NATIVE_FLOAT;
            type_size = sizeof(float);
            break;
        case INT_TYPE:
            dtype = H5T_NATIVE_INT;
            type_size = sizeof(int);
            break;
        default:
            fprintf(stderr, "Error: Unsupported matrix type.\n");
            H5Sclose(dataspace_id);
            H5Fclose(file_id);
            return EXIT_FAILURE;
    }

    if (H5Lexists(file_id, matname, H5P_DEFAULT) > 0) {
        fprintf(stderr, "Dataset '%s' already exists. Deleting it.\n", matname);
        H5Ldelete(file_id, matname, H5P_DEFAULT);
    }

    hid_t dset_id = H5Dcreate(file_id, matname, dtype, dataspace_id, 
                              H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    if (dset_id < 0) {
        fprintf(stderr, "Error creating dataset '%s'.\n", matname);
        H5Sclose(dataspace_id);
        H5Fclose(file_id);
        return EXIT_FAILURE;
    }

    herr_t status = H5Dwrite(dset_id, dtype, H5S_ALL, H5S_ALL, H5P_DEFAULT, mat);
    if (status < 0) {
        fprintf(stderr, "Error writing data to dataset '%s'.\n", matname);
        H5Dclose(dset_id);
        H5Sclose(dataspace_id);
        H5Fclose(file_id);
        return EXIT_FAILURE;
    }

    // Cleanup
    H5Dclose(dset_id);
    H5Sclose(dataspace_id);
    H5Fclose(file_id);

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
