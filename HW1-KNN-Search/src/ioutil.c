#include "ioutil.h"
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
    mat_t *matfp;
    matvar_t *matvar;
    void *data = NULL;

    // Open the .mat file for reading
    matfp = Mat_Open(filename, MAT_ACC_RDONLY);
    if (!matfp) 
    {
        fprintf(stderr, "Error opening MAT file \'%s\': %s\n", filename, strerror(errno));
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
        for (int i = 0; i < *rows; i++) 
        {
            for (int j = 0; j < *cols; j++) 
            {
                ((double *)data)[i * (*cols) + j] = ((double *)matvar->data)[j * (*rows) + i];
            }
        }
    }
    else if (matvar->rank == 2 && matvar->data_type == MAT_T_INT32)
    {
        // Store dimensions
        *rows = matvar->dims[0];
        *cols = matvar->dims[1];

        // Allocate memory for a int matrix
        data = (int *)malloc((*rows) * (*cols) * sizeof(int));
        if (!data) 
        {
            fprintf(stderr, "Error allocating memory for int matrix data.\n");
            Mat_VarFree(matvar);
            Mat_Close(matfp);
            return NULL;
        }

        // Copy matrix data with transposition
        for (int i = 0; i < *rows; i++) 
        {
            for (int j = 0; j < *cols; j++) 
            {
                ((int*)data)[i * (*cols) + j] = ((int *)matvar->data)[j * (*rows) + i];
            }
        }
    }
    else 
    {
        fprintf(stderr, "The variable '%s' is not a 2D int or double matrix.\n", matname);
        Mat_VarFree(matvar);
        Mat_Close(matfp);
        return NULL;
    }

    // Clean up
    Mat_VarFree(matvar);
    Mat_Close(matfp);

    return data;
}


int store_matrix(const void* mat, const char* matname, int rows, int cols, const char *filename, MATRIX_TYPE type, const char mode)
{
    if (!mat) 
    {
        fprintf(stderr, "Error: Matrix data is NULL.\n");
        return EXIT_FAILURE;
    }

    mat_t* matfp = NULL;

    // Check mode and open the MAT file accordingly
    if (mode == 'w')
    {
        // Overwrite mode: create a new file
        errno = 0;
        matfp = Mat_CreateVer(filename, NULL, MAT_FT_MAT5);
        if (!matfp) 
        {
            fprintf(stderr, "Error creating MAT file '%s'. %s\n", filename, strerror(errno));
            return EXIT_FAILURE;
        }
    }
    else if (mode == 'a')
    {
        // Append mode: open an existing file or create it if it doesn't exist
        errno = 0;  // clear errors
        matfp = Mat_Open(filename, MAT_ACC_RDWR);
        if (!matfp && errno == ENOENT)  // If file does not exist, create it
        {
            matfp = Mat_CreateVer(filename, NULL, MAT_FT_MAT5);
            if (!matfp) 
            {
                fprintf(stderr, "Error creating MAT file '%s'. %s\n", filename, strerror(errno));
                return EXIT_FAILURE;
            }
        }
        else if(!matfp) // file already exists but cannot open it
        {
            fprintf(stderr, "Error opening MAT file '%s'. %s\n", filename, strerror(errno));
            return EXIT_FAILURE;
        }
    }
    else
    {
        fprintf(stderr, "Error: Unsupported file mode. Use 'w' for write or 'a' for append.\n");
        return EXIT_FAILURE;
    }


    size_t dims[2] = {rows, cols};
    matvar_t *matvar = NULL;
    
    if (type == DOUBLE_TYPE)
    {
        double *tmp = (double *)malloc(sizeof(double) * rows * cols);
        if (!tmp)
        {
            fprintf(stderr, "Error allocating memory for temporary matrix\n");
            Mat_Close(matfp);
            return EXIT_FAILURE;
        }

        // Copy the contents of mat in column major order
        int k = 0;
        for (int j = 0; j < cols; j++) 
        {
            for (int i = 0; i < rows; i++) 
            {
                tmp[k++] = ((double *)mat)[i * cols + j];
            }
        }

        matvar = Mat_VarCreate(matname, MAT_C_DOUBLE, MAT_T_DOUBLE, 2, dims, (void *)tmp, 0);
        free(tmp);
    }
    else if (type == INT_TYPE)
    {
        int *tmp = (int *)malloc(sizeof(int) * rows * cols);
        if (!tmp)
        {
            fprintf(stderr, "Error allocating memory for temporary matrix\n");
            Mat_Close(matfp);
            return EXIT_FAILURE;
        }

        // Copy the contents of mat in column major order
        int k = 0;
        for (int j = 0; j < cols; j++) 
        {
            for (int i = 0; i < rows; i++) 
            {
                tmp[k++] = ((int *)mat)[i * cols + j];
            }
        }
        matvar = Mat_VarCreate(matname, MAT_C_INT32, MAT_T_INT32, 2, dims, (void *)tmp, 0);
        free(tmp);
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
    fprintf(stderr, "Usage: %s <filename>.mat [-a] [-s] [-o <output_file>.mat] [-jN]\n", program_name);
    fprintf(stderr, "Options:\n");
    fprintf(stderr, "  -s, --sorted          Use sorted data\n");
    fprintf(stderr, "  -a, --approx          For approximate solution\n");
    fprintf(stderr, "  -o, --output <file>   Specify output file (default is stdout)\n");
    fprintf(stderr, "  -jN                   Number of threads (N should be an integer)\n");
}


int parse_arguments(int argc, char *argv[], Options *opts, const char **filename) 
{
    int opt;
    
    // Initialize options
    opts->sorted = 0;              // Default: not sorted
    opts->approx = 0;              // Default: find the exact solution
    opts->output_filename = NULL;  // Default: no output filename
    opts->num_threads = -1;        // Default: automatically determine the number of threads

    // Parse optional arguments
    const char *optstring = "sao:j:";
    const struct option long_options[] = {
        {"sorted", no_argument, NULL, 's'},
        {"approx", no_argument, NULL, 'a'},
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
            case 'a':
                opts->approx = 1;  // Set the approximate solution flag
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
    if (argc - optind != 1) {
        fprintf(stderr, "Expected input file <filename>.mat\n");
        print_usage(argv[0]);
        return EXIT_FAILURE; // Return error
    }

    // Positional arguments
    *filename = argv[optind];

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
