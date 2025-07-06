#ifndef IOUTIL_H
#define IOUTIL_H

#include <stddef.h>

#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define MIN(a, b) ((a) < (b) ? (a) : (b))

typedef enum { DOUBLE_TYPE, INT_TYPE } MATRIX_TYPE;


/**
 * Loads a matrix from a MAT file. Supports double and int data types.
 * 
 * @param filename the name of the MAT file
 * @param matname the name of the matrix to be loaded
 * @param rows stores the number of rows of the matrix
 * @param cols stores the number of columns of the matrix
 * @return a pointer to the dynamically allocated array of data
 * or NULL if an error occured
 * @note memory deallocation should take place outside the function
 */
void *readMATFile(const char *filename, const char* matname, int* rows, int* cols);

/**
 * Driver function for loading a 2D matrix from a file.
 * Supported file types are MAT and HDF5.
 * 
 * @param filename the name of the file
 * @param matname the name of the matrix to be loaded
 * @param rows stores the number of rows of the matrix
 * @param cols stores the number of columns of the matrix
 * @return a pointer to the dynamically allocated array of data
 * or NULL if an error occured
 * @note memory deallocation should take place outside the function
 */
void* load_matrix(const char *filename, const char* matname, int* rows, int* cols);


/**
 * Function to save a 2D matrix to a .mat file. Supports double and int data types.
 * If the file already exists it overwrites it and if it doesn't it creates it.
 *
 * @param mat Pointer to the matrix data.
 * @param matname The name of the matrix in the .mat file.
 * @param rows Number of rows in the matrix.
 * @param cols Number of columns in the matrix.
 * @param filename The name of the .mat file to save the matrix.
 * @param type The data type of the matrix (DOUBLE_TYPE or INT_TYPE).
 * @param mode the mode to open the file. 'a' appends the data to file and 'w' overwrites it.
 * @return EXIT_SUCCESS if successful, otherwise EXIT_FAILURE.
 */
int store_matrix(const void* mat, const char* matname, int rows, int cols, const char *filename, MATRIX_TYPE type, const char mode);


/**
 * Prints the matrix to standard output. Supports double and int data types.
 * 
 * @param mat the matrix (void pointer to accommodate different data types)
 * @param name the name of the matrix
 * @param rows the number of rows of the matrix
 * @param cols the number of columns of the matrix
 * @param type the data type of the matrix (DOUBLE_TYPE or int_TYPE)
 */
void print_matrix(const void* mat, const char* name, int rows, int cols, MATRIX_TYPE type);


/**
 * Checks if a file has a specific extension
 * 
 * @param filename the name of the file
 * @param extension the extension
 * @return returns 0 if the file does not have the specified extension
 * and 1 otherwise
 */
int has_extension(const char *filename, const char *extension);


/**
 * Function to compare file paths alphabetically for get_files_sorted.
 * 
 * @param a the first file path
 * @param b the second file path
 * @return 0 if a == b, 1 if a > b and -1 if a < b
 */
int compare_file_paths(const void *a, const void *b);


/**
 * Returns the paths of the files that have a specific extension in a directory.
 * 
 * @param directory_path The path to the directory
 * @param extension The file extension to search for
 * @param file_count The number of files found
 * @param sorted Flag for sorting the file paths. If set to a non negative value,
 * the file paths are returned in ascending order, otherwise no sorting is applied.
 */
char **get_file_paths(const char *directory_path, const char *extension, size_t *file_count, int sorted);

/**
 * Function to read a 2D matrix from an HDF5 file.
 * 
 * @param file_name the name of the HDF5 file
 * @param dataset_name the name of the dataset i.e. the 2D matrix
 * @param rows the rows of the matrix
 * @param cols the columns of the matrix
 * @return a pointer the the matrix data and NULL if an error occured
 */
void *readHDF5File(const char *file_name, const char *dataset_name, int *rows, int *cols);

#endif