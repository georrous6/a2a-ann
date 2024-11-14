#ifndef IOUTIL_H
#define IOUTIL_H

#include <stddef.h>

#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define MIN(a, b) ((a) < (b) ? (a) : (b))

typedef enum { DOUBLE_TYPE, INT_TYPE } MATRIX_TYPE;


typedef struct {
    int sorted;            // Flag for sorted data
    int approx;            // Flag for approximate solution
    char *output_filename; // Output filename
    int num_threads;       // Number of threads
} Options;


/**
 * Loads a matrix from a .mat file. Supports double and int data types.
 * 
 * @param filename the name of the .mat file
 * @param matname the name of the matrix to be loaded
 * @param rows stores the number of rows of the matrix
 * @param cols stores the number of columns of the matrix
 * @return a pointer to the dynamically allocated array of data
 * or NULL if an error occured
 * @note memory deallocation should take place outside the function
 */
void* load_matrix(const char *filename, const char* matname, int* rows, int* cols);


/**
 * Converts a string to int.
 * 
 * @param value the evaluation of the string
 * @param str the string to be evaluated
 * @return EXIT_SUCCESS if the evaluation was successfull and EXIT_FAILURE otherwise
 */
int str2int(int* value, const char *str);


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
 * @return EXIT_SUCCESS if successful, otherwise EXIT_FAILURE.
 */
int store_matrix(const void* mat, const char* matname, int rows, int cols, const char *filename, MATRIX_TYPE type);


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
 * Parses command-line arguments for the K-Nearest Neighbors search program.
 *
 * @param argc The count of command-line arguments passed to the program, including the program name.
 * @param argv An array of strings representing the command-line arguments. 
 *             The first argument (argv[0]) is the name of the program.
 * @param opts A pointer to an Options struct that will be populated with parsed optional arguments:
 *             - sorted: Indicates whether the input data is sorted (1 for true, 0 for false).
 *             - output_filename: A pointer to a string holding the name of the output file. 
 *                               If no output filename is provided, it will be set to NULL.
 *             - num_threads: An integer representing the number of threads to be used, defaulting to 1.
 * @param filename A pointer to a string that will hold the filename for the input data after parsing.
 * @param CNAME A pointer to a string that will hold the corpus matrix name after parsing.
 * @param QNAME A pointer to a string that will hold the queries matrix name after parsing.
 * @param K A pointer to a int variable that will hold the value of K (the number of nearest neighbors).
 * 
 * @return EXIT_SUCCESS (0) on successful parsing of the arguments.
 *         EXIT_FAILURE (1) if there is an error in parsing, including:
 *         - Invalid number of positional arguments.
 *         - Invalid thread count.
 *         - Any other parsing error (indicated by the '?' case in getopt_long).
 *
 * The function supports the following options:
 * - --sorted or -s: A flag indicating whether the input data is sorted. Default is 0 (not sorted).
 * - --output <filename> or -o <filename>: Specifies the output file name. If not specified, 
 *                                          the output will go to standard output.
 * - --threads <N> or -j <N>: Specifies the number of threads to be used. The value must be 
 *                            greater than or equal to 1. Default is 1.
 *
 * The function checks that exactly four positional arguments are provided after the optional arguments:
 * 1. <filename>: The input data file.
 * 2. <CNAME>: The name of the corpus matrix.
 * 3. <QNAME>: The name of the queries matrix.
 * 4. <K>: The number of nearest neighbors.
 *
 * The function converts the value of <K> from a string to int using strtoull.
 */
int parse_arguments(int argc, char *argv[], Options *opts, const char **filename, const char **CNAME, const char **QNAME, int *K);


/**
 * Function to print usage of the program.
 * 
 * @param program_name the name of the program
 */
void print_usage(const char *program_name);


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


#endif