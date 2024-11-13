#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "ioutil.h"
#include "knnsearch.h"
#include "ioutil.h"


int main(int argc, char *argv[])
{
    Options opts;
    int CM, CN, QM, QN, K;
    const char *filename, *CNAME, *QNAME;
    if (parse_arguments(argc, argv, &opts, &filename, &CNAME, &QNAME, &K))
    {
        if (opts.output_filename)
        {
            free(opts.output_filename);
        }
        return EXIT_FAILURE;
    }

    printf("Filename: %s\n", filename);
    printf("CNAME: %s\n", CNAME);
    printf("QNAME: %s\n", QNAME);
    printf("K: %d\n", K);
    printf("Sorted: %d\n", opts.sorted);
    printf("Output Filename: %s\n", opts.output_filename ? opts.output_filename : "None");
    printf("Number of Threads: %d\n", opts.num_threads);
    
    double* C = (double *)load_matrix(filename, CNAME, &CM, &CN);
    if (!C)
    {
        free(opts.output_filename);
        return EXIT_FAILURE;
    }

    double* Q = (double *)load_matrix(filename, QNAME, &QM, &QN);
    if (!Q)
    {
        free(C);
        free(opts.output_filename);
        return EXIT_FAILURE;
    }

    if (CN != QN)
    {
        fprintf(stderr, "Invalid dimensions for corpus and queries data\n");
        free(C);
        free(Q);
        free(opts.output_filename);
        return EXIT_FAILURE;
    }

    if (K > CM)
    {
        fprintf(stderr, "Invalid K value; must be smaller or equal to the corpus size i.e. K <= %d\n", CM);
        free(C);
        free(Q);
        free(opts.output_filename);
        return EXIT_FAILURE;
    }

    double *D = (double *)malloc(QM * K * sizeof(double));
    if (!D)
    {
        fprintf(stderr, "Error allocating memory for matrix D\n");
        free(C);
        free(Q);
        free(opts.output_filename);
        return EXIT_FAILURE;
    }

    int *IDX = (int *)malloc(QM * K * sizeof(int));
    if (!IDX)
    {
        fprintf(stderr, "Error allocating memory for matrix IDX\n");
        free(C);
        free(Q);
        free(D);
        free(opts.output_filename);
        return EXIT_FAILURE;
    }

    printf("Number of queries: %d\n", QM);
    printf("Number of corpus data: %d\n", CM);
    printf("Dimension: %d\n", CN);

    clock_t start = clock();
    if (knnsearch(Q, C, IDX, D, QM, CM, QN, K, opts.sorted, -1, opts.approx))
    {
        free(C);
        free(Q);
        free(D);
        free(IDX);
        free(opts.output_filename);
        return EXIT_FAILURE;
    }

    clock_t end = clock();
    printf("Execution time: %lf sec\n", ((double) (end - start)) / CLOCKS_PER_SEC);

    // print_matrix(C, CNAME, CM, CN, DOUBLE_TYPE);
    // print_matrix(Q, QNAME, QM, QN, DOUBLE_TYPE);
    // print_matrix(D, "D", QM, K, DOUBLE_TYPE);
    // print_matrix(IDX, "IDX", QM, K, INT_TYPE);

    free(C);
    free(Q);
    free(D);
    free(IDX);
    free(opts.output_filename);
    return EXIT_SUCCESS;

}