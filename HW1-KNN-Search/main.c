#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "ioutil.h"
#include "knnsearch.h"
#include "ioutil.h"


int main(int argc, char *argv[])
{
    Options opts;
    opts.output_filename = NULL;
    int CM, CN, QM, QN;
    const char *filename, *C_NAME, *Q_NAME, *K_NAME, *IDX_NAME, *D_NAME;
    double *C = NULL, *Q = NULL, *D = NULL;
    int *IDX = NULL, *K = NULL;
    int status = EXIT_FAILURE;

    // Parse command line arguments
    if (parse_arguments(argc, argv, &opts, &filename, &C_NAME, &Q_NAME, &K_NAME, &IDX_NAME, &D_NAME)) goto cleanup;
    
    // load matrices from input file
    C = (double *)load_matrix(filename, C_NAME, &CM, &CN);
    if (!C) goto cleanup;
    Q = (double *)load_matrix(filename, Q_NAME, &QM, &QN);
    if (!Q) goto cleanup;
    int a, b;  // dummy variables
    K  = (int *)load_matrix(filename, K_NAME, &a, &b);
    if (!K) goto cleanup;

    if (CN != QN)
    {
        fprintf(stderr, "Invalid dimensions for corpus and queries data\n");
        goto cleanup;
    }

    if (opts.approx == 1)
    {
        *K = (*K > QM) ? QM : *K;
    }
    else
    {
        *K = (*K > CM) ? CM : *K;  // if K is greater than the corpus size, set K to the coprus size
    }

    if (opts.verbose == 1)
    {
        printf("Input Filename: %s\n", filename);
        printf("Output Filename: %s\n", opts.output_filename ? opts.output_filename : "None");
        printf("Size of Corpus: %d\n", CM);
        printf("Size of Queries: %d\n", QM);
        printf("K: %d\n", *K);
        printf("Dimensions: %d\n", CN);
        printf("Sort Distances: %s\n", opts.sorted == 1 ? "Yes" : "No");
        printf("Approximate Solution: %s\n", opts.approx == 1 ? "Yes" : "No");
        printf("Verbose: %s\n", opts.verbose == 1 ? "Yes" : "No");
        printf("Number of Threads: %d\n", opts.num_threads == -1 ? 1 : opts.num_threads);
    }

    clock_t start = clock();
    if (knnsearch(Q, C, &IDX, &D, QM, CM, QN, *K, opts.sorted, opts.num_threads, opts.approx)) goto cleanup;
    clock_t end = clock();
 
    if (opts.output_filename)  // save the results in ouptut file
    {
        if (store_matrix((void *)D, D_NAME, QM, *K, opts.output_filename, DOUBLE_TYPE, 'a')) goto cleanup;
        if (store_matrix((void *)IDX, IDX_NAME, QM, *K, opts.output_filename, INT_TYPE, 'a')) goto cleanup;
        if (opts.verbose == 1) printf("\nOutput data are saved at: %s\n", opts.output_filename);
    }
    else  // no output file specified, diplay the results in standard output
    {
        print_matrix((void *)D, D_NAME, QM, *K, DOUBLE_TYPE);
        print_matrix((void *)IDX, IDX_NAME, QM, *K, INT_TYPE);
    }

    status = EXIT_SUCCESS;

cleanup:
    if (C) free(C);
    if (Q) free(Q);
    if (K) free(K);
    if (D) free(D);
    if (IDX) free(IDX);
    if (opts.output_filename) free(opts.output_filename);

    if (opts.verbose == 1)
    {
        printf("\n-------------------\n");
        if (status == EXIT_SUCCESS)
        {
            printf("Proccess finished successfully. Ellapsed time: %lf sec\n", ((double) (end - start)) / CLOCKS_PER_SEC);
        }
        else
        {
            printf("Proccess terminated unexpectedly\n");
        }
    }

    return status;
}