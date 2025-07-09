#ifndef TEMPLATE_DEFINITIONS_H
#define TEMPLATE_DEFINITIONS_H

#ifdef FLOAT_DTYPE
    #define DTYPE float
    #define GEMM cblas_sgemm
    #define DOT cblas_sdot
    #define SQRT sqrtf
#else
    #define DTYPE double
    #define GEMM cblas_dgemm
    #define DOT cblas_ddot
    #define SQRT sqrt
#endif

#endif // TEMPLATE_DEFINITIONS_H