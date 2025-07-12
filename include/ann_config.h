#ifndef TEMPLATE_DEFINITIONS_H
#define TEMPLATE_DEFINITIONS_H

#include <float.h>
#include <cblas.h>
#include <math.h>

#ifdef SINGLE_PRECISION
    #define DTYPE float
    #define GEMM cblas_sgemm
    #define DOT cblas_sdot
    #define SQRT sqrtf
    #define ZERO 0.0f
    #define INF FLT_MAX
#else
    #define DTYPE double
    #define GEMM cblas_dgemm
    #define DOT cblas_ddot
    #define SQRT sqrt
    #define ZERO 0.0
    #define INF DBL_MAX
#endif

#endif // TEMPLATE_DEFINITIONS_H