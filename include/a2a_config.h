#ifndef A2A_CONFIG_H
#define A2A_CONFIG_H

#include <float.h>
#include <cblas.h>
#include <math.h>

#ifdef SINGLE_PRECISION
    #define DTYPE float
    #define GEMM cblas_sgemm
    #define DOT cblas_sdot
    #define SQRT sqrtf
    #define SUFFIX(X) X##f
    #define INF FLT_MAX
#else
    #define DTYPE double
    #define GEMM cblas_dgemm
    #define DOT cblas_ddot
    #define SQRT sqrt
    #define SUFFIX(X) X
    #define INF DBL_MAX
#endif

#ifdef DEBUG_CONFIG
    #define DEBUG_PRINT(...) printf(__VA_ARGS__)
    #define DEBUG_ASSERT(cond, msg) if (!(cond)) { printf("Assertion failed: %s\n", msg); abort(); }
#else
    #define DEBUG_PRINT(...) ((void)0)
    #define DEBUG_ASSERT(cond, msg) ((void)0)
#endif

#endif