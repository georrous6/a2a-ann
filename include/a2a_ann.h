#ifndef A2A_ANN_H
#define A2A_ANN_H

#include "template_definitions.h"


typedef struct {
    int* indices;
    int count;
} ClusterIndex;


int a2a_annsearch(const DTYPE* C, const int N, const int L, const int K, 
    const int Kc, int* IDX, DTYPE* D, const int max_iter);


#endif // A2A_ANN_H