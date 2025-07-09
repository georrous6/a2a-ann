#include "ann_config.h"
#include <sys/sysinfo.h>
#include <unistd.h>
#include <cblas.h>

void ann_set_num_threads(int n) {

    const long num_cores = sysconf(_SC_NPROCESSORS_ONLN);  // Number of online processors
    if (num_cores < 1) {
        perror("sysconf\n");
        return;
    }

    n = n < 1 ? (int)num_cores : n;

    atomic_store(&ANN_NUM_TREADS, n);
    if (n > 1) {
        openblas_set_num_threads(1);
    }
    else { // n == 1
        openblas_set_num_threads((int)num_cores);
    }
}


void ann_set_num_threads_cblas(int n) {

    const long num_cores = sysconf(_SC_NPROCESSORS_ONLN);  // Number of online processors
    if (num_cores < 1) {
        perror("sysconf\n");
        return;
    }

    n = n < 1 ? (int)num_cores : n;

    openblas_set_num_threads(n);
}


int get_num_threads_cblas(void) {
    int num_threads = openblas_get_num_threads();
    if (num_threads < 1) {
        perror("openblas_get_num_threads\n");
        return 1;  // Default to 1 thread if the call fails
    }
    return num_threads;
}


int ann_get_num_threads(void) {
    return atomic_load(&ANN_NUM_TREADS);
}


unsigned long get_available_memory_bytes() 
{
    unsigned long available_memory = 0;
    struct sysinfo info;
    if (sysinfo(&info) == 0) 
    {
        available_memory = info.freeram * info.mem_unit;  // Multiply by unit size
    }
    return available_memory;
}
