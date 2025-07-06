# All2All-ANN

A high-performance C library for solving the **All-to-All Approximate Nearest Neighbors** 
problem with parallelization support. The library leverages **POSIX Threads (pthreads)**, 
**OpenMP**, and **OpenCilk** to efficiently compute approximate nearest neighbors across 
large datasets.  

Additionally, it provides a parallelized implementation of the **k-Nearest Neighbors** 
algorithm using **pthreads** for scalable performance on multicore systems.

---

## Requirements

- **CMake** >= 3.10
- **OpenBLAS**
- **MATLAB** (only for `TESTS` configuration)
- **HDF5** (only for `BENCHMARKS` configuration)

---

## Build with CMake

The build process supports three mutually exclusive configurations:

| Configuration | Dependencies |
|---------------|--------------|
| `LIBRARY` (default) | OpenBLAS |
| `TESTS` | OpenBLAS + MATLAB |
| `BENCHMARKS` | OpenBLAS + HDF5 |

The configuration is selected via the `BUILD_CONFIGURATION`.
According to your preffered configuration run:

**Library only:**
```bash
cmake -S . -B build -DBUILD_CONFIGURATION=LIBRARY
cmake --build build
```

**Library + tests:**
```bash
cmake -S . -B build -DBUILD_CONFIGURATION=TESTS -DMATLAB_ROOT=/path/to/MATLAB/R2024b
cmake --build build
```

**Library + benchmarks:**
```bash
cmake -S . -B build -DBUILD_CONFIGURATION=BENCHMARKS
cmake --build build
```
