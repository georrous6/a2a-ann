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
- **MATLAB** (only for `TEST` and `BENCHMARK` configuration)
- **HDF5** (only for `TEST` and `BENCHMARK` configuration)

---

## Build with CMake

The build process supports three mutually exclusive configurations:

| Configuration | Dependencies |
|---------------|--------------|
| `LIBRARY` (default) | OpenBLAS |
| `TEST` | OpenBLAS + MATLAB + HDF5 |
| `BENCHMARK` | OpenBLAS + MATLAB + HDF5 |

The configuration is selected via the `BUILD_CONFIGURATION`.
According to your preffered configuration run:

**Library only:**
```bash
cmake -S . -B build -DBUILD_CONFIGURATION=LIBRARY
cmake --build build
```

**Library + tests:**
```bash
cmake -S . -B build -DBUILD_CONFIGURATION=TEST -DMATLAB_ROOT=/path/to/MATLAB/R2024b
cmake --build build
```
To run the tests type
```bash
cd test
chmod +x run_tests.sh
./run_tests.sh
```

**Library + benchmarks:**
```bash
cmake -S . -B build -DBUILD_CONFIGURATION=BENCHMARK -DMATLAB_ROOT=/path/to/MATLAB/R2024b
cmake --build build
```
To run the benchmarks type
```bash
cd benchmark
chmod +x run_benchmarks.sh
./run_benchmarks.sh
```
