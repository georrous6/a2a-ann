# all2all-ann

**all2all-ann** is a high-performance C library for solving the **All-to-All Approximate Nearest Neighbors (A2A-ANN)** problem. It includes highly parallelized implementations of both:

- **All-to-All Approximate Nearest Neighbors**
- **k-Nearest Neighbors (k-NN)**

The library is optimized for multicore systems using:
- **POSIX Threads (pthreads)**
- **OpenBLAS**
- Optional: **MATLAB** and **HDF5** for testing and benchmarking utilities

---

## Requirements

- **CMake** >= 3.10
- **OpenBLAS**
- **MATLAB** (required for `DEBUG` configuration -- used in tests and benchmarks)
- **HDF5** (required for `DEBUG` configuration)

> *Attetion*: In `DEBUG` mode, `MATLAB_ROOT` must be specified via `-DMATLAB_ROOT=/path/to/MATLAB`.

---

## Project Structure

- **`benchmark/`**  
  Contains benchmarking tools and implementations for both k-NN and A2A-ANN algorithms.

- **`docs/`**  
  Documentation, figures, and plots generated from benchmark results.

- **`include/`**  
  Public header files for the core library.

- **`src/`**  
  Source files implementing the core functionality of the ANN and k-NN algorithms.

- **`test/`**  
  Unit and integration tests for verifying the k-NN implementation.

- **`util/`**  
  Shared utility functions and helper code used across tests and benchmarks.


## Build Instructions

This project supports two build configurations:

| Build Type     | Description                           | Dependencies               |
|----------------|---------------------------------------|----------------------------|
| `RELEASE`      | Build the standalone library only     | OpenBLAS                   |
| `DEBUG`        | Build tests and benchmarks (with MATLAB + HDF5 support) | OpenBLAS + MATLAB + HDF5 |

You can select the build mode using the `BUILD_CONFIGURATION` flag.

---

### Building the Library (RELEASE)

You can specify the precision of the library by setting the `PRECISION` flag to `SINGLE` or `DOUBLE`.
The default configuration is `DOUBLE`.

```bash
cmake -S . -B build -DBUILD_CONFIGURATION=RELEASE -DPRECISION=SINGLE
cmake --build build
```
This will compile the static library `libann.a` in `build/`.

### Building with Tests and Benchmarks (DEBUG)

```bash
cmake -S . -B build -DBUILD_CONFIGURATION=DEBUG -DMATLAB_ROOT=/path/to/MATLAB/R2024b
cmake --build build
```
Replace `path/to/MATLAB/R2024b` with your actual MATLAB installation path -- 
usually `usr/local/MATLAB/R2024b` on Linux.

## Running Tests

After building in `DEBUG` mode:
```bash
cd test
chmod +x run_tests.sh
./run_tests.sh
```
This will automatically generate test files using MATLAB and run tests against those datasets to 
verify correctness.

## Running Benchmarks
Benchmarks were conducted on Ubuntu 22.04 LTS using a 4-core machine and the
[MNIST dataset](https://github.com/erikbern/ann-benchmarks).

### KNN Benchmarks
After building in `DEBUG` mode, run the benchmark script
```bash
cd benchmark/knn-benchmark
chmod +x run_knn_benchmarks.sh
./run_knn_benchmarks.sh <path/to/dataset>
```
- The benchmark output will be saved to: `benchmark/knn-benchmark/knn_benchmark_output.mat`
- The benchmark plot will be saved to: `docs/figures/throughtput_vs_threads.png`. 

You may also run benchmarks using a custom .hdf5 dataset. The dataset must include the 
following fields:

- `/train`: Corpus matrix, single precision, row-major order
- `/test`: Query matrix, single precision, row-major order
- `/neighbors`: Ground truth indices (int32), row-major order

![knn benchmarks](docs/figures/knn_throughput_vs_threads.png)

### ANN Benchmarks
Again, after building in `DEBUG` mode, run the following
```bash
cd benchmark/ann-benchmark
chmod +x run_ann_benchmarks.sh
./run_ann_benchmarks.sh <path/to/dataset>
```