# a2a-ann

![Build Status](https://github.com/georrous6/a2a-ann/actions/workflows/ci-build.yml/badge.svg)
![Test Status](https://github.com/georrous6/a2a-ann/actions/workflows/ci-test.yml/badge.svg)
![Valgrind Status](https://github.com/georrous6/a2a-ann/actions/workflows/ci-valgrind.yml/badge.svg)


**a2a-ann** is a high-performance C library for solving the **All-to-All Approximate Nearest Neighbors (A2A-ANN)** problem. It includes highly parallelized implementations of both:

- **All-to-All Approximate Nearest Neighbors**
- **k-Nearest Neighbors (k-NN)**

The library is optimized for multicore systems using:
- **POSIX Threads (pthreads)**
- **OpenBLAS**
- **OpenMP** (default secondary parallelization method)
- **OpenCilk** (alternative secondary parallelization method)

---

## Table of Contents

- [Requirements](#requirements)
- [Project Structure](#project-structure)
- [Build Options](#build-options)
- [Build Instructions](#build-instructions)
  - [Build with OpenMP](#build-with-openmp)
  - [Build with OpenCilk](#build-with-opencilk)
- [Running Tests](#running-tests)
- [Running Benchmarks](#running-benchmarks)
  - [KNN Benchmarks](#knn-benchmarks)
  - [ANN Benchmarks](#ann-benchmarks)
- [Valgrind](#valgrind)
  - [Install Valgrind](#install-valgrind)
  - [Running Valgrind](#running-valgrind)

---

## Requirements

- **C standard** >= **C11**
- **CMake** >= 3.10
- **OpenBLAS**
- **OpenMP** (enabled by default unless OpenCilk is used)
- **OpenCilk** (can be enabled via a CMake option)
- Optional: **HDF5** (required for running tests and benchmarks in `Debug` configuration)
- Optional: **Python 3** (required for running tests and benchmarks in `Debug` configuration)

---

## Project Structure

- **`.github/`**  
  Contains GitHub Actions workflows for continuous integration and deployment (CI/CD).

- **`.vscode/`**  
  Configuration files to facilitate quick and consistent setup in the VSCode development environment.

- **`benchmarks/`**  
  Contains benchmarking tools and implementations for both k-NN and A2A-ANN algorithms.

- **`docs/`**  
  Documentation, figures, and plots generated from benchmark results.

- **`include/`**  
  Public header files exposing the core library’s API.

- **`src/`**  
  Source files implementing the core functionality of the ANN and k-NN algorithms.

- **`tests/`**  
  Unit and integration tests for verifying the k-NN implementation.

- **`utils/`**  
  Shared utility functions and helper code used across tests and benchmarks.

- **`valgrind/`**   
  Scripts and configuration files for memory analysis using Valgrind.

---

## Build Options

The following table shows the available build options:

| Build Option       | Description                      | Values            | Default Value |
|--------------------|----------------------------------|-------------------|---------------|
| `CMAKE_BUILD_TYPE` | Select build configuration       | `Debug`/`Release` | `Debug`       |
| `PRECISION`        | Set library precision            | `SINGLE`/`DOUBLE` | `DOUBLE`      |
| `USE_OPENCILK`     | Use OpenCilk for parallelization | `ON`/`OFF`        | `OFF`         |

The `Debug` build includes extra targets for testing and benchmarking, while the Release build 
compiles only the main library.

- By default, the library uses **OpenMP** as a secondary parallelization method (along with PTHREADS). 
You can optionally enable **OpenCilk** by setting the `USE_OPENCILK` flag to `ON`. Only one secondary 
parallelization method can be used at a time.

---

## Build Instructions

Clone the repository
```bash
git clone git@github.com:georrous6/a2a-ann.git
cd a2a-ann
```

### Build with OpenMP

By default, the library uses **OpenMP** for parallelization. Make sure to use a compiler that 
supports **OpenMP**, such as GCC:
```bash
CC=gcc cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DPRECISION=SINGLE
cmake --build build
```

### Build with OpenCilk

To use **OpenCilk** instead of OpenMP, enable the `USE_OPENCILK` option. You must use a 
compiler with **OpenCilk** support, such as a custom-built **clang** from the 
[OpenCilk project](https://opencilk.org/):

```bash
CC=/path/to/opencilk/bin/clang cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DPRECISION=SINGLE -DUSE_OPENCILK=ON
cmake --build build
```
Replace `/path/to/opencilk` with the installation path of the custom-built clang.

These will compile the static library `liba2ann.a` in `build/`.

---

## Running Tests

Build the project with `Debug` configuration
```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build
```

You also need to install the Python dependencies for the test scripts:
```bash
pip install -r requirements.txt
```

Then run the test suite as follows:
```bash
cd tests
chmod +x run_tests.sh
./run_tests.sh
```
This script will automatically generate test datasets using Python 3 and execute the tests to 
verify the correctness of the implementation.

## Running Benchmarks
Benchmarks were conducted on Ubuntu 22.04 LTS using a 4-core machine and the
[MNIST dataset](https://github.com/erikbern/ann-benchmarks).

To run the benchmarks you have to perform two seperate builds for OpenMP and OpenCilk
```bash 
CC=gcc cmake -S . -B build_openmp -DCMAKE_BUILD_TYPE=Debug 
cmake --build build_openmp 
CC=/path/to/opencilk/bin/clang cmake -S . -B build_opencilk -DCMAKE_BUILD_TYPE=Debug -DUSE_OPENCILK=ON 
cmake --build build_opencilk
```
Replace `/path/to/opencilk` with the installation path of the custom-built clang. 

Make sure to install the Python requirements for the benchmark scripts
```bash
pip install -r requirements.txt
```

### KNN Benchmarks
To run the benchmarks for the k-Nearset Neighbors solution type
```bash
cd benchmarks/knn-benchmarks
chmod +x run_knn_benchmarks.sh
./run_knn_benchmarks.sh <path/to/dataset>
```
- The benchmark output will be saved to: `benchmarks/knn-benchmarks/knn_benchmark_output.hdf5`
- The benchmark plot will be saved to: `docs/figures/knn_throughput_vs_threads.png`. 

You may also run benchmarks using a custom .hdf5 dataset. The dataset must include the 
following fields:

- `/train`: Corpus matrix, single precision, row-major order
- `/test`: Query matrix, single precision, row-major order
- `/neighbors`: Ground truth indices (int32), row-major order

![KNN throughput vs number of threads](docs/figures/knn_throughput_vs_threads.png)

### ANN Benchmarks
Run
```bash
cd benchmarks/ann-benchmarks
chmod +x run_ann_benchmarks.sh
./run_ann_benchmarks.sh <path/to/dataset>
```
To use a custom .hdf5 dataset, ensure it follows the same format as used in the KNN benchmarks. 
The benchmark process is the following:

- The `train/` and `test/` matrices are being concatenated.
- The exact all-to-all neighbors are computed.
- The value of `K` corresponds to the number of columns in the `neighbors` matrix.

The benchmark output is then validated against the exact solution, and the following files 
are generated:

- Benchmark results: `benchmarks/ann-benchmarks/ann_benchmark_output.hdf5`
- Recall vs. number of clusters plot: `docs/figures/ann_recall_vs_clusters.png`.
- Throughput vs. number of threads plot: `docs/figures/ann_throughput_vs_threads.png`.
- Recall vs. throughput plot: `docs/figures/ann_recall_vs_throughput.png`

![ANN recall vs number of clusters](docs/figures/ann_recall_vs_clusters.png)
![ANN throughput vs number of threads](docs/figures/ann_throughput_vs_threads.png)
![ANN recall vs throughput](docs/figures/ann_recall_vs_throughput.png)

> Note: Running benchmarks may take a significant amount of time (potentially an hour or more) 

## Valgrind

Valgrind is used to detect memory leaks and invalid memory accesses in the codebase. 
To use Valgrind, you first need to install it on your system.

### Install Valgrind

**On Ubuntu/Debian:**
```bash
sudo apt update
sudo apt install valgrind
```

**On macOS (using Homebrew):**
```bash
brew install valgrind
```
> Note: Valgrind's support on macOS is limited and may not be available or fully functional 
on newer macOS versions.

### Running Valgrind

Build the project with `Debug` configuration
```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build
```

Then run
```bash
cd valgrind
chmod +x run_valgrind.sh
./run_valgrind.sh
```
You can review `valgrind_knn_example.log` and `valgrind_ann_example.log` files
to identify memory leaks, invalid reads/writes, or other memory-related issues.
