#!/bin/bash

# Exit immediately if any command fails
set -e

# Check if dataset path argument is provided
if [ -z "$1" ]; then
    echo "Usage: $0 <dataset-path>"
    exit 1
fi

DATASET_PATH="$1"
SCRIPT_DIR="$(dirname "$0")"

# Executable paths
EXECUTABLES=(
    "$SCRIPT_DIR/../../build_openmp/knn_benchmark_openmp"
    "$SCRIPT_DIR/../../build_opencilk/knn_benchmark_opencilk"
)

BENCHMARK_OUTPUT="$SCRIPT_DIR/knn_benchmark_output.hdf5"

# Check if dataset file exists
if [ ! -f "$DATASET_PATH" ]; then
    echo "Error: Dataset file '$DATASET_PATH' not found."
    exit 1
fi

# Check if executables exist
for exec in "${EXECUTABLES[@]}"; do
    if [ ! -f "$exec" ]; then
        echo "Error: Executable '$exec' not found. Please build the project first using CMake with Debug configuration."
        exit 1
    fi
done

# Run each executable
for exec in "${EXECUTABLES[@]}"; do
    echo "Running $(basename "$exec")..."
    if ! "$exec" "$DATASET_PATH" "$BENCHMARK_OUTPUT"; then
        echo "Error: Execution of '$(basename "$exec")' failed."
        exit 1
    fi
done

# Run the plotting script
echo "Generating plots..."
if ! python3 plot_knn_benchmarks.py "$BENCHMARK_OUTPUT"; then
    echo "Error: Plotting script failed."
    exit 1
fi

echo "KNN benchmark completed successfully."
