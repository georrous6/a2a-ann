#!/bin/bash

# Check if dataset path argument is provided
if [ -z "$1" ]; then
    echo "Usage: $0 <dataset-path>"
    exit 1
fi

# Define variables
DATASET_PATH="$1"
SCRIPT_DIR="$(dirname "$0")"                        # Directory where this script is located
EXECUTABLE_PATH="$SCRIPT_DIR/../build/benchmarks"   # Path to the executable

# Extract directory and filename without extension from DATASET_PATH
DATASET_DIR="$(dirname "$DATASET_PATH")"
DATASET_FILENAME="$(basename "$DATASET_PATH")"
DATASET_NAME="${DATASET_FILENAME%.*}"  # remove extension

# Construct benchmark output file name in the same directory as dataset
BENCHMARK_OUTPUT="$DATASET_DIR/benchmark-${DATASET_NAME}.mat"

# Check if the executable exists
if [ ! -f "$EXECUTABLE_PATH" ]; then
    echo "Error: Executable '$EXECUTABLE_PATH' not found."
    echo "Please build the project first using 'make'."
    exit 1
fi

# Check if the dataset file exists
if [ ! -f "$DATASET_PATH" ]; then
    echo "Error: Dataset file '$DATASET_PATH' not found."
    exit 1
fi

matlab -batch "compute_all_to_all_knn('$DATASET_PATH'); exit;"

# Run the executable with the dataset path and benchmark output file as arguments
"$EXECUTABLE_PATH" "$DATASET_PATH" "$BENCHMARK_OUTPUT"
if [ $? -ne 0 ]; then
    echo "Error: Execution of the program failed."
    exit 1
fi

# Call the MATLAB function to plot results, passing the benchmark output file
matlab -batch "plot_benchmarks('$BENCHMARK_OUTPUT'); exit;"

echo "Benchmark completed successfully."
