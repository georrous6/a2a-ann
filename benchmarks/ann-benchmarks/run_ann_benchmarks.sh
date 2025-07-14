#!/bin/bash

# Check if dataset path argument is provided
if [ -z "$1" ]; then
    echo "Usage: $0 <dataset-path>"
    exit 1
fi

# Define variables
DATASET_PATH="$1"
SCRIPT_DIR="$(dirname "$0")"                              # Directory where this script is located
EXECUTABLE_PATH="$SCRIPT_DIR/../../build/ann_benchmarks"  # Path to the executable

# Construct benchmark output file name in the same directory as dataset
BENCHMARK_OUTPUT="$SCRIPT_DIR/ann_benchmark_output.hdf5"

# Check if the executable exists
if [ ! -f "$EXECUTABLE_PATH" ]; then
    echo "Error: Executable '$EXECUTABLE_PATH' not found."
    echo "Please build the project first using cmake with Debug configuration"
    exit 1
fi

# Check if the dataset file exists
if [ ! -f "$DATASET_PATH" ]; then
    echo "Error: Dataset file '$DATASET_PATH' not found."
    exit 1
fi

# Run the Python script to compute all-to-all KNN
python3 compute_all_to_all_knn.py "$DATASET_PATH"
if [ $? -ne 0 ]; then
    echo "Error: Failed to compute all-to-all KNN."
    exit 1
fi

# Run the executable with the dataset path and benchmark output file as arguments
"$EXECUTABLE_PATH" "$DATASET_PATH" "$BENCHMARK_OUTPUT"
if [ $? -ne 0 ]; then
    echo "Error: Execution of the program failed."
    exit 1
fi

# Call the MATLAB function to plot results, passing the benchmark output file
python3 plot_ann_benchmarks.py "$BENCHMARK_OUTPUT"

echo "ANN benchmark completed successfully."
