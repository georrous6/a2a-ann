#!/bin/bash
set -e  # Exit immediately if any command fails

# Usage check
if [ -z "$1" ]; then
    echo "Usage: $0 <dataset-path>"
    exit 1
fi

# Variables
DATASET_PATH="$1"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCHMARK_OUTPUT="$SCRIPT_DIR/ann_benchmark_output.hdf5"

declare -A EXECUTABLES=(
    [OpenMP]="$SCRIPT_DIR/../../build_openmp/ann_benchmark_openmp"
    [OpenCilk]="$SCRIPT_DIR/../../build_opencilk/ann_benchmark_opencilk"
)

# Check if dataset exists
if [ ! -f "$DATASET_PATH" ]; then
    echo "Error: Dataset file '$DATASET_PATH' not found."
    exit 1
fi

# Check if executables exist
for NAME in "${!EXECUTABLES[@]}"; do
    EXECUTABLE="${EXECUTABLES[$NAME]}"
    if [ ! -f "$EXECUTABLE" ]; then
        echo "Error: $NAME executable not found at '$EXECUTABLE'. Please build it."
        exit 1
    fi
done

# Step 1: Compute ground truth
python3 compute_all_to_all_knn.py "$DATASET_PATH"

# Step 2: Run benchmarks
for NAME in "${!EXECUTABLES[@]}"; do
    EXECUTABLE="${EXECUTABLES[$NAME]}"
    echo "Step 2: Running $NAME benchmark..."
    "$EXECUTABLE" "$DATASET_PATH" "$BENCHMARK_OUTPUT"
done

# Step 3: Plot results
python3 plot_ann_benchmarks.py "$BENCHMARK_OUTPUT"

echo "ANN Benchmark completed successfully."
