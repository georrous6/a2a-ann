#!/bin/bash

# Define variables
SOURCE_FILE="benchmarks.c"
BUILD_DIR="../build"
EXECUTABLE_NAME="benchmarks"
SCRIPT_DIR=$(dirname "$0")  # Directory where this script is located

# Check if a file path is provided as an argument
if [[ $# -ne 1 ]]; then
    echo "Usage: $0 <path_to_dataset>"
    exit 1
fi

DATASET="$1"

# Check if the provided file exists
if [[ ! -f "$DATASET" ]]; then
    echo "Error: Dataset '$DATASET' not found."
    exit 1
fi

# Check if the source file exists
if [[ ! -f "$SOURCE_FILE" ]]; then
    echo "Error: Source file '$SOURCE_FILE' not found."
    exit 1
fi

# Ensure the build directory exists
if [[ ! -d "$BUILD_DIR" ]]; then
    mkdir -p "$BUILD_DIR"
fi

# Compile the source file
gcc -fdiagnostics-color=always "$SOURCE_FILE" ../src/*.c -o "$BUILD_DIR/$EXECUTABLE_NAME" -I/opt/OpenBLAS/include -I../include -I/usr/local/MATLAB/R2024b/extern/include -I/usr/lib/x86_64-linux-gnu/hdf5/serial/include -L/usr/local/MATLAB/R2024b/bin/glnxa64 -L/opt/OpenBLAS/lib -L/usr/local/MATLAB/R2024b/sys/os/glnxa64 -L/usr/lib/x86_64-linux-gnu/hdf5/serial -lstdc++ -lopenblas -lm -lpthread -lmat -lmx -lhdf5 -lcurl
if [[ $? -ne 0 ]]; then
    echo "Error: Compilation failed."
    exit 1
fi

# Run the executable
"$BUILD_DIR/$EXECUTABLE_NAME" "$DATASET"
if [[ $? -ne 0 ]]; then
    echo "Error: Execution of the program failed."
    exit 1
fi

echo "Benchmarks with dataset '$DATASET' executed succesfully."
