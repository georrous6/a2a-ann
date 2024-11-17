#!/bin/bash

# Define variables
SOURCE_FILE="knnsearch_exact_tests.c"
BUILD_DIR="../build"
EXECUTABLE_NAME="knnsearch_exact_tests"
SCRIPT_DIR=$(dirname "$0")  # Directory where this script is located
TEST_DIR="$SCRIPT_DIR/test_files"  # Directory to look for test files

# Check if the source file exists
if [[ ! -f "$SOURCE_FILE" ]]; then
    echo "Error: Source file '$SOURCE_FILE' not found."
    exit 1
fi

# Ensure the build directory exists
if [[ ! -d "$BUILD_DIR" ]]; then
    echo "Build directory not found. Creating '$BUILD_DIR'..."
    mkdir -p "$BUILD_DIR"
fi

# Compile the source file
gcc -fdiagnostics-color=always "$SOURCE_FILE" ../src/*.c -o "$BUILD_DIR/$EXECUTABLE_NAME" -I/opt/OpenBLAS/include -I../include -I/usr/local/MATLAB/R2024b/extern/include -L/usr/local/MATLAB/R2024b/bin/glnxa64 -L/opt/OpenBLAS/lib -L/usr/local/MATLAB/R2024b/sys/os/glnxa64 -lstdc++ -lopenblas -lm -lpthread -lmat -lmx
if [[ $? -ne 0 ]]; then
    echo "Error: Compilation failed."
    exit 1
fi

# Check if the test directory exists or is empty
if [ ! -d "$TEST_DIR" ] || [ -z "$(ls -A "$TEST_DIR" 2>/dev/null)" ]; then
    echo "Generating test files..."
    matlab -batch "generate_test_files; exit;"
    MATLAB_STATUS=$?
    
    if [ "$MATLAB_STATUS" -ne 0 ]; then
        echo "Error: MATLAB script to generate test files failed."
        exit 1
    fi
    
    echo "Test files generated successfully."
fi

# Ensure the directory is still valid after generation
if [ ! -d "$TEST_DIR" ] || [ -z "$(ls -A "$TEST_DIR" 2>/dev/null)" ]; then
    echo "Error: Test directory is still missing or empty after generation."
    exit 1
fi

# Run the executable
"$BUILD_DIR/$EXECUTABLE_NAME" "$TEST_DIR"
if [[ $? -ne 0 ]]; then
    echo "Error: Execution of the program failed."
    exit 1
fi
