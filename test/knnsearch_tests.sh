#!/bin/bash

# Define variables
SOURCE_FILE="../main.c"
BUILD_DIR="../build"
EXECUTABLE_NAME="knnsearch"
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
gcc -fdiagnostics-color=always "$SOURCE_FILE" ../src/*.c -o "$BUILD_DIR/$EXECUTABLE_NAME" -I/opt/OpenBLAS/include -I../include -I/usr/local/MATLAB/R2024b/extern/include -I/usr/lib/x86_64-linux-gnu/hdf5/serial/include -L/usr/local/MATLAB/R2024b/bin/glnxa64 -L/opt/OpenBLAS/lib -L/usr/local/MATLAB/R2024b/sys/os/glnxa64 -L/usr/lib/x86_64-linux-gnu/hdf5/serial -lstdc++ -lopenblas -lm -lpthread -lmat -lmx -lhdf5
if [[ $? -ne 0 ]]; then
    echo "Error: Compilation failed."
    exit 1
fi

# Check if the test directory exists or is empty
if [ ! -d "$TEST_DIR" ] || [ -z "$(ls -A "$TEST_DIR" 2>/dev/null)" ]; then
    echo "Generating test files..."
    matlab -batch "generate_knn_tests; exit;"
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

TOTAL_TESTS=0
PASSED_TESTS=0

# Loop through all files in the test directory
for FILE in "$TEST_DIR"/*; do
    # Skip if not a regular file
    if [ ! -f "$FILE" ]; then
        continue
    fi

    TOTAL_TESTS=$((TOTAL_TESTS + 1))
    # echo "Processing $FILE..."

    # Run the executable with the file as an argument
    "$BUILD_DIR/$EXECUTABLE_NAME" "$FILE" "C" "Q" "K" "my_IDX" "my_D" "-o$FILE"
    if [ $? -ne 0 ]; then
        echo "Error: Executable failed for $FILE."
        exit 1
    fi

    # Run the MATLAB testing function
    matlab -batch "result = file_testing_function('$FILE'); exit(result);"
    MATLAB_STATUS=$?

    # MATLAB returns 0 for successful execution, 1 otherwise
    if [ "$MATLAB_STATUS" -eq 0 ]; then
        PASSED_TESTS=$((PASSED_TESTS + 1))
        echo -e "\033[1;32mPassed\033[0m"  # Green
    else
        echo -e "\033[1;31mFailed\033[0m"  # Red
    fi
done

echo "Tests passed ($PASSED_TESTS/$TOTAL_TESTS)"

