#!/bin/bash

# Check for proper arguments
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 /path/to/executable"
    exit 1
fi

EXECUTABLE=$1
SCRIPT_DIR=$(dirname "$0")  # Directory where this script is located
TEST_DIR="$SCRIPT_DIR/knn_tests"  # Directory to look for test files

# Ensure the executable exists and is executable
if [ ! -x "$EXECUTABLE" ]; then
    echo "Error: $EXECUTABLE is not executable or does not exist."
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
    $EXECUTABLE "$FILE" "C" "Q" "K" "my_IDX" "my_D" "-o$FILE"
    if [ $? -ne 0 ]; then
        echo "Error: Executable failed for $FILE."
        exit 1
    fi

    # Run the MATLAB testing function
    matlab -batch "result = knn_testing_function('$FILE'); exit(result);"
    MATLAB_STATUS=$?

    # MATLAB returns 0 for successful execution, 1 otherwise
    if [ "$MATLAB_STATUS" -eq 0 ]; then
        PASSED_TESTS=$((PASSED_TESTS + 1))
        echo -e "\033[0;32mPassed\033[0m"  # Green
    else
        echo -e "\033[0;31mFailed\033[0m"  # Red
    fi
done

echo "Tests passed ($PASSED_TESTS/$TOTAL_TESTS)"

