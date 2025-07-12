#!/bin/bash

# Define variables
SCRIPT_DIR=$(dirname "$0")  # Directory where this script is located
TEST_DIR="$SCRIPT_DIR/data"  # Directory to look for test files
EXECUTABLE_PATH="$SCRIPT_DIR/../build/tests"  # Path th the executable

# Check if the executable file exists inside the build directory
if [ ! -f "$EXECUTABLE_PATH" ]; then
    echo "Error: Executable '$EXECUTABLE_PATH' not found."
    echo "Please build the project first using 'make'."
    exit 1
fi

# Check if the test directory exists or is empty
if [ ! -d "$TEST_DIR" ] || [ -z "$(ls -A "$TEST_DIR" 2>/dev/null)" ]; then
    echo "Generating test files..."
    python3 generate_tests.py 
    if [ $? -ne 0 ]; then
        echo "Error: Python script 'generate_tests.py' failed."
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
"$EXECUTABLE_PATH" "$TEST_DIR"
if [[ $? -ne 0 ]]; then
    echo "Error: Execution of the program failed."
    exit 1
fi
