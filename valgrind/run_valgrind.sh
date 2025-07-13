#!/bin/bash
# Get the directory where the script is located
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BUILD_DIR="$SCRIPT_DIR/../build"
KNN_EXEC="$BUILD_DIR/knn_example"
ANN_EXEC="$BUILD_DIR/ann_example"
DATA="data.hdf5"



# Generate data
echo "Generating test data..."
if ! python3 generate_data.py "$DATA"; then
    echo "Error: Failed to generate data."
    exit 1
fi

# Ensure executables exist
if [[ ! -x "$KNN_EXEC" ]]; then
    echo "Error: knn_example executable not found at $KNN_EXEC."
    exit 1
fi

if [[ ! -x "$ANN_EXEC" ]]; then
    echo "Error: ann_example executable not found at $ANN_EXEC."
    exit 1
fi

# Run Valgrind on knn_example
KNN_LOG="$SCRIPT_DIR/valgrind_knn_example.log"
echo "Running Valgrind on knn_example..."

if ! valgrind --leak-check=full --track-origins=yes --show-leak-kinds=all \
     --error-exitcode=1 "$KNN_EXEC" "$DATA" 2> "$KNN_LOG"; then
    echo "Valgrind found errors in knn_example."
    exit 1
fi

# Additional check for error summary
if grep -q "ERROR SUMMARY: [1-9][0-9]* errors" "$KNN_LOG"; then
    echo "Valgrind found errors in knn_example."
    exit 1
fi

echo "Valgrind output saved to $KNN_LOG"

# Run Valgrind on ann_example
ANN_LOG="$SCRIPT_DIR/valgrind_ann_example.log"
echo "Running Valgrind on ann_example..."

if ! valgrind --leak-check=full --track-origins=yes --show-leak-kinds=all \
     --error-exitcode=1 "$ANN_EXEC" "$DATA" 2> "$ANN_LOG"; then
    echo "Valgrind found errors in ann_example."
    exit 1
fi

# Additional check for error summary
if grep -q "ERROR SUMMARY: [1-9][0-9]* errors" "$ANN_LOG"; then
    echo "Valgrind found errors in ann_example."
    exit 1
fi

echo "Valgrind output saved to $ANN_LOG"

echo "No errors found by Valgrind"
exit 0