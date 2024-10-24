cmake_minimum_required(VERSION 3.10)
project(KNN_Search_serial C)

# Set C standard
set(CMAKE_C_STANDARD 11)

# Find OpenBLAS
find_package(OpenBLAS REQUIRED)

# Include OpenBLAS headers
include_directories(${OpenBLAS_INCLUDE_DIRS})

# Find HDF5
#find_package(HDF5 REQUIRED)

# Manually set include and library paths
set(MATIO_INCLUDE_DIRS "/usr/local/include/")
set(MATIO_LIBRARIES "/usr/local/lib/libmatio.so")  # Adjust if necessary

# Include directories
include_directories(${MATIO_INCLUDE_DIRS} ${HDF5_INCLUDE_DIRS})

# Add the executable
add_executable(knnsearch_serial knnsearch_serial.c)

# Link libraries
target_link_libraries(knnsearch_serial ${OpenBLAS_LIBRARIES} ${MATIO_LIBRARIES} ${HDF5_LIBRARIES} m)
