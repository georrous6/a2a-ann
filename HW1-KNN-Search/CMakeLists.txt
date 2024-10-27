cmake_minimum_required(VERSION 3.10)
project(KNN_Search C)

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
include_directories(${MATIO_INCLUDE_DIRS} ${HDF5_INCLUDE_DIRS} ${CMAKE_CURRENT_SOURCE_DIR}/include)

# Source files
file(GLOB SOURCES src/*.c)

# Add the executable
add_executable(knnsearch_exact knnsearch_exact.c ${SOURCES})

# Link libraries
target_link_libraries(knnsearch_exact ${OpenBLAS_LIBRARIES} ${MATIO_LIBRARIES} ${HDF5_LIBRARIES} m)
