cmake_minimum_required(VERSION 3.10)
project(my_c_library C)

# Allow user to specify MATLAB root folder (e.g., /usr/local/MATLAB/R2024b)
set(MATLAB_ROOT "" CACHE PATH "Root directory of MATLAB installation")

if(NOT MATLAB_ROOT)
    message(FATAL_ERROR "Please specify MATLAB_ROOT path to your MATLAB installation")
endif()

# Construct include and library directories based on MATLAB_ROOT
set(MATLAB_INCLUDE_DIR "${MATLAB_ROOT}/extern/include")
set(MATLAB_BIN_DIR "${MATLAB_ROOT}/bin/${CMAKE_SYSTEM_PROCESSOR}")  # e.g., glnxa64, win64, etc.

# Find all source files recursively
file(GLOB_RECURSE SRC_FILES "${CMAKE_CURRENT_SOURCE_DIR}/src/*.c")

# Create static library target
add_library(my_c_library ${SRC_FILES})

# Include MATLAB headers
target_include_directories(my_c_library 
    PUBLIC 
        "${MATLAB_INCLUDE_DIR}" 
        "${CMAKE_CURRENT_SOURCE_DIR}/include"
)

# Link MATLAB libraries
set(MATLAB_LIB_MAT "${MATLAB_BIN_DIR}/libmat.so")
set(MATLAB_LIB_MX "${MATLAB_BIN_DIR}/libmx.so")
set(MATLAB_LIB_MEX "${MATLAB_BIN_DIR}/libmex.so")

target_link_libraries(my_c_library
    PUBLIC
        ${MATLAB_LIB_MAT}
        ${MATLAB_LIB_MX}
        ${MATLAB_LIB_MEX}
)


# Find OpenBLAS and HDF5 as before
find_package(OpenBLAS REQUIRED)
find_package(HDF5 REQUIRED COMPONENTS C)

target_include_directories(my_c_library
    PUBLIC
        ${OpenBLAS_INCLUDE_DIRS}
        ${HDF5_INCLUDE_DIRS}
)

target_link_libraries(my_c_library
    PUBLIC
        ${OpenBLAS_LIBRARIES}
        ${HDF5_LIBRARIES}
)
