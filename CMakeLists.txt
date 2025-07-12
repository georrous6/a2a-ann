cmake_minimum_required(VERSION 3.10)
project(a2a_annlib C)

# Set CMake policies to ensure compatibility
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

# Set the default matlab root directory to an empty string (used only for DEBUG configuration)
set(MATLAB_ROOT "" CACHE PATH "Root directory of MATLAB installation (required for DEBUG configuration)")

# Set default build configuration to RELEASE
set(BUILD_CONFIGURATION "RELEASE" CACHE STRING "Build configuration: DEBUG or RELEASE")
set_property(CACHE BUILD_CONFIGURATION PROPERTY STRINGS DEBUG RELEASE)

# Set default library precision to DOUBLE (only used for RELEASE configuration)
set(PRECISION "DOUBLE" CACHE STRING "Library precision: DOUBLE or SINGLE")
set_property(CACHE PRECISION PROPERTY STRINGS DOUBLE SINGLE)

message(STATUS "Build configuration: ${BUILD_CONFIGURATION}")

# Always find OpenBLAS
find_package(OpenBLAS REQUIRED)

# Find all core library sources
file(GLOB_RECURSE SRC_FILES "${CMAKE_CURRENT_SOURCE_DIR}/src/*.c")


if("${BUILD_CONFIGURATION}" STREQUAL "DEBUG")

    if(NOT MATLAB_ROOT)
        message(FATAL_ERROR "MATLAB_ROOT must be specified for DEBUG configuration.")
    endif()

    # MATLAB includes and libraries
    set(MATLAB_INCLUDE_DIR "${MATLAB_ROOT}/extern/include")
    set(MATLAB_LIB_DIR "${MATLAB_ROOT}/bin/glnxa64")

    set(MATLAB_LIB_MAT "${MATLAB_LIB_DIR}/libmat.so")
    set(MATLAB_LIB_MX "${MATLAB_LIB_DIR}/libmx.so")
    set(MATLAB_LIB_MEX "${MATLAB_LIB_DIR}/libmex.so")

    # Find HDF5
    find_package(HDF5 REQUIRED COMPONENTS C)

    # === Create single precision library for benchmarks ===
    add_library(anns_debug ${SRC_FILES})
    target_compile_definitions(anns_debug 
        PRIVATE 
            SINGLE_PRECISION 
            ${BUILD_CONFIGURATION}_CONFIG
    )
    target_include_directories(anns_debug
        PUBLIC
            ${OpenBLAS_INCLUDE_DIRS}
            "${PROJECT_SOURCE_DIR}/include"
    )
    target_link_libraries(anns_debug
        PUBLIC
            ${OpenBLAS_LIBRARIES}
    )

    # === Create double precision library for tests ===
    add_library(annd_debug ${SRC_FILES})
    target_compile_definitions(annd_debug 
        PRIVATE 
        DOUBLE_PRECISION 
        ${BUILD_CONFIGURATION}_CONFIG
    )
    target_include_directories(annd_debug
        PUBLIC
            ${OpenBLAS_INCLUDE_DIRS}
            "${PROJECT_SOURCE_DIR}/include"
    )
    target_link_libraries(annd_debug
        PUBLIC
            ${OpenBLAS_LIBRARIES}
    )

    # === Test target ===

    # Test sources
    file(GLOB TEST_SOURCES 
        "${PROJECT_SOURCE_DIR}/test/*.c" 
        "${PROJECT_SOURCE_DIR}/util/*.c"
    )

    add_executable(tests
        ${TEST_SOURCES}
    )
    target_compile_definitions(tests 
        PRIVATE 
        DOUBLE_PRECISION 
        ${BUILD_CONFIGURATION}_CONFIG
    )
    target_include_directories(tests
        PRIVATE
            "${MATLAB_INCLUDE_DIR}"
            ${HDF5_INCLUDE_DIRS}
            "${PROJECT_SOURCE_DIR}/include"
            "${PROJECT_SOURCE_DIR}/util"
    )
    target_link_libraries(tests
        PRIVATE
            annd_debug
            ${MATLAB_LIB_MAT}
            ${MATLAB_LIB_MX}
            ${MATLAB_LIB_MEX}
            ${HDF5_LIBRARIES}
    )

    # === KNN benchmark target ===

    # KNN benchmark sources
    file(GLOB KNN_BENCHMARK_SOURCES 
        "${PROJECT_SOURCE_DIR}/benchmark/knn-benchmark/*.c" 
        "${PROJECT_SOURCE_DIR}/util/*.c"
    )

    add_executable(knn_benchmarks
        ${KNN_BENCHMARK_SOURCES}
    )
    target_compile_definitions(knn_benchmarks 
        PRIVATE 
            SINGLE_PRECISION 
            ${BUILD_CONFIGURATION}_CONFIG
    )
    target_include_directories(knn_benchmarks
        PRIVATE
            "${MATLAB_INCLUDE_DIR}"
            ${HDF5_INCLUDE_DIRS}
            "${PROJECT_SOURCE_DIR}/include"
            "${PROJECT_SOURCE_DIR}/util"
    )
    target_link_libraries(knn_benchmarks
        PRIVATE
            anns_debug
            ${MATLAB_LIB_MAT}
            ${MATLAB_LIB_MX}
            ${MATLAB_LIB_MEX}
            ${HDF5_LIBRARIES}
    )

    # === ANN benchmark target ===

    # ANN benchmark sources
    file(GLOB ANN_BENCHMARK_SOURCES 
        "${PROJECT_SOURCE_DIR}/benchmark/ann-benchmark/*.c" 
        "${PROJECT_SOURCE_DIR}/util/*.c"
    )

    add_executable(ann_benchmarks
        ${ANN_BENCHMARK_SOURCES}
    )
    target_compile_definitions(ann_benchmarks 
        PRIVATE 
            SINGLE_PRECISION 
            ${BUILD_CONFIGURATION}_CONFIG
    )
    target_include_directories(ann_benchmarks
        PRIVATE
            "${MATLAB_INCLUDE_DIR}"
            ${HDF5_INCLUDE_DIRS}
            "${PROJECT_SOURCE_DIR}/include"
            "${PROJECT_SOURCE_DIR}/util"
    )
    target_link_libraries(ann_benchmarks
        PRIVATE
            anns_debug
            ${MATLAB_LIB_MAT}
            ${MATLAB_LIB_MX}
            ${MATLAB_LIB_MEX}
            ${HDF5_LIBRARIES}
    )
else()
    
    add_library(ann STATIC ${SRC_FILES})
    target_compile_definitions(ann 
        PRIVATE 
            ${PRECISION}_PRECISION 
            ${BUILD_CONFIGURATION}_CONFIG
    )
    target_include_directories(ann
        PUBLIC
            ${OpenBLAS_INCLUDE_DIRS}
            "${PROJECT_SOURCE_DIR}/include"
    )
    target_link_libraries(ann
        PUBLIC
            ${OpenBLAS_LIBRARIES}
    )
endif()
