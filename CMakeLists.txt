cmake_minimum_required(VERSION 3.10)
project(annlib C)

# Allow user to specify MATLAB root folder (optional)
set(MATLAB_ROOT "" CACHE PATH "Root directory of MATLAB installation (required for TEST and BENCHMARK configurations)")

# User build configuration selection
set(BUILD_CONFIGURATION "LIBRARY" CACHE STRING "Build configuration: LIBRARY, TEST, or BENCHMARK")
set_property(CACHE BUILD_CONFIGURATION PROPERTY STRINGS LIBRARY TEST BENCHMARK)

message(STATUS "Build configuration: ${BUILD_CONFIGURATION}")

# Always find OpenBLAS
find_package(OpenBLAS REQUIRED)

# Find all core library sources
file(GLOB_RECURSE SRC_FILES "${CMAKE_CURRENT_SOURCE_DIR}/src/*.c")

# === Create double precision library ===
add_library(annlib_d ${SRC_FILES})
target_compile_definitions(annlib_d PRIVATE DOUBLE_DTYPE)
target_include_directories(annlib_d
    PUBLIC
        ${OpenBLAS_INCLUDE_DIRS}
        "${PROJECT_SOURCE_DIR}/include"
)
target_link_libraries(annlib_d
    PUBLIC
        ${OpenBLAS_LIBRARIES}
)

# === Create single precision library ===
add_library(annlib_f ${SRC_FILES})
target_compile_definitions(annlib_f PRIVATE FLOAT_DTYPE)
target_include_directories(annlib_f
    PUBLIC
        ${OpenBLAS_INCLUDE_DIRS}
        "${PROJECT_SOURCE_DIR}/include"
)
target_link_libraries(annlib_f
    PUBLIC
        ${OpenBLAS_LIBRARIES}
)

if(BUILD_CONFIGURATION STREQUAL "TEST" OR BUILD_CONFIGURATION STREQUAL "BENCHMARK")

    if(NOT MATLAB_ROOT)
        message(FATAL_ERROR "MATLAB_ROOT must be specified for TEST and BENCHMARK configurations.")
    endif()

    # MATLAB includes and libraries
    set(MATLAB_INCLUDE_DIR "${MATLAB_ROOT}/extern/include")
    set(MATLAB_LIB_DIR "${MATLAB_ROOT}/bin/glnxa64")

    set(MATLAB_LIB_MAT "${MATLAB_LIB_DIR}/libmat.so")
    set(MATLAB_LIB_MX "${MATLAB_LIB_DIR}/libmx.so")
    set(MATLAB_LIB_MEX "${MATLAB_LIB_DIR}/libmex.so")

    # Find HDF5 in both test and benchmark configurations
    find_package(HDF5 REQUIRED COMPONENTS C)

endif()

if(BUILD_CONFIGURATION STREQUAL "TEST")

    # Test sources
    file(GLOB TEST_SOURCES 
        "${PROJECT_SOURCE_DIR}/test/*.c" 
        "${PROJECT_SOURCE_DIR}/util/*.c"
    )

    add_executable(tests
        ${TEST_SOURCES}
    )
    target_compile_definitions(tests PRIVATE DOUBLE_DTYPE)
    target_include_directories(tests
        PRIVATE
            "${MATLAB_INCLUDE_DIR}"
            ${HDF5_INCLUDE_DIRS}
            "${PROJECT_SOURCE_DIR}/test"
            "${PROJECT_SOURCE_DIR}/include"
            "${PROJECT_SOURCE_DIR}/util"
    )
    target_link_libraries(tests
        PRIVATE
            annlib_d
            ${MATLAB_LIB_MAT}
            ${MATLAB_LIB_MX}
            ${MATLAB_LIB_MEX}
            ${HDF5_LIBRARIES}
    )

elseif(BUILD_CONFIGURATION STREQUAL "BENCHMARK")

    # Benchmark sources
    file(GLOB BENCHMARK_SOURCES 
        "${PROJECT_SOURCE_DIR}/benchmark/*.c" 
        "${PROJECT_SOURCE_DIR}/util/*.c"
    )

    add_executable(benchmarks
        ${BENCHMARK_SOURCES}
    )
    target_compile_definitions(benchmarks PRIVATE FLOAT_DTYPE)
    target_include_directories(benchmarks
        PRIVATE
            "${MATLAB_INCLUDE_DIR}"
            ${HDF5_INCLUDE_DIRS}
            "${PROJECT_SOURCE_DIR}/benchmark"
            "${PROJECT_SOURCE_DIR}/include"
            "${PROJECT_SOURCE_DIR}/util"
    )
    target_link_libraries(benchmarks
        PRIVATE
            annlib_f
            ${MATLAB_LIB_MAT}
            ${MATLAB_LIB_MX}
            ${MATLAB_LIB_MEX}
            ${HDF5_LIBRARIES}
    )

else() # LIBRARY
    message(STATUS "Building library only. No tests or benchmarks.")
endif()