cmake_minimum_required(VERSION 3.10)
project(a2a_annlib C)

# Set CMake policies to ensure compatibility
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

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

    # Find HDF5
    find_package(HDF5 REQUIRED COMPONENTS C)

    # === Create single precision library for benchmarks ===
    add_library(anns_debug STATIC ${SRC_FILES})
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
    add_library(annd_debug STATIC ${SRC_FILES})
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
            ${HDF5_INCLUDE_DIRS}
            "${PROJECT_SOURCE_DIR}/include"
            "${PROJECT_SOURCE_DIR}/util"
    )
    target_link_libraries(tests
        PRIVATE
            annd_debug
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
            ${HDF5_INCLUDE_DIRS}
            "${PROJECT_SOURCE_DIR}/include"
            "${PROJECT_SOURCE_DIR}/util"
    )
    target_link_libraries(knn_benchmarks
        PRIVATE
            anns_debug
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
            ${HDF5_INCLUDE_DIRS}
            "${PROJECT_SOURCE_DIR}/include"
            "${PROJECT_SOURCE_DIR}/util"
    )
    target_link_libraries(ann_benchmarks
        PRIVATE
            anns_debug
            ${HDF5_LIBRARIES}
    )

    # === KNN example target ===
    file(GLOB KNN_EXAMPLE_SOURCES 
        "${PROJECT_SOURCE_DIR}/examples/knn_example.c" 
        "${PROJECT_SOURCE_DIR}/util/*.c"
    )

    add_executable(knn_example
        ${KNN_EXAMPLE_SOURCES}
    )
    target_compile_definitions(knn_example 
        PRIVATE 
            SINGLE_PRECISION 
            ${BUILD_CONFIGURATION}_CONFIG
    )
    target_include_directories(knn_example
        PRIVATE
            ${HDF5_INCLUDE_DIRS}
            "${PROJECT_SOURCE_DIR}/include"
            "${PROJECT_SOURCE_DIR}/util"
    )
    target_link_libraries(knn_example
        PRIVATE
            anns_debug
            ${HDF5_LIBRARIES}
    )

    # === ANN example target ===
    file(GLOB ANN_EXAMPLE_SOURCES
        "${PROJECT_SOURCE_DIR}/examples/ann_example.c" 
        "${PROJECT_SOURCE_DIR}/util/*.c"
    )
    add_executable(ann_example
        ${ANN_EXAMPLE_SOURCES}
    )
    target_compile_definitions(ann_example
        PRIVATE 
            SINGLE_PRECISION 
            ${BUILD_CONFIGURATION}_CONFIG
    )
    target_include_directories(ann_example
        PRIVATE
            ${HDF5_INCLUDE_DIRS}
            "${PROJECT_SOURCE_DIR}/include"
            "${PROJECT_SOURCE_DIR}/util"
    )
    target_link_libraries(ann_example
        PRIVATE
            anns_debug
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
