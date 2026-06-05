cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(GEMM_Benchmark LANGUAGES CXX CUDA)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find the CUDA Toolkit (specifically for cuBLAS)
find_package(CUDAToolkit REQUIRED)

# Add Include directories so the compiler can find utils.cuh and kernels.cuh
include_directories(include)

# Define the executable and its source files
add_executable(benchmark
    src/main.cu
    kernels/gemm_v0.cu
    kernels/gemm_cublas.cu
)

# Link the cuBLAS library
target_link_libraries(benchmark PRIVATE CUDA::cublas)