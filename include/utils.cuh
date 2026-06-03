#pragma once
#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <cuda_runtime.h>
#include <cublas_v2.h>

// ==============================================================================
// Error Checking Macros
// ==============================================================================
#define CHECK_CUDA(call)                                                    \
    do {                                                                    \
        cudaError_t err = call;                                             \
        if (err != cudaSuccess) {                                           \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__    \
                      << " code=" << err << " \"" << cudaGetErrorString(err)\
                      << "\"" << std::endl;                                 \
            exit(EXIT_FAILURE);                                             \
        }                                                                   \
    } while (0)

#define CHECK_LAST_CUDA_ERROR()                                             \
    do {                                                                    \
        cudaError_t err = cudaGetLastError();                               \
        if (err != cudaSuccess) {                                           \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__    \
                      << " code=" << err << " \"" << cudaGetErrorString(err)\
                      << "\"" << std::endl;                                 \
            exit(EXIT_FAILURE);                                             \
        }                                                                   \
    } while (0)

#define CHECK_CUBLAS(call)                                                  \
    do {                                                                    \
        cublasStatus_t status = call;                                       \
        if (status != CUBLAS_STATUS_SUCCESS) {                              \
            std::cerr << "cuBLAS error at " << __FILE__ << ":" << __LINE__  \
                      << " code=" << status << std::endl;                   \
            exit(EXIT_FAILURE);                                             \
        }                                                                   \
    } while (0)

// ==============================================================================
// Utility Functions
// ==============================================================================
template <typename T>
void init_random_matrix(std::vector<T>& mat) {
    std::mt19937 gen(42); 
    std::uniform_real_distribution<T> dist(static_cast<T>(-1.0), static_cast<T>(1.0));
    for (auto& val : mat) {
        val = dist(gen);
    }
}

template <typename T>
bool verify_results(const std::vector<T>& ref, const std::vector<T>& gpu, float tolerance = 1e-4) {
    for (size_t i = 0; i < ref.size(); ++i) {
        if (std::abs(ref[i] - gpu[i]) > tolerance) {
            std::cerr << "Mismatch at index " << i << ": CPU=" << ref[i] << ", GPU=" << gpu[i] << std::endl;
            return false;
        }
    }
    return true;
}