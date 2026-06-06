#pragma once
#include <cuda_runtime.h>
#include <cstddef>

template <typename T>
void launch_gemm_kernel_v0(size_t m, size_t n, size_t k, T const* alpha,
                            T const* A, size_t lda, T const* B, size_t ldb,
                            T const* beta, T* C, size_t ldc,
                            cudaStream_t stream);

template <typename T>
void launch_gemm_cublas(size_t m, size_t n, size_t k, T const* alpha,
                        T const* A, size_t lda, T const* B, size_t ldb,
                        T const* beta, T* C, size_t ldc,
                        cudaStream_t stream);

template <typename T>
void launch_gemm_kernel_v1(size_t m, size_t n, size_t k, T const* alpha,
                            T const* A, size_t lda, T const* B, size_t ldb,
                            T const* beta, T* C, size_t ldc,
                            cudaStream_t stream);

void launch_gemm_kernel_v2(size_t m, size_t n, size_t k, T const* alpha,
                            T const* A, size_t lda, T const* B, size_t ldb,
                            T const* beta, T* C, size_t ldc,
                            cudaStream_t stream);