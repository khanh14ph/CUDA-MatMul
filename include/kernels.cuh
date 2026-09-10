#pragma once
#include <cuda_runtime.h>
#include <cstddef>

template <typename T>
void launch_gemm_kernel_v0(int m, int n, int k, T const* alpha,
                           T const* A, int lda, T const* B, int ldb,
                           T const* beta, T* C, int ldc,
                           cudaStream_t stream);

template <typename T>
void launch_gemm_cublas(int m, int n, int k, T const* alpha,
                        T const* A, int lda, T const* B, int ldb,
                        T const* beta, T* C, int ldc,
                        cudaStream_t stream);

template <typename T>
void launch_gemm_kernel_v1(int m, int n, int k, T const* alpha,
                           T const* A, int lda, T const* B, int ldb,
                           T const* beta, T* C, int ldc,
                           cudaStream_t stream);
template <typename T>
void launch_gemm_kernel_v2(int m, int n, int k, T const* alpha,
                           T const* A, int lda, T const* B, int ldb,
                           T const* beta, T* C, int ldc,
                           cudaStream_t stream);


template <typename T>
void launch_gemm_kernel_v3(int m, int n, int k, T const* alpha,
                           T const* A, int lda, T const* B, int ldb,
                           T const* beta, T* C, int ldc,
                           cudaStream_t stream);