#include "kernels.cuh"
#include "utils.cuh"

template <typename T>
__global__ void gemm_v1(size_t m, size_t n, size_t k, T alpha, T const* A,  size_t lda, T const* B, size_t ldb, T beta, T* C, size_t ldc){
    size_t const C_row_idx{blockIdx.x * blockDim.x + threadIdx.x};
    size_t const C_col_idx{blockIdx.y * blockDim.y + threadIdx.y};
    
}