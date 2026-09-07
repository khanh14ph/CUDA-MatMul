#include "kernels.cuh"
#include "utils.cuh"
#include <type_traits>

template <typename T>
void launch_gemm_cublas(int m, int n, int k, T const* alpha,
                        T const* A, int lda, T const* B, int ldb,
                        T const* beta, T* C, int ldc,
                        cudaStream_t stream)
{
    static cublasHandle_t handle = nullptr;
    if (handle == nullptr) {
        CHECK_CUBLAS(cublasCreate(&handle));
    }
    CHECK_CUBLAS(cublasSetStream(handle, stream));

    if constexpr (std::is_same_v<T, float>) {
        CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, 
                                 n, m, k, alpha, B, ldb, A, lda, beta, C, ldc));
    } else if constexpr (std::is_same_v<T, double>) {
        CHECK_CUBLAS(cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, 
                                 n, m, k, alpha, B, ldb, A, lda, beta, C, ldc));
    }
}

// Explicit template instantiation
template void launch_gemm_cublas<float>(int, int, int, float const*, float const*, int, float const*, int, float const*, float*, int, cudaStream_t);
template void launch_gemm_cublas<double>(int, int, int, double const*, double const*, int, double const*, int, double const*, double*, int, cudaStream_t);