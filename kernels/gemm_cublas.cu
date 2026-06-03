#include "kernels.cuh"
#include "utils.cuh"
#include <type_traits>

template <typename T>
void launch_gemm_cublas(size_t m, size_t n, size_t k, T const* alpha,
                        T const* A, size_t lda, T const* B, size_t ldb,
                        T const* beta, T* C, size_t ldc,
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
template void launch_gemm_cublas<float>(size_t, size_t, size_t, float const*, float const*, size_t, float const*, size_t, float const*, float*, size_t, cudaStream_t);
template void launch_gemm_cublas<double>(size_t, size_t, size_t, double const*, double const*, size_t, double const*, size_t, double const*, double*, size_t, cudaStream_t);