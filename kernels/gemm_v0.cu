#include "kernels.cuh"
#include "utils.cuh"

template <typename T>
__global__ void gemm_v0(int m, int n, int k, T alpha, T const* A, int lda,
                        T const* B, int ldb, T beta, T* C, int ldc)
{
    int const row = blockIdx.x * blockDim.x + threadIdx.x;
    int const col = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < m && col < n)
    {
        T const* A_row{A + row * lda};
        T const* B_col{B + col};
        T* C_ptr{C + row * ldc + col};

        T sum{0};
        for (int kk{0}; kk < k; ++kk)
        {
            sum += A_row[kk] * B_col[kk * ldb];
        }
        *C_ptr = alpha * sum + beta * *C_ptr;
    }
}

template <typename T>
void launch_gemm_kernel_v0(int m, int n, int k, T const* alpha, T const* A,
                           int lda, T const* B, int ldb, T const* beta, T* C,
                           int ldc, cudaStream_t stream)
{
    dim3 const block_dim{32U, 32U, 1U};
    dim3 const grid_dim{
        (m + block_dim.x - 1U) / block_dim.x,
        (n + block_dim.y - 1U) / block_dim.y, 1U};

    gemm_v0<T><<<grid_dim, block_dim, 0U, stream>>>(m, n, k, *alpha, A, lda, B,
                                                    ldb, *beta, C, ldc);
    CHECK_LAST_CUDA_ERROR();
}

// Explicit template instantiation for the types you intend to use
template void launch_gemm_kernel_v0<float>(int, int, int, float const*,
                                           float const*, int, float const*, int,
                                           float const*, float*, int,
                                           cudaStream_t);
template void launch_gemm_kernel_v0<double>(int, int, int, double const*,
                                            double const*, int, double const*,
                                            int, double const*, double*, int,
                                            cudaStream_t);
