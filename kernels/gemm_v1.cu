#include "kernels.cuh"
#include "utils.cuh"

template <typename T>
__global__ void gemm_v1(int m, int n, int k, T alpha, T const* A, int lda,
                        T const* B, int ldb, T beta, T* C, int ldc)
{
    // Transposed thread mapping vs v0: x -> columns (n), y -> rows (m)
    int const col{static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x)};
    int const row{static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y)};

    if (row < m && col < n)
    {
        // 64-bit offsets computed once; everything else stays 32-bit
        T const* A_row{A + static_cast<size_t>(row) * lda};
        T const* B_col{B + col};
        T* C_ptr{C + static_cast<size_t>(row) * ldc + col};

        T sum{static_cast<T>(0)};
        for (int kk{0}; kk < k; ++kk)
        {
            sum += A_row[kk] * B_col[static_cast<size_t>(kk) * ldb];
        }
        *C_ptr = alpha * sum + beta * *C_ptr;
    }
}

template <typename T>
void launch_gemm_kernel_v1(int m, int n, int k, T const* alpha, T const* A,
                           int lda, T const* B, int ldb, T const* beta, T* C,
                           int ldc, cudaStream_t stream)
{
    dim3 const block_dim{32U, 32U, 1U};
    // x covers columns (n), y covers rows (m) to match the kernel's mapping
    dim3 const grid_dim{
        (static_cast<unsigned int>(n) + block_dim.x - 1U) / block_dim.x,
        (static_cast<unsigned int>(m) + block_dim.y - 1U) / block_dim.y, 1U};

    gemm_v1<T><<<grid_dim, block_dim, 0U, stream>>>(m, n, k, *alpha, A, lda, B,
                                                    ldb, *beta, C, ldc);
    CHECK_LAST_CUDA_ERROR();
}


template void launch_gemm_kernel_v1<float>(int, int, int, float const*,
                                           float const*, int, float const*, int,
                                           float const*, float*, int,
                                           cudaStream_t);
template void launch_gemm_kernel_v1<double>(int, int, int, double const*,
                                            double const*, int, double const*,
                                            int, double const*, double*, int,
                                            cudaStream_t);
