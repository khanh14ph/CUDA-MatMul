#include "kernels.cuh"
#include "utils.cuh"

// 1D blocktiling: each block computes a BM x BN tile of C, stepping through K
// in BK-wide chunks. Each thread owns a TM-tall column strip of the tile.
// Requires m % BM == 0, n % BN == 0, k % BK == 0 (same assumption as before).
template <typename T, int BM, int BN, int BK, int TM>
__global__ void gemm_v3(int m, int n, int k, T alpha, T const* A, int lda,
                        T const* B, int ldb, T beta, T* C, int ldc) {
    constexpr int NT = BM * BN / TM;  // threads per block
    __shared__ T A_shared[BM * BK];
    __shared__ T B_shared[BK * BN];
    int const tid = threadIdx.x;
    int const threadRow = tid / BN;
    int const threadCol = tid % BN; 

    A += BM * blockIdx.x * lda;
    B += BN * blockIdx.y;
    C += BM * blockIdx.x * ldc + BN * blockIdx.y;

    T threadResult[TM] = {0};

    for (int kk = 0; kk < k; kk += BK) {
        // cooperative load: each thread loads (BM*BK)/NT A elements and
        // (BK*BN)/NT B elements (exactly 1 each for 64/64/8/8)

        for (int s = 0; s < (BM * BK) / NT; ++s) {
            int const idx = tid + s * NT;
            int const r = idx / BK, c = idx % BK;
            A_shared[r * BK + c] = A[r * lda + c];
        }

        for (int s = 0; s < (BK * BN) / NT; ++s) {
            int const idx = tid + s * NT;
            int const r = idx / BN, c = idx % BN;
            B_shared[r * BN + c] = B[r * ldb + c];
        }
        __syncthreads();

        A += BK;
        B += BK * ldb;

        for (int i = 0; i < BK; ++i) {
            T const B_cell = B_shared[i * BN + threadCol];
            for (int r = 0; r < TM; ++r) {
                threadResult[r] += A_shared[(threadRow * TM + r) * BK + i] * B_cell;
            }
        }
        __syncthreads();
    }

    for (int r = 0; r < TM; ++r) {
        int const row = threadRow * TM + r;
        C[row * ldc + threadCol] = alpha * threadResult[r] + beta * C[row * ldc + threadCol];
    }
}

template <typename T>
void launch_gemm_kernel_v3(int m, int n, int k, T const* alpha, T const* A, int lda,
                           T const* B, int ldb, T const* beta, T* C, int ldc,
                           cudaStream_t stream) {
    constexpr int BM = 64, BN = 64, BK = 8, TM = 8;
    constexpr int NT = BM * BN / TM;  // 512 threads

    dim3 const block_size{NT, 1U, 1U};
    dim3 const grid_size((m + BM - 1) / BM, (n + BN - 1) / BN, 1U);
    gemm_v3<T, BM, BN, BK, TM><<<grid_size, block_size, 0U, stream>>>(
        m, n, k, *alpha, A, lda, B, ldb, *beta, C, ldc);
    CHECK_LAST_CUDA_ERROR();
}

template void launch_gemm_kernel_v3<float>(int, int, int, float const*, float const*, int,
                                           float const*, int, float const*, float*, int,
                                           cudaStream_t);
template void launch_gemm_kernel_v3<double>(int, int, int, double const*, double const*, int,
                                            double const*, int, double const*, double*, int,
                                            cudaStream_t);
