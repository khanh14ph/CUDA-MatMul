#include "kernels.cuh"
#include "utils.cuh"
#define BLOCK_SIZE 32
template <typename T>
__global__ void gemm_v2(int m, int n, int k, T alpha, T const* A, int lda, T const* B, int ldb, T beta,  T* C, int ldc){
    
    __shared__ T A_shared[BLOCK_SIZE*BLOCK_SIZE];
    __shared__ T B_shared[BLOCK_SIZE*BLOCK_SIZE];
    int tid_col=threadIdx.x;
    int tid_row=threadIdx.y;

    A+=BLOCK_SIZE*blockIdx.x*lda;
    B+=BLOCK_SIZE*blockIdx.y;
    C+=BLOCK_SIZE*blockIdx.x*ldc+BLOCK_SIZE*blockIdx.y;
    T sum=0;
    for (int kk=0;kk<k;kk+=BLOCK_SIZE){
        A_shared[tid_row*BLOCK_SIZE+tid_col]=A[tid_col+tid_row*lda];
        B_shared[tid_row*BLOCK_SIZE+tid_col]=B[tid_col+tid_row*ldb];
        __syncthreads();
        A+=BLOCK_SIZE;
        B+=BLOCK_SIZE*ldb;
        
        for (int i=0;i<BLOCK_SIZE;i++){
            sum+=A_shared[tid_row*BLOCK_SIZE+i]*B_shared[tid_col+i*BLOCK_SIZE];
        }
        __syncthreads();
    }
    C[tid_row*ldc+tid_col]=alpha*sum+beta*C[tid_row*ldc+tid_col];
}

template <typename T>
void launch_gemm_kernel_v2(int m, int n, int k, T const* alpha, T const* A, int lda, T const* B, int ldb, T const* beta,  T* C, int ldc, cudaStream_t stream){
    dim3 const block_size{32U,32U,1U};
    dim3 const grid_size{
        (m+block_size.x-1)/block_size.x,
        (n+block_size.y-1)/block_size.y, 1U};
    gemm_v2<T><<<grid_size, block_size, 0U, stream>>>(m, n, k, *alpha, A, lda, B,
                                                    ldb, *beta, C, ldc);
    CHECK_LAST_CUDA_ERROR();
}
template void launch_gemm_kernel_v2<float>(int, int, int, float const*,
                                           float const*, int, float const*, int,
                                           float const*, float*, int,
                                           cudaStream_t);
template void launch_gemm_kernel_v2<double>(int, int, int, double const*,
                                            double const*, int, double const*,
                                            int, double const*, double*, int,
                                            cudaStream_t);

