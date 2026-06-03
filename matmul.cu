#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <chrono>
#include <cuda_runtime.h>
#include <cublas_v2.h> // 1. Include cuBLAS Header

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

// 2. cuBLAS Error Checking Macro
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
// CPU Reference Implementation for Correctness Checking
// ==============================================================================
template <typename T>
void cpu_gemm(size_t m, size_t n, size_t k, T alpha, T const* A, size_t lda,
              T const* B, size_t ldb, T beta, T* C, size_t ldc) {
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < n; ++j) {
            T sum = 0;
            for (size_t p = 0; p < k; ++p) {
                sum += A[i * lda + p] * B[p * ldb + j];
            }
            C[i * ldc + j] = alpha * sum + beta * C[i * ldc + j];
        }
    }
}

// ==============================================================================
// Your Kernels Go Here
// ==============================================================================
template <typename T>
__global__ void gemm_v00(size_t m, size_t n, size_t k, T alpha, T const* A,
                         size_t lda, T const* B, size_t ldb, T beta, T* C,
                         size_t ldc)
{
    size_t const C_row_idx{blockIdx.x * blockDim.x + threadIdx.x};
    size_t const C_col_idx{blockIdx.y * blockDim.y + threadIdx.y};

    if (C_row_idx < m && C_col_idx < n)
    {
        T sum{static_cast<T>(0)};
        for (size_t k_idx{0U}; k_idx < k; ++k_idx)
        {
            sum += A[C_row_idx * lda + k_idx] * B[k_idx * ldb + C_col_idx];
        }
        C[C_row_idx * ldc + C_col_idx] =
            alpha * sum + beta * C[C_row_idx * ldc + C_col_idx];
    }
}

template <typename T>
void launch_gemm_kernel_v00(size_t m, size_t n, size_t k, T const* alpha,
                            T const* A, size_t lda, T const* B, size_t ldb,
                            T const* beta, T* C, size_t ldc,
                            cudaStream_t stream)
{
    dim3 const block_dim{32U, 32U, 1U};
    dim3 const grid_dim{
        (static_cast<unsigned int>(m) + block_dim.x - 1U) / block_dim.x,
        (static_cast<unsigned int>(n) + block_dim.y - 1U) / block_dim.y, 1U};
    
    gemm_v00<T><<<grid_dim, block_dim, 0U, stream>>>(m, n, k, *alpha, A, lda, B,
                                                     ldb, *beta, C, ldc);
    CHECK_LAST_CUDA_ERROR();
}

// 3. cuBLAS Wrapper matching your benchmarking function pointer signature
template <typename T>
void launch_gemm_cublas(size_t m, size_t n, size_t k, T const* alpha,
                        T const* A, size_t lda, T const* B, size_t ldb,
                        T const* beta, T* C, size_t ldc,
                        cudaStream_t stream)
{
    // Using a static handle to avoid creation overhead during benchmark iterations
    static cublasHandle_t handle = nullptr;
    if (handle == nullptr) {
        CHECK_CUBLAS(cublasCreate(&handle));
    }
    CHECK_CUBLAS(cublasSetStream(handle, stream));

    // cuBLAS assumes column-major layout. 
    // To calculate row-major C = alpha*A*B + beta*C, we compute its transpose:
    // C^T = alpha * B^T * A^T + beta * C^T
    if constexpr (std::is_same_v<T, float>) {
        CHECK_CUBLAS(cublasSgemm(handle, 
                                 CUBLAS_OP_N, CUBLAS_OP_N, 
                                 n, m, k, 
                                 alpha, 
                                 B, ldb, 
                                 A, lda, 
                                 beta, 
                                 C, ldc));
    } else if constexpr (std::is_same_v<T, double>) {
        CHECK_CUBLAS(cublasDgemm(handle, 
                                 CUBLAS_OP_N, CUBLAS_OP_N, 
                                 n, m, k, 
                                 alpha, 
                                 B, ldb, 
                                 A, lda, 
                                 beta, 
                                 C, ldc));
    }
}


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

// ==============================================================================
// Benchmarking Logic
// ==============================================================================
template <typename T>
void benchmark_kernel(
    void (*launch_kernel)(size_t, size_t, size_t, T const*, T const*, size_t, T const*, size_t, T const*, T*, size_t, cudaStream_t),
    const char* kernel_name,
    size_t m, size_t n, size_t k,
    T alpha, T beta,
    T* d_A, T* d_B, T* d_C) 
{
    const int WARMUP_ITERS = 3;
    const int BENCH_ITERS = 10;
    
    cudaStream_t stream = nullptr;

    // Warm-up
    for (int i = 0; i < WARMUP_ITERS; ++i) {
        launch_kernel(m, n, k, &alpha, d_A, k, d_B, n, &beta, d_C, n, stream);
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    // Setup CUDA Events for precise timing
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    // Record start, loop, record stop
    CHECK_CUDA(cudaEventRecord(start, stream));
    for (int i = 0; i < BENCH_ITERS; ++i) {
        launch_kernel(m, n, k, &alpha, d_A, k, d_B, n, &beta, d_C, n, stream);
    }
    CHECK_CUDA(cudaEventRecord(stop, stream));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
    
    float avg_ms = ms / BENCH_ITERS;
    
    // Total floating point operations for GEMM: 2 * M * N * K
    double tflops = (2.0 * m * n * k) / (avg_ms * 1e6);

    std::cout << "[Benchmark] " << kernel_name 
              << " | Avg Time: " << avg_ms << " ms "
              << " | Performance: " << tflops << " TFLOPS\n";

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
}

// ==============================================================================
// Main Execution
// ==============================================================================
int main() {
    using T = float;

    size_t m = 1024;
    size_t n = 1024;
    size_t k = 1024;

    T alpha = 1.0f;
    T beta = 0.0f;

    std::cout << "Allocating memory for Matrices (" << m << "x" << n << ")...\n";

    // 1. Allocate Host Memory
    std::vector<T> h_A(m * k);
    std::vector<T> h_B(k * n);
    std::vector<T> h_C_cpu_ref(m * n, 0.0f);
    std::vector<T> h_C_gpu_res(m * n, 0.0f);

    // 2. Initialize Host Matrices
    init_random_matrix(h_A);
    init_random_matrix(h_B);

    // 3. Allocate Device Memory
    T *d_A, *d_B, *d_C;
    CHECK_CUDA(cudaMalloc(&d_A, m * k * sizeof(T)));
    CHECK_CUDA(cudaMalloc(&d_B, k * n * sizeof(T)));
    CHECK_CUDA(cudaMalloc(&d_C, m * n * sizeof(T)));

    // 4. Copy data from Host to Device
    CHECK_CUDA(cudaMemcpy(d_A, h_A.data(), m * k * sizeof(T), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B.data(), k * n * sizeof(T), cudaMemcpyHostToDevice));
    
    // 5. Run CPU Reference
    std::cout << "Running CPU Reference...\n";
    cpu_gemm(m, n, k, alpha, h_A.data(), k, h_B.data(), n, beta, h_C_cpu_ref.data(), n);

    // 6. Test Kernel Correctness (Naive)
    std::cout << "Verifying V00 Kernel correctness...\n";
    CHECK_CUDA(cudaMemset(d_C, 0, m * n * sizeof(T))); 
    launch_gemm_kernel_v00(m, n, k, &alpha, d_A, k, d_B, n, &beta, d_C, n, nullptr);
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaMemcpy(h_C_gpu_res.data(), d_C, m * n * sizeof(T), cudaMemcpyDeviceToHost));
    
    if (!verify_results(h_C_cpu_ref, h_C_gpu_res)) return -1;

    // 4b. Test cuBLAS Correctness
    std::cout << "Verifying cuBLAS correctness...\n";
    CHECK_CUDA(cudaMemset(d_C, 0, m * n * sizeof(T)));
    launch_gemm_cublas(m, n, k, &alpha, d_A, k, d_B, n, &beta, d_C, n, nullptr);
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaMemcpy(h_C_gpu_res.data(), d_C, m * n * sizeof(T), cudaMemcpyDeviceToHost));

    if (verify_results(h_C_cpu_ref, h_C_gpu_res)) {
        std::cout << "SUCCESS: cuBLAS output matches CPU reference!\n";
    } else {
        std::cout << "FAILED: cuBLAS output does not match CPU reference.\n";
        return -1;
    }

    // 7. Benchmark the Kernel(s)
    std::cout << "\nStarting Benchmarks...\n";
    std::cout << "M=" << m << " N=" << n << " K=" << k << "\n";
    std::cout << "--------------------------------------------------------\n";
    
    benchmark_kernel(launch_gemm_kernel_v00<T>, "gemm_v00 (Naive)", m, n, k, alpha, beta, d_A, d_B, d_C);
    
    // 5b. Benchmark cuBLAS
    benchmark_kernel(launch_gemm_cublas<T>, "cuBLAS Baseline", m, n, k, alpha, beta, d_A, d_B, d_C);

    // 8. Cleanup Device Memory
    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C));

    return 0;
}