#include <iostream>
#include <vector>
#include "utils.cuh"
#include "kernels.cuh"

// CPU Reference Implementation for Correctness Checking
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

// Benchmarking Logic
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

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(cudaEventRecord(start, stream));
    for (int i = 0; i < BENCH_ITERS; ++i) {
        launch_kernel(m, n, k, &alpha, d_A, k, d_B, n, &beta, d_C, n, stream);
    }
    CHECK_CUDA(cudaEventRecord(stop, stream));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
    
    float avg_ms = ms / BENCH_ITERS;
    double tflops = (2.0 * m * n * k) / (avg_ms * 1e6);

    std::cout << "[Benchmark] " << kernel_name 
              << " | Avg Time: " << avg_ms << " ms "
              << " | Performance: " << tflops << " TFLOPS\n";

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
}

int main() {
    using T = float;

    size_t m = 1024;
    size_t n = 1024;
    size_t k = 1024;

    T alpha = 1.0f;
    T beta = 0.0f;

    std::cout << "Allocating memory for Matrices (" << m << "x" << n << ")...\n";

    std::vector<T> h_A(m * k);
    std::vector<T> h_B(k * n);
    std::vector<T> h_C_cpu_ref(m * n, 0.0f);
    std::vector<T> h_C_gpu_res(m * n, 0.0f);

    init_random_matrix(h_A);
    init_random_matrix(h_B);

    T *d_A, *d_B, *d_C;
    CHECK_CUDA(cudaMalloc(&d_A, m * k * sizeof(T)));
    CHECK_CUDA(cudaMalloc(&d_B, k * n * sizeof(T)));
    CHECK_CUDA(cudaMalloc(&d_C, m * n * sizeof(T)));

    CHECK_CUDA(cudaMemcpy(d_A, h_A.data(), m * k * sizeof(T), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B.data(), k * n * sizeof(T), cudaMemcpyHostToDevice));
    
    std::cout << "Running CPU Reference...\n";
    cpu_gemm(m, n, k, alpha, h_A.data(), k, h_B.data(), n, beta, h_C_cpu_ref.data(), n);

    std::cout << "Verifying V00 Kernel correctness...\n";
    CHECK_CUDA(cudaMemset(d_C, 0, m * n * sizeof(T))); 
    launch_gemm_kernel_v00(m, n, k, &alpha, d_A, k, d_B, n, &beta, d_C, n, nullptr);
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaMemcpy(h_C_gpu_res.data(), d_C, m * n * sizeof(T), cudaMemcpyDeviceToHost));
    
    if (!verify_results(h_C_cpu_ref, h_C_gpu_res)) return -1;

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

    std::cout << "\nStarting Benchmarks...\n";
    std::cout << "M=" << m << " N=" << n << " K=" << k << "\n";
    std::cout << "--------------------------------------------------------\n";
    
    benchmark_kernel(launch_gemm_kernel_v00<T>, "gemm_v00 (Naive)", m, n, k, alpha, beta, d_A, d_B, d_C);
    benchmark_kernel(launch_gemm_cublas<T>, "cuBLAS Baseline", m, n, k, alpha, beta, d_A, d_B, d_C);

    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C));

    return 0;
}