# CUDA-MatMul

A step-by-step implementation of CUDA GEMM (General Matrix Multiplication) kernels, benchmarked against cuBLAS.

## Overview

This project implements the operation:

```
C = alpha * A * B + beta * C
```

where A (M×K), B (K×N), and C (M×N) are matrices of type `float` or `double`.

The goal is to progressively optimize custom CUDA kernels and compare their performance to the cuBLAS baseline.

## Project Structure

```
CUDA-MatMul/
├── include/
│   ├── kernels.cuh       # Kernel launch function declarations
│   └── utils.cuh         # Error-checking macros and utility functions
├── kernels/
│   ├── gemm_cublas.cu    # cuBLAS wrapper (baseline)
│   ├── gemm_v0.cu        # Naive kernel: one thread per output element
│   └── gemm_v1.cu        # (WIP) Next optimization
├── src/
│   └── main.cu           # Correctness verification and benchmarking harness
└── CMakeLists.txt
```

## Kernels

| Version | Description | Status |
|---------|-------------|--------|
| cuBLAS  | NVIDIA cuBLAS `Sgemm`/`Dgemm` | Baseline |
| v0      | Naive: 32×32 thread blocks, one thread per output element | Done |
| v1      | TBD | WIP |

## Requirements

- CUDA Toolkit (>= 11.0 recommended)
- CMake >= 3.18
- A CUDA-capable GPU
- C++17 compiler

## Build

```bash
mkdir build && cd build
cmake ..
make -j
```

### Rebuild after editing a kernel

No need to re-run `cmake` — it only recompiles the files you changed and relinks:

```bash
cmake --build build && ./build/benchmark
```

Re-run `cmake` (the configure step) only when you change the build setup itself,
e.g. adding a new `.cu` file to `add_executable(...)` in `CMakeLists.txt`.

```
nvcc -std=c++17 -Iinclude -c kernels/gemm_v2.cu -o /dev/null
```

## Run

```bash
./build/benchmark
```

Sample output:
```
Allocating memory for Matrices (1024x1024)...
Running CPU Reference...
Verifying V0 Kernel correctness...
Verifying cuBLAS correctness...
SUCCESS: cuBLAS output matches CPU reference!

Starting Benchmarks...
M=1024 N=1024 K=1024
--------------------------------------------------------
[Benchmark] gemm_v0 (Naive)   | Avg Time: X.XX ms | Performance: X.XX TFLOPS
[Benchmark] cuBLAS Baseline   | Avg Time: X.XX ms | Performance: X.XX TFLOPS
```

## How It Works

- `src/main.cu` allocates matrices, runs a CPU reference GEMM, verifies GPU kernel correctness, then benchmarks each kernel with 3 warm-up and 10 timed iterations using CUDA events.
- `include/utils.cuh` provides `CHECK_CUDA`, `CHECK_CUBLAS`, `CHECK_LAST_CUDA_ERROR` macros and helper functions for random matrix initialization and result verification (tolerance: 1e-4).
