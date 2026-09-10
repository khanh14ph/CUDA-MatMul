// nvcc src/gpu_infor.cu -o peak_bw && ./peak_bw          (human readable)
// ./peak_bw --json                                         (one JSON object per line, for scripts)
#include <cstdio>
#include <cstring>
#include <cuda_runtime.h>

// FP32 cores per SM by compute capability (same table as CUDA's deviceQuery sample)
static int cores_per_sm(int major, int minor) {
    switch (major * 10 + minor) {
        case 30: case 35: case 37: return 192;
        case 50: case 52: case 53: return 128;
        case 60: return 64;
        case 61: case 62: return 128;
        case 70: case 72: case 75: return 64;
        case 80: return 64;
        case 86: case 87: case 89: return 128;
        case 90: return 128;
        case 100: case 101: case 120: return 128;
        default: return 128;
    }
}

// FP64 : FP32 throughput ratio. Datacenter parts are 1/2, consumer parts 1/32 or 1/64.
static double fp64_ratio(int major, int minor) {
    int cc = major * 10 + minor;
    if (cc == 60 || cc == 70 || cc == 80 || cc == 90 || cc == 100) return 1.0 / 2;
    if (cc == 86 || cc == 89 || cc == 120) return 1.0 / 64;
    return 1.0 / 32;
}

int main(int argc, char** argv) {
    bool json = argc > 1 && strcmp(argv[1], "--json") == 0;
    int count = 0;
    cudaGetDeviceCount(&count);

    for (int d = 0; d < count; ++d) {
        cudaDeviceProp p;
        cudaGetDeviceProperties(&p, d);

        double bw_gbps   = 2.0 * p.memoryClockRate * 1e3 * (p.memoryBusWidth / 8.0) / 1e9;
        int    cores     = cores_per_sm(p.major, p.minor) * p.multiProcessorCount;
        double fp32_tf   = 2.0 * cores * p.clockRate * 1e3 / 1e12;       // FMA = 2 FLOP
        double fp64_tf   = fp32_tf * fp64_ratio(p.major, p.minor);

        if (json) {
            printf("{\"device\":%d,\"name\":\"%s\",\"cc\":\"%d.%d\",\"sms\":%d,"
                   "\"mem_clock_mhz\":%.0f,\"bus_width_bits\":%d,\"mem_bw_gbps\":%.1f,"
                   "\"core_clock_mhz\":%.0f,\"fp32_tflops\":%.3f,\"fp64_tflops\":%.4f,"
                   "\"l2_mb\":%d}\n",
                   d, p.name, p.major, p.minor, p.multiProcessorCount,
                   p.memoryClockRate / 1e3, p.memoryBusWidth, bw_gbps,
                   p.clockRate / 1e3, fp32_tf, fp64_tf, p.l2CacheSize >> 20);
        } else {
            printf("Device %d: %s (sm_%d%d)\n", d, p.name, p.major, p.minor);
            printf("  Memory clock : %.0f MHz\n", p.memoryClockRate / 1e3);
            printf("  Bus width    : %d bits\n", p.memoryBusWidth);
            printf("  Peak BW      : %.1f GB/s\n", bw_gbps);
            printf("  %d SMs @ %.0f MHz -> %d fp32 cores\n", p.multiProcessorCount, p.clockRate / 1e3, cores);
            printf("  Peak fp32    : %.2f TFLOP/s\n", fp32_tf);
            printf("  Peak fp64    : %.3f TFLOP/s\n", fp64_tf);
            printf("  L2 cache     : %d MB\n\n", p.l2CacheSize >> 20);
        }
    }
    return 0;
}
