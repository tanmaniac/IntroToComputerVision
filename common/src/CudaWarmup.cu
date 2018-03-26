#include "../include/common/CudaWarmup.h"
#include "../include/common/CudaCommon.cuh"

// Simple kernel to warm up the GPU before doing a timed kernel launch
__global__ void warmupKernel() {
    int threadId = blockIdx.x * blockDim.x + threadIdx.x;
    float a = 1.f;
    float b = 2.f;

    a = a * threadId;
    b = b * threadId;
}

void common::warmup() {
    dim3 blocks(10, 1, 1);
    dim3 threads(64, 1, 1);

    warmupKernel<<<blocks, threads>>>();
}