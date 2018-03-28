#pragma once

// Common functions between all CUDA kernels

#include <cuda.h>
#include <cuda_runtime.h>

#include <stdio.h>

// For checking CUDA errors
#define checkCudaErrors(err) __checkCudaErrors(err, __FILE__, __LINE__)

inline void __checkCudaErrors(cudaError_t err, const char* file, const int line) {
    if (cudaSuccess != err) {
        fprintf(stderr,
                "CUDA Driver API error: %s at file <%s>, line %i.\n",
                cudaGetErrorString(err),
                file,
                line);
        exit(-1);
    }
}

__device__ inline uint2 getPosition() {
    return make_uint2(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);
}

__device__ inline unsigned int convert2dTo1d(const uint2 loc, const size_t numCols) {
    return loc.y * numCols + loc.x;
}
