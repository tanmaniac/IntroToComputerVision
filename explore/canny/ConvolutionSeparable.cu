/*
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

#include <assert.h>
#include <cooperative_groups.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/copy.h>
#include <thrust/device_vector.h>

#include <iostream>
#include <vector>

////////////////////////////////////////////////////////////////////////////////
// Row convolution filter
////////////////////////////////////////////////////////////////////////////////
#define ROWS_BLOCKDIM_X 16
#define ROWS_BLOCKDIM_Y 4
#define ROWS_RESULT_STEPS 8
#define ROWS_HALO_STEPS 1

__global__ void convolutionRowsKernel(float* d_Dst,
                                      float* d_Src,
                                      float* d_kernel,
                                      int kernelRadius,
                                      int imageW,
                                      int imageH,
                                      int pitch) {
    __shared__ float s_Data[ROWS_BLOCKDIM_Y]
                           [(ROWS_RESULT_STEPS + 2 * ROWS_HALO_STEPS) * ROWS_BLOCKDIM_X];

    // Offset to the left halo edge
    const int baseX =
        (blockIdx.x * ROWS_RESULT_STEPS - ROWS_HALO_STEPS) * ROWS_BLOCKDIM_X + threadIdx.x;
    const int baseY = blockIdx.y * ROWS_BLOCKDIM_Y + threadIdx.y;

    d_Src += baseY * pitch + baseX;
    d_Dst += baseY * pitch + baseX;

// Load main data
#pragma unroll

    for (int i = ROWS_HALO_STEPS; i < ROWS_HALO_STEPS + ROWS_RESULT_STEPS; i++) {
        s_Data[threadIdx.y][threadIdx.x + i * ROWS_BLOCKDIM_X] = d_Src[i * ROWS_BLOCKDIM_X];
    }

// Load left halo
#pragma unroll

    for (int i = 0; i < ROWS_HALO_STEPS; i++) {
        s_Data[threadIdx.y][threadIdx.x + i * ROWS_BLOCKDIM_X] =
            (baseX >= -i * ROWS_BLOCKDIM_X) ? d_Src[i * ROWS_BLOCKDIM_X] : 0;
    }

// Load right halo
#pragma unroll

    for (int i = ROWS_HALO_STEPS + ROWS_RESULT_STEPS;
         i < ROWS_HALO_STEPS + ROWS_RESULT_STEPS + ROWS_HALO_STEPS;
         i++) {
        s_Data[threadIdx.y][threadIdx.x + i * ROWS_BLOCKDIM_X] =
            (imageW - baseX > i * ROWS_BLOCKDIM_X) ? d_Src[i * ROWS_BLOCKDIM_X] : 0;
    }

    // Compute and store results
    __syncthreads();
#pragma unroll

    for (int i = ROWS_HALO_STEPS; i < ROWS_HALO_STEPS + ROWS_RESULT_STEPS; i++) {
        float sum = 0;

#pragma unroll

        for (int j = -kernelRadius; j <= kernelRadius; j++) {
            sum += d_kernel[kernelRadius - j] *
                   s_Data[threadIdx.y][threadIdx.x + i * ROWS_BLOCKDIM_X + j];
        }

        d_Dst[i * ROWS_BLOCKDIM_X] = sum;
    }
}

void convolutionRowsGPU(std::vector<float>& dest,
                        std::vector<float>& source,
                        std::vector<float>& kernel,
                        int imageW,
                        int imageH) {
    thrust::device_vector<float> d_source(source);
    thrust::device_vector<float> d_dest(dest);
    thrust::device_vector<float> d_kernel(kernel);

    dim3 blocks(imageW / (ROWS_RESULT_STEPS * ROWS_BLOCKDIM_X), imageH / ROWS_BLOCKDIM_Y);
    dim3 threads(ROWS_BLOCKDIM_X, ROWS_BLOCKDIM_Y);

    convolutionRowsKernel<<<blocks, threads>>>(d_dest.data().get(),
                                               d_source.data().get(),
                                               d_kernel.data().get(),
                                               d_kernel.size() / 2,
                                               imageW,
                                               imageH,
                                               imageW);
    cudaDeviceSynchronize();
    cudaGetLastError();

    // Copy back to host
    thrust::copy(d_dest.begin(), d_dest.end(), dest.begin());
}

////////////////////////////////////////////////////////////////////////////////
// Column convolution filter
////////////////////////////////////////////////////////////////////////////////
#define COLUMNS_BLOCKDIM_X 16
#define COLUMNS_BLOCKDIM_Y 8
#define COLUMNS_RESULT_STEPS 8
#define COLUMNS_HALO_STEPS 1

__global__ void convolutionColumnsKernel(float* d_Dst,
                                         float* d_Src,
                                         float* d_kernel,
                                         int kernelRadius,
                                         int imageW,
                                         int imageH,
                                         int pitch) {
    __shared__ float
        s_Data[COLUMNS_BLOCKDIM_X]
              [(COLUMNS_RESULT_STEPS + 2 * COLUMNS_HALO_STEPS) * COLUMNS_BLOCKDIM_Y + 1];

    // Offset to the upper halo edge
    const int baseX = blockIdx.x * COLUMNS_BLOCKDIM_X + threadIdx.x;
    const int baseY =
        (blockIdx.y * COLUMNS_RESULT_STEPS - COLUMNS_HALO_STEPS) * COLUMNS_BLOCKDIM_Y + threadIdx.y;
    d_Src += baseY * pitch + baseX;
    d_Dst += baseY * pitch + baseX;

// Main data
#pragma unroll

    for (int i = COLUMNS_HALO_STEPS; i < COLUMNS_HALO_STEPS + COLUMNS_RESULT_STEPS; i++) {
        s_Data[threadIdx.x][threadIdx.y + i * COLUMNS_BLOCKDIM_Y] =
            d_Src[i * COLUMNS_BLOCKDIM_Y * pitch];
    }

// Upper halo
#pragma unroll

    for (int i = 0; i < COLUMNS_HALO_STEPS; i++) {
        s_Data[threadIdx.x][threadIdx.y + i * COLUMNS_BLOCKDIM_Y] =
            (baseY >= -i * COLUMNS_BLOCKDIM_Y) ? d_Src[i * COLUMNS_BLOCKDIM_Y * pitch] : 0;
    }

// Lower halo
#pragma unroll

    for (int i = COLUMNS_HALO_STEPS + COLUMNS_RESULT_STEPS;
         i < COLUMNS_HALO_STEPS + COLUMNS_RESULT_STEPS + COLUMNS_HALO_STEPS;
         i++) {
        s_Data[threadIdx.x][threadIdx.y + i * COLUMNS_BLOCKDIM_Y] =
            (imageH - baseY > i * COLUMNS_BLOCKDIM_Y) ? d_Src[i * COLUMNS_BLOCKDIM_Y * pitch] : 0;
    }

    // Compute and store results
    __syncthreads();
#pragma unroll

    for (int i = COLUMNS_HALO_STEPS; i < COLUMNS_HALO_STEPS + COLUMNS_RESULT_STEPS; i++) {
        float sum = 0;
#pragma unroll

        for (int j = -kernelRadius; j <= kernelRadius; j++) {
            sum += d_kernel[kernelRadius - j] *
                   s_Data[threadIdx.x][threadIdx.y + i * COLUMNS_BLOCKDIM_Y + j];
        }

        d_Dst[i * COLUMNS_BLOCKDIM_Y * pitch] = sum;
    }
}

void convolutionColumnsGPU(std::vector<float>& dest,
                           std::vector<float>& source,
                           std::vector<float>& kernel,
                           int imageW,
                           int imageH) {
    thrust::device_vector<float> d_source(source);
    thrust::device_vector<float> d_dest(dest);
    thrust::device_vector<float> d_kernel(kernel);

    dim3 blocks(imageW / COLUMNS_BLOCKDIM_X, imageH / (COLUMNS_RESULT_STEPS * COLUMNS_BLOCKDIM_Y));
    dim3 threads(COLUMNS_BLOCKDIM_X, COLUMNS_BLOCKDIM_Y);

    convolutionColumnsKernel<<<blocks, threads>>>(d_dest.data().get(),
                                                  d_source.data().get(),
                                                  d_kernel.data().get(),
                                                  d_kernel.size() / 2,
                                                  imageW,
                                                  imageH,
                                                  imageW);
    cudaDeviceSynchronize();
    cudaGetLastError();

    // Copy back to host
    thrust::copy(d_dest.begin(), d_dest.end(), dest.begin());
}
