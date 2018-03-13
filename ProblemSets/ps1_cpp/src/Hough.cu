#include <common/CudaCommon.cuh>
#include "Hough.h"

#include <opencv2/core/cuda_devptrs.hpp>
#include <thrust/functional.h>
#include <thrust/device_vector.h>

#include <iostream>
#include <thread>

// Compute Hough transform accumulator matrix and local maxima in Hough space

#define PI 3.14159265
#define MIN_THETA -90
#define MAX_THETA 91
#define THETA_WIDTH (MAX_THETA - MIN_THETA)

__host__ __device__ inline float degToRad(float theta) {
    return theta * PI / 180.f;
}

/**
 * \brief Compute Hough transform accumulator
 * \param edgeMask pointer to matrix containing masked edge points
 * \param diagDist diagonal size of the input matrix
 * \param rhoBinSize size of rho bins
 * \param thetaBinSize size of theta bins
 * \param histo output matrix of histogram values
 */
__global__ void houghAccumulateKernel(const cv::gpu::PtrStepSz<unsigned char> edgeMask,
                                      const size_t diagDist,
                                      const size_t rhoBinSize,
                                      const size_t thetaBinSize,
                                      cv::gpu::PtrStepSz<int> histo) {
    const uint2 threadPos = getPosition();

    // Return if we're outside the bounds of the image, or if this is not a masked point
    if (threadPos.x >= edgeMask.cols || threadPos.y >= edgeMask.rows ||
        edgeMask(threadPos.x, threadPos.y) == 0) {
        return;
    }

    // Iterate over all values of theta and sum up in histogram
    for (float theta = MIN_THETA; theta < MAX_THETA; theta += thetaBinSize) {
        float thetaRad = degToRad(theta);
        int rho = roundf(threadPos.x * cosf(thetaRad) + threadPos.y * sinf(thetaRad)) + diagDist;
        int rhoBin = roundf(rho / rhoBinSize);
        int thetaBin = roundf((theta - MIN_THETA) / thetaBinSize);
        atomicAdd(&histo(rhoBin, thetaBin), 1);
    }
}

/**
 * \brief Find the local maxima of the Hough transform accumulator. If it's a maxima, set the index
 * in the mask output to 1; if not, set to 0.
 */
__global__ void findLocalMaximaKernel(const cv::gpu::PtrStepSz<int> accumulator,
                                      cv::gpu::PtrStepSz<unsigned char> localMaximaMask) {
    const uint2 threadPos = getPosition();

    // Return if outside the bounds of the image
    if (threadPos.x >= accumulator.cols || threadPos.y >= accumulator.rows) {
        return;
    }

    // TODO: use shared memory
    bool isBig = true;
    for (int y = max(0, int(threadPos.y) - 1); y < min(accumulator.rows - 1, threadPos.y + 1); y++) {
        for (int x = max(0, int(threadPos.x) - 1); x < min(accumulator.cols - 1, threadPos.x + 1); x++) {
            if (accumulator(y, x) > accumulator(threadPos.y, threadPos.x)) {
                isBig = false;
            }
        }
    }

    localMaximaMask(threadPos.y, threadPos.x) = isBig ? 1 : 0;
}

template <typename T>
struct Threshold : public thrust::binary_function<T, unsigned int, bool> {
    __host__ __device__ bool operator()(const T val, const unsigned int threshold) {
        return val >= threshold;
    }
};

template <typename T>
struct BinaryMask : public thrust::unary_function<T, bool> {
    __host__ __device__ bool operator()(const T val) {
        return val != 0;
    }
});

// C++ wrappers
void houghAccumulate(const cv::gpu::GpuMat& edgeMask,
                     const size_t rhoBinSize,
                     const size_t thetaBinSize,
                     cv::gpu::GpuMat& accumulator) {
    assert(edgeMask.type() == CV_8UC1);
    static constexpr size_t TILE_SIZE = 16;

    const size_t maxDist =
        ceil(sqrt(edgeMask.rows * edgeMask.rows + edgeMask.cols * edgeMask.cols));
    const size_t rhoBins = (max(size_t(1), size_t(ceil(float(2 * maxDist) / float(rhoBinSize)))));
    const size_t thetaBins =
        (max(size_t(1), size_t(ceil(float(THETA_WIDTH) / float(thetaBinSize)))));
    accumulator.create(rhoBins, thetaBins, CV_32SC1);
    std::cout << "accumulator size = " << accumulator.rows << " x " << accumulator.cols
              << std::endl;

    // Determine block and grid size
    dim3 blocks(max(1, (unsigned int)ceil(float(edgeMask.cols) / float(TILE_SIZE))),
                max(1, (unsigned int)ceil(float(edgeMask.rows) / float(TILE_SIZE))));
    dim3 threads(TILE_SIZE, TILE_SIZE);

    // Launch kernel. cv::gpu::GpuMat types are convertable to cv::gpu::PtrStepSz wrapper types
    houghAccumulateKernel<<<blocks, threads>>>(
        edgeMask, maxDist, rhoBinSize, thetaBinSize, accumulator);
    cudaDeviceSynchronize();
    checkCudaErrors(cudaGetLastError());
}

void houghAccumulate(const cv::Mat& edgeMask,
                     const size_t rhoBinSize,
                     const size_t thetaBinSize,
                     cv::Mat& accumulator) {
    assert(edgeMask.type() == CV_8UC1);
    cv::gpu::GpuMat d_edgeMask, d_accumulator;

    // Copy input to GPU
    d_edgeMask.upload(edgeMask);
    std::cout << "d_edgemask size = " << d_edgeMask.rows << " x " << d_edgeMask.cols << std::endl;

    houghAccumulate(d_edgeMask, rhoBinSize, thetaBinSize, d_accumulator);

    // Copy result back to CPU
    d_accumulator.download(accumulator);
}

void findLocalMaxima(const cv::gpu::GpuMat& accumulator, cv::gpu::GpuMat& localMaximaMask) {
    static constexpr size_t TILE_SIZE = 16;

    localMaximaMask.create(accumulator.rows, accumulator.cols, CV_8UC1);

    // Determine block and grid size
    dim3 blocks(max(1, (unsigned int)ceil(float(accumulator.cols) / float(TILE_SIZE))),
                max(1, (unsigned int)ceil(float(accumulator.rows) / float(TILE_SIZE))));
    dim3 threads(TILE_SIZE, TILE_SIZE);

    findLocalMaximaKernel<<<dim3(1, blocks.y), threads>>>(accumulator, localMaximaMask);
    cudaDeviceSynchronize();
    checkCudaErrors(cudaGetLastError());

    // Filter by threshold, where threshold is the minimum number of votes required to count as a peak

}

void findLocalMaxima(const cv::Mat& accumulator, cv::Mat& localMaximaMask) {
    cv::gpu::GpuMat d_accumulator, d_localMaximaMask;

    // Copy to GPU
    d_accumulator.upload(accumulator);
    findLocalMaxima(d_accumulator, d_localMaximaMask);

    // Copy back to CPU
    d_localMaximaMask.download(localMaximaMask);
}