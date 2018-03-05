#include <common/utils.cuh>

#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <opencv2/core/core.hpp>

#include <iostream>
#include <vector>

#define CUDA_PI_F 3.141592654f

__global__ void naiveGlobalConvolutionKernel(const float* const source,
                                             const float* const kernel,
                                             const size_t sourceRows,
                                             const size_t sourceCols,
                                             const size_t kernelRadius,
                                             float* dest) {
    const uint2 threadPos = getPosition();
    const unsigned int threadLoc = convert2dTo1d(threadPos, sourceCols);

    // Really inefficient global-memory-based convolution
    if (threadPos.x >= sourceCols || threadPos.y >= sourceRows) {
        return;
    }
    float sum = 0.f;
    int halfRadius = kernelRadius / 2;

#pragma unroll
    for (int y = 0; y < kernelRadius; ++y) {
        int kernelY = y - halfRadius;
        int imageWindowY = min(max(threadPos.y + kernelY, int(0)), int(sourceRows - 1));

#pragma unroll
        for (int x = 0; x < kernelRadius; ++x) {
            int kernelX = x - halfRadius;
            int imageWindowX = min(max(threadPos.x + kernelX, int(0)), int(sourceCols - 1));

            float sourceVal = source[imageWindowY * sourceCols + imageWindowX];
            float kernelVal = kernel[y * kernelRadius + x];

            sum += sourceVal * kernelVal;
        }
    }

    // Copy to output
    dest[threadLoc] = (unsigned int)sum;
}

__global__ void makeGaussianKernel(float* gaussian, const size_t kernelRadius, const float sigma) {
    const uint2 threadPos = getPosition();
    const unsigned int threadLoc = convert2dTo1d(threadPos, kernelRadius);

    if (threadPos.x >= kernelRadius || threadPos.y >= kernelRadius) {
        return;
    }

    unsigned int origin = kernelRadius / 2;
    float x = fabsf(float(origin) - float(threadPos.x));
    float y = fabsf(float(origin) - float(threadPos.y));

    float val = (1.f / (2.f * CUDA_PI_F * powf(sigma, 2.f))) *
                expf(-1.f * (powf(x, 2.f) + powf(y, 2.f)) / (2.f * powf(sigma, 2.f)));

    gaussian[threadLoc] = val;
}

void naiveGlobalConvolution(const cv::Mat& source, const cv::Mat& gaussian, cv::Mat& dest) {
    static constexpr size_t TILE_SIZE = 16;

    const size_t rows = source.rows;
    const size_t cols = source.cols;

    // Copy input image to GPU memory
    thrust::device_vector<float> sourceVec(rows * cols);
    for (unsigned int y = 0; y < rows; y++) {
        for (unsigned int x = 0; x < cols; x++) {
            sourceVec[y * cols + x] = source.at<float>(y, x, 0);
        }
    }

    // Copy Gaussian kernel to GPU memory
    const size_t kernelSize = gaussian.cols;
    thrust::device_vector<float> gaussianVec(kernelSize * kernelSize);
    for (unsigned int y = 0; y < kernelSize; y++) {
        for (unsigned int x = 0; x < kernelSize; x++) {
            gaussianVec[y * kernelSize + x] = gaussian.at<float>(y, x, 0);
        }
    }

    // Allocate space for destination
    thrust::device_vector<float> destVec(rows * cols);

    dim3 blocks(max(1, (unsigned int)ceil(float(cols) / float(TILE_SIZE))),
                max(1, (unsigned int)ceil(float(rows) / float(TILE_SIZE))));
    dim3 threads(TILE_SIZE, TILE_SIZE);

    naiveGlobalConvolutionKernel<<<blocks, threads>>>(sourceVec.data().get(),
                                                      gaussianVec.data().get(),
                                                      rows,
                                                      cols,
                                                      kernelSize,
                                                      destVec.data().get());

    // Copy back to destination
    dest = cv::Mat(rows, cols, source.type());
    for (unsigned int y = 0; y < rows; y++) {
        for (unsigned int x = 0; x < cols; x++) {
            dest.at<float>(y, x, 0) = destVec[y * cols + x];
        }
    }
}

void buildGaussian(const size_t kernelRadius, const float sigma, cv::Mat& gaussianMat) {
    // Allocate memory
    thrust::device_vector<float> gaussian(kernelRadius * kernelRadius);

    // Determine block and grid size
    dim3 blocks(1);
    dim3 threads(kernelRadius, kernelRadius);

    // Launch kernel
    makeGaussianKernel<<<blocks, threads>>>(gaussian.data().get(), kernelRadius, sigma);

    std::cout << std::endl;

    // Copy back to OpenCV Mat
    gaussianMat = cv::Mat(kernelRadius, kernelRadius, CV_32FC1);
    for (unsigned int y = 0; y < kernelRadius; y++) {
        for (unsigned int x = 0; x < kernelRadius; x++) {
            gaussianMat.at<float>(y, x, 0) = gaussian[y * kernelRadius + x];
        }
    }
}