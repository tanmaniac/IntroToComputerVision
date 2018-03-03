#include <common/utils.cuh>

#include <cuda.h>
#include <cuda_runtime.h>
//#include <math.h>
#include <opencv2/core/core.hpp>
#include <thrust/device_vector.h>

#include <iostream>

#define CUDA_PI_F 3.141592654f

__global__ void naiveGlobalConvolution(const unsigned char* const source,
                                       const float* const kernel,
                                       const size_t sourceRows,
                                       const size_t sourceCols,
                                       const size_t kernelRadius,
                                       unsigned char* dest) {
    const uint2 threadPos = getPosition();
    const unsigned int threadLoc = convert2dTo1d(threadPos, sourceCols);

    // Really inefficient global-memory-based convolution
    if (threadPos.x >= sourceCols || threadPos.y >= sourceRows) {
        return;
    }
    float sum = 0;
    int halfRadius = kernelRadius / 2;
    for (int y = (-1) * halfRadius; y <= halfRadius; y++) {
        for (int x = (-1) * halfRadius; x <= halfRadius; x++) {
            uint2 kernelPos = make_uint2(threadPos.x + x, threadPos.y + y);
            if (kernelPos.x < sourceCols && kernelPos.y < sourceRows) {
                int kernelLoc = (halfRadius + y) * kernelRadius + (halfRadius + x);
                sum += kernel[kernelLoc] * float(source[threadLoc]);
            }
        }
    }

    // Copy to output
    dest[threadLoc] = (unsigned char)sum;
}

__global__ void makeGaussianKernel(float* gaussian, const size_t kernelRadius, const float sigma) {
    const uint2 threadPos = getPosition();
    const unsigned int threadLoc = convert2dTo1d(threadPos, kernelRadius);

    if (threadPos.x >= kernelRadius || threadPos.y >= kernelRadius) {
        printf("woah!\n");
        return;
    }

    unsigned int origin = kernelRadius / 2;
    float x = fabsf(float(origin - threadPos.x));
    float y = fabsf(float(origin - threadPos.y));

    gaussian[threadLoc] = (1 / 2 * CUDA_PI_F * powf(sigma, 2.f)) *
                          expf(-1 * (powf(x, 2) + powf(y, 2)) / (2 * powf(sigma, 2)));
}

void testAdapter() {
    // naiveGlobalConvolution<<<1, 1>>>();
}

void makeGaussianAdapter(const size_t kernelRadius, const float sigma, cv::Mat& gaussianMat) {
    // Allocate memory
    thrust::device_vector<float> gaussian(kernelRadius * kernelRadius);

    // Determine block and grid size
    dim3 blocks(1);
    dim3 threads(kernelRadius, kernelRadius);

    // Launch kernel
    makeGaussianKernel<<<blocks, threads>>>(gaussian.data().get(), kernelRadius, sigma);

    for (const auto& val : gaussian) {
        std::cout << val << " ";
    }
    std::cout << std::endl;

    // Copy back to OpenCV Mat
    gaussianMat.create(kernelRadius, kernelRadius, CV_32FC1);

    /*float* cvPtr = gaussianMat.ptr<float>(0);
    for (size_t i = 0; i < kernelRadius * kernelRadius; i++) {
        cvPtr[4 * i] = gaussian[i];
    }*/
}