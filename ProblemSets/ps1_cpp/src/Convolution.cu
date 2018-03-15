// Separable convolution kernel - should be faster than the OpenCV version
#include <common/CudaCommon.cuh>
#include "Convolution.h"

#include <cuda.h>
#include <thrust/device_vector.h>
#include <opencv2/core/cuda.hpp>

#include <cassert>
#include <thread>

// Row convolution. This step should be run before column convolution
// input - input image
// kernel - first row of the *separable* filter
// numRows - number of rows in the input image
// numCols - number of columns in the input image
// kernelWidth - width of the input filter
// scale - scaling factor of kernel
// dest - destination image
__global__ void rowConvolutionKernel(const float* const input,
                                     const float* const kernel,
                                     const size_t numRows,
                                     const size_t numCols,
                                     const size_t kernelWidth,
                                     const float scale,
                                     float* dest) {
    const uint2 threadPos = getPosition();

    // Quit if we exceed the bounds of the image
    if (threadPos.x >= numCols || threadPos.y >= numRows) {
        return;
    }

    const unsigned int threadLoc = convert2dTo1d(threadPos, numCols);
    const size_t kernelRadius = kernelWidth / 2;

    // Copy to shared memory
    extern __shared__ float shmem[];
    // Store the kernel in the first kernelWidth indices, and the actual row values in the rest of
    // the allocated memory
    float* shKernel = shmem;
    // TODO: This means that I start using negative array indexing later - not sure if this is
    // generally considered good practice @tanmaniac
    float* shRowData = shmem + kernelWidth + kernelRadius;

    // Only need the first kernelWidth number of threads to copy the kernel
    if (threadIdx.x < kernelWidth) {
        shKernel[threadIdx.x] = kernel[threadIdx.x];
    }
    // Copy pixels to shared mem
    shRowData[threadIdx.x] = input[threadLoc];

    // Get left edge pixels
    if (threadIdx.x == 0) {
        for (int i = -1; i >= -1 * int(kernelRadius); i--) {
            shRowData[i] = (threadPos.x != 0) ? input[threadLoc + i] : input[threadLoc];
        }
    }
    // Get right edge pixels
    if (threadIdx.x == blockDim.x - 1 || threadPos.x == numCols - 1) {
        for (int i = 1; i <= kernelRadius; i++) {
            shRowData[threadIdx.x + i] =
                (threadPos.x != numCols - 1) ? input[threadLoc + i] : input[threadLoc];
        }
    }
    __syncthreads();

    // Iterate over the kernel and sum up the values
    float sum = 0.f;
    for (int idx = 0; idx < kernelWidth; idx++) {
        sum += shKernel[idx] * shRowData[threadIdx.x + idx - kernelRadius];
    }

    dest[threadLoc] = sum;
}

// Column convolution. This should be run after row convolution.
// input - input image
// kernel - first row of the *separable* filter
// numRows - number of rows in the input image
// numCols - number of columns in the input image
// kernelWidth - width of the input filter
// scale - scaling factor of kernel
// dest - destination image
__global__ void columnConvolutionKernel(const float* const input,
                                        const float* const kernel,
                                        const size_t numRows,
                                        const size_t numCols,
                                        const size_t kernelWidth,
                                        const float scale,
                                        float* dest) {
    const uint2 threadPos = getPosition();

    // Quit if we exceed the bounds of the image
    if (threadPos.x >= numCols || threadPos.y >= numRows) {
        return;
    }

    const unsigned int threadLoc = convert2dTo1d(threadPos, numCols);
    const size_t kernelRadius = kernelWidth / 2;

    // Copy to shared memory
    extern __shared__ float shmem[];
    // Store the kernel in the first kernelWidth indices, and the actual row values in the rest of
    // the allocated memory
    float* shKernel = shmem;
    // TODO: This means that I start using negative array indexing later - not sure if this is
    // generally considered good practice @tanmaniac
    float* shRowData = shmem + kernelWidth + kernelRadius;

    // Only need the first kernelWidth number of threads to copy the kernel
    if (threadIdx.y < kernelWidth) {
        shKernel[threadIdx.y] = kernel[threadIdx.y];
    }
    // Copy pixels to shared mem
    shRowData[threadIdx.y] = input[threadLoc];

    // Get top edge pixels
    if (threadIdx.y == 0) {
        for (int i = -1; i >= -1 * int(kernelRadius); i--) {
            shRowData[i] = (threadPos.y != 0) ? input[threadLoc + (i * numCols)] : input[threadLoc];
        }
    }
    // Get bottom edge pixels
    if (threadIdx.y == blockDim.y - 1 || threadPos.y == numCols - 1) {
        for (int i = 1; i <= kernelRadius; i++) {
            shRowData[threadIdx.y + i] =
                (threadPos.y != numCols - 1) ? input[threadLoc + (i * numCols)] : input[threadLoc];
        }
    }
    __syncthreads();

    // Iterate over the kernel and sum up the values
    float sum = 0.f;
    for (int idx = 0; idx < kernelWidth; idx++) {
        sum += shKernel[idx] * shRowData[threadIdx.y + idx - kernelRadius];
    }

    dest[threadLoc] = sum;
}

//-----------------------------------------------------------------------------
// C++ runner functions
void rowConvolution(const cv::cuda::GpuMat& d_input,
                    const cv::cuda::GpuMat& d_kernel,
                    cv::cuda::GpuMat& d_dest) {
    assert(d_input.channels() == 1 && d_kernel.channels() == 1 && d_kernel.rows == 1 &&
           d_input.type() == CV_32FC1 && d_kernel.type() == CV_32FC1);

    static constexpr size_t THREADS_PER_BLOCK = 256;
    const size_t rows = d_input.rows;
    const size_t cols = d_input.cols;
    const size_t kernelSize = d_kernel.cols;
    assert(kernelSize % 2 == 1);

    d_dest = cv::cuda::createContinuous(rows, cols, d_input.type());

    // Run convolution kernel
    dim3 blocks(max(1, (unsigned int)ceil(float(cols) / float(THREADS_PER_BLOCK))), rows);
    dim3 threads(THREADS_PER_BLOCK);
    //std::cout << "blocks = " << blocks.x << ", " << blocks.y << std::endl;
    //std::cout << "threads = " << threads.x << ", " << threads.y << std::endl;

    const size_t shmSize = (THREADS_PER_BLOCK + 2 * kernelSize) * sizeof(float);

    rowConvolutionKernel<<<blocks, threads, shmSize>>>(d_input.ptr<float>(),
                                                       d_kernel.ptr<float>(),
                                                       rows,
                                                       cols,
                                                       kernelSize,
                                                       1.f,
                                                       d_dest.ptr<float>());
    cudaDeviceSynchronize();
    checkCudaErrors(cudaGetLastError());
}

void columnConvolution(const cv::cuda::GpuMat& d_input,
                       const cv::cuda::GpuMat& d_kernel,
                       cv::cuda::GpuMat& d_dest) {
    assert(d_input.channels() == 1 && d_kernel.channels() == 1 && d_kernel.rows == 1 &&
           d_input.type() == CV_32FC1 && d_kernel.type() == CV_32FC1);

    static constexpr size_t THREADS_PER_BLOCK = 256;
    const size_t rows = d_input.rows;
    const size_t cols = d_input.cols;
    const size_t kernelSize = d_kernel.cols;
    assert(kernelSize % 2 == 1);

    d_dest = cv::cuda::createContinuous(rows, cols, d_input.type());

    // Run convolution kernel
    dim3 blocks(cols, max(1, (unsigned int)ceil(float(rows) / float(THREADS_PER_BLOCK))));
    dim3 threads(1, THREADS_PER_BLOCK);
    //std::cout << "blocks = " << blocks.x << ", " << blocks.y << std::endl;
    //std::cout << "threads = " << threads.x << ", " << threads.y << std::endl;

    const size_t shmSize = (THREADS_PER_BLOCK + 2 * kernelSize) * sizeof(float);

    columnConvolutionKernel<<<blocks, threads, shmSize>>>(d_input.ptr<float>(),
                                                          d_kernel.ptr<float>(),
                                                          rows,
                                                          cols,
                                                          kernelSize,
                                                          1.f,
                                                          d_dest.ptr<float>());
    cudaDeviceSynchronize();
    checkCudaErrors(cudaGetLastError());
}

void rowConvolution(const cv::Mat& input, const cv::Mat& kernel, cv::Mat& dest) {
    assert(input.channels() == 1 && kernel.channels() == 1 && kernel.rows == 1 &&
           input.type() == CV_32FC1 && kernel.type() == CV_32FC1);

    // Copy to device
    cv::cuda::GpuMat d_input, d_kernel, d_dest;

    // Separate thread since upload() is a blocking call
    std::thread copyInputThread([&d_input, &input]() { d_input.upload(input); });
    std::thread copyKernelThread([&d_kernel, &kernel]() { d_kernel.upload(kernel); });

    copyKernelThread.join();
    copyInputThread.join();

    rowConvolution(d_input, d_kernel, d_dest);

    // Copy output back to destination matrix
    dest = cv::Mat(d_dest.rows, d_dest.cols, input.type());
    d_dest.download(dest);
}

void columnConvolution(const cv::Mat& input, const cv::Mat& kernel, cv::Mat& dest) {
    assert(input.channels() == 1 && kernel.channels() == 1 && kernel.rows == 1 &&
           input.type() == CV_32FC1 && kernel.type() == CV_32FC1);

    // Copy to device
    cv::cuda::GpuMat d_input, d_kernel, d_dest;

    // Separate thread since upload() is a blocking call
    std::thread copyInputThread([&d_input, &input]() { d_input.upload(input); });
    std::thread copyKernelThread([&d_kernel, &kernel]() { d_kernel.upload(kernel); });

    copyKernelThread.join();
    copyInputThread.join();

    columnConvolution(d_input, d_kernel, d_dest);

    // Copy output back to destination matrix
    dest = cv::Mat(d_dest.rows, d_dest.cols, input.type());
    d_dest.download(dest);
}

// Convolve will do both row and column convolution steps together, so there doesn't need to be a
// buffer between each step. Params:
//  input       OpenCV Matrix (CPU) of the input image. Must be grayscale, type CV_32FC1
//  rowKernel   First row of a separable filter kernel. Type CV_32FC1
//  colKernel   First column of the separable filter kernel. Type CV_32FC1
//  dest        Destination matrix to which the result should be written.
void separableConvolution(const cv::Mat& input,
                          const cv::Mat& rowKernel,
                          const cv::Mat& colKernel,
                          cv::Mat& dest) {
    assert(input.channels() == 1 && rowKernel.channels() == 1 && colKernel.channels() == 1);

    const size_t rows = input.rows;
    const size_t cols = input.cols;
    const size_t rowKernelSize = rowKernel.cols;
    const size_t colKernelSize = colKernel.rows;
    assert(rowKernelSize % 2 == 1 && colKernelSize % 2 == 1);

    // TODO: colKernel is supposed to be a column-vector, but accept row vectors too since the
    // column-vector is just transposed to a row vector anway
    cv::Mat colKernelRow;
    cv::transpose(colKernel, colKernelRow);

    // Copy to device
    cv::cuda::GpuMat d_input, d_rowKernel, d_colKernel;
    cv::cuda::GpuMat d_buffer = cv::cuda::createContinuous(rows, cols, input.type());
    cv::cuda::GpuMat d_dest = cv::cuda::createContinuous(rows, cols, input.type());

    // Separate thread since upload() is a blocking call
    std::thread copyInputThread([&d_input, &input]() { d_input.upload(input); });
    std::thread copyRowKernelThread(
        [&d_rowKernel, &rowKernel]() { d_rowKernel.upload(rowKernel); });
    std::thread copyColKernelThread(
        [&d_colKernel, &colKernelRow]() { d_colKernel.upload(colKernelRow); });

    copyRowKernelThread.join();
    copyColKernelThread.join();
    copyInputThread.join();

    // Run row convolution
    rowConvolution(d_input, d_rowKernel, d_buffer);

    // Run column convolution kernel
    columnConvolution(d_buffer, d_colKernel, d_dest);

    // Copy output back to destination matrix
    dest = cv::Mat(rows, cols, input.type());
    d_dest.download(dest);
}
