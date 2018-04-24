#include <common/GpuTimer.h>
#include <common/Utils.h>
#include <common/CudaCommon.cuh>
#include "../include/Config.h"
#include "../include/Harris.h"

#include <spdlog/spdlog.h>

#include <opencv2/core/core.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/core/cuda/common.hpp>
#include <opencv2/core/cuda_stream_accessor.hpp>
#include <opencv2/core/opengl.hpp>
#include <opencv2/imgproc/imgproc.hpp>

// Texture memory for fast access to spatially colocated memory
texture<float, cudaTextureType2D, cudaReadModeElementType> texGradX;    // X-direction gradient
texture<float, cudaTextureType2D, cudaReadModeElementType> texGradY;    // Y-direction gradient
texture<float, cudaTextureType2D, cudaReadModeElementType> texGauss;    // Gaussian kernel

// Implementations

// Implements a fused-multiply-add for a 4-vector of single-precision floating points.
__device__ float4 __fmaf4(float weight, float4 A, float4 B) {
    float4 result;
    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(result.x) : "f"(weight), "f"(A.x), "f"(B.x));
    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(result.y) : "f"(weight), "f"(A.y), "f"(B.y));
    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(result.z) : "f"(weight), "f"(A.z), "f"(B.z));
    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(result.w) : "f"(weight), "f"(A.w), "f"(B.w));
    return result;
}

// Naive corner response kernel (unoptimized)
template <typename T>
__global__ void cornerResponseKernel(const cv::cuda::PtrStepSz<T> gradX,
                                     const cv::cuda::PtrStepSz<T> gradY,
                                     const size_t windowSize,
                                     const float harrisScore,
                                     cv::cuda::PtrStepSz<T> cornerResponse) {
    const int2 gThreadPos =
        make_int2(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);

    if (gThreadPos.x >= gradX.cols || gThreadPos.y >= gradX.rows) {
        return;
    }

    float Ix, Iy; // Values of gradients at a given x and y
    // moment and intensity vectors are treated as 2x2 matrices structured as
    // [ x y
    //   z w ]
    float4 intensity;                                // Intensity matrix
    float4 moment = make_float4(0.f, 0.f, 0.f, 0.f); // Second moment matrix

    // Iterate over window
    int windowRad = windowSize / 2;
    for (int wy = -windowRad; wy <= windowRad; wy++) {
        for (int wx = -windowRad; wx <= windowRad; wx++) {
            Ix = tex2D(texGradX, gThreadPos.x + wx, gThreadPos.y + wy);
            Iy = tex2D(texGradY, gThreadPos.x + wx, gThreadPos.y + wy);

            float weight = tex2D(texGauss, wx + windowRad, wy + windowRad);
            // Build up gradient matrix
            intensity = make_float4(Ix * Ix, Ix * Iy, Ix * Iy, Iy * Iy);

            moment = __fmaf4(weight, intensity, moment);
        }
    }

    float trace = moment.x + moment.w;
    float determinant = moment.x * moment.w - moment.z * moment.y;
    float response = determinant - harrisScore * trace * trace;

    cornerResponse(gThreadPos.y, gThreadPos.x) = response;
}

void harris::gpu::getCornerResponse(const cv::Mat& gradX,
                                    const cv::Mat& gradY,
                                    const size_t windowSize,
                                    const double gaussianSigma,
                                    const float harrisScore,
                                    cv::Mat& cornerResponse) {
    assert(gradX.rows == gradY.rows && gradX.cols == gradY.cols && gradX.type() == CV_32F &&
           gradX.type() == gradY.type());
    assert(windowSize % 2 == 1);

    auto logger = spdlog::get(config::STDOUT_LOGGER);
    auto flogger = spdlog::get(config::FILE_LOGGER);

    cornerResponse = cv::Mat::zeros(gradX.rows, gradX.cols, CV_32F);
    flogger->info("Resized cornerResponse to match gradient sizes: {} rows x {} cols",
                  cornerResponse.rows,
                  cornerResponse.cols);
    // Get a 1D Gaussian kernel with a given size and sigma
    cv::Mat gauss = cv::getGaussianKernel(windowSize, gaussianSigma, gradX.type());
    // Outer product for a 2D matrix
    gauss = gauss * gauss.t();

    // Allocate GPU memory and async streams
    cv::cuda::GpuMat d_gradX(gradX.rows, gradX.cols, gradX.type()),
        d_gradY(gradY.rows, gradY.cols, gradY.type()),
        d_gauss(gauss.rows, gauss.cols, gauss.type()),
        d_cornerResponse(cornerResponse.rows, cornerResponse.cols, cornerResponse.type());
    cv::cuda::Stream stream;
    stream.enqueueHostCallback(common::checkCopySuccess, nullptr);

    // Asynchronously copy memory
    d_gradX.upload(gradX, stream);
    d_gradY.upload(gradY, stream);
    d_gauss.upload(gauss, stream);
    d_cornerResponse.setTo(cv::Scalar(0), stream);

    // Bind GPU matricies to texture memory
    cv::cuda::device::bindTexture<float>(&texGradX, d_gradX);
    cv::cuda::device::bindTexture<float>(&texGradY, d_gradY);
    cv::cuda::device::bindTexture<float>(&texGauss, d_gauss);

    // Set up kernel call
    static constexpr size_t TILE_SIZE = 16;

    dim3 blocks(common::divRoundUp(gradX.cols, TILE_SIZE),
                common::divRoundUp(gradX.rows, TILE_SIZE));
    dim3 threads(TILE_SIZE, TILE_SIZE);

    // No shared memory

    logger->info("Launching cornerResponseKernel");
    GpuTimer timer;
    timer.start();

    cornerResponseKernel<float>
        <<<blocks, threads, 0, cv::cuda::StreamAccessor::getStream(stream)>>>(
            d_gradX, d_gradY, windowSize, harrisScore, d_cornerResponse);

    cudaDeviceSynchronize();
    checkCudaErrors(cudaGetLastError());

    timer.stop();
    logger->info("cornerResponseKernel execution took {} ms", timer.getTime());

    // Copy back to CPU
    d_cornerResponse.download(cornerResponse, stream);
}