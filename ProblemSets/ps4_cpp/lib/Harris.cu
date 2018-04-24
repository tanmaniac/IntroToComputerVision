#include <common/GpuTimer.h>
#include <common/OpenCVThrustInterop.h>
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

// Num cols each thread processes for non-maximum suppression
static constexpr size_t NMS_COLS_PER_THREAD = 32;

// Texture memory for fast access to spatially colocated memory
texture<float, cudaTextureType2D, cudaReadModeElementType> texGradX; // X-direction gradient
texture<float, cudaTextureType2D, cudaReadModeElementType> texGradY; // Y-direction gradient
texture<float, cudaTextureType2D, cudaReadModeElementType> texGauss; // Gaussian kernel
texture<float, cudaTextureType2D, cudaReadModeElementType> texCornerResponse; // Corner responses

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
__global__ void cornerResponseKernel(const size_t rows,
                                     const size_t cols,
                                     const size_t windowSize,
                                     const float alpha,
                                     cv::cuda::PtrStepSz<T> cornerResponse) {
    const int2 gThreadPos =
        make_int2(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);

    if (gThreadPos.x >= cols || gThreadPos.y >= rows) {
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
    float response = determinant - alpha * trace * trace;

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

    flogger->info("Launching cornerResponseKernel");
    GpuTimer timer;
    timer.start();

    cornerResponseKernel<float>
        <<<blocks, threads, 0, cv::cuda::StreamAccessor::getStream(stream)>>>(
            d_gradX.rows, d_gradX.cols, windowSize, harrisScore, d_cornerResponse);

    cudaDeviceSynchronize();
    checkCudaErrors(cudaGetLastError());

    timer.stop();
    logger->info("cornerResponseKernel execution took {} ms", timer.getTime());

    // Copy back to CPU
    d_cornerResponse.download(cornerResponse, stream);
}

//--------- Corner refining functions -------------

template <typename T>
__global__ void refineCornersKernel(const size_t rows,
                                    const size_t cols,
                                    const double threshold,
                                    const int minDistance,
                                    cv::cuda::PtrStepSz<T> corners) {
    // Each thread processes NMS_COLS_PER_THREAD pixels horizontally
    const int2 gThreadPos =
        make_int2(blockIdx.x * NMS_COLS_PER_THREAD, blockIdx.y * blockDim.y + threadIdx.y);

    if (gThreadPos.x >= cols || gThreadPos.y >= rows) {
        return;
    }

    float curVal;    // Current value in corner response matrix being processed
    int2 texPos;     // Position in texture map
    bool isLocalMax; // Whether or not the current pixel is a local maxima

#pragma unroll
    for (int idx = 0; idx < NMS_COLS_PER_THREAD; idx++) {
        texPos = make_int2(gThreadPos.x + idx, gThreadPos.y);
        curVal = tex2D(texCornerResponse, texPos.x, texPos.y);
        if (curVal >= threshold) {
            isLocalMax = true;
            // Iterate over window, which is (minDistance * 2 + 1)^2 pixels
            for (int wy = -minDistance; wy <= minDistance; wy++) {
                for (int wx = -minDistance; wx <= minDistance; wx++) {
                    // Skip if comparison point is the current point
                    int compY = min(max(0, texPos.y + wy), int(rows - 1));
                    int compX = min(max(0, texPos.x + wx), int(cols - 1));
                    if (texPos.y == compY && texPos.x == compX) continue;

                    if (curVal <= tex2D(texCornerResponse, compX, compY)) {
                        isLocalMax = false;
                        break;
                    }
                }
                if (!isLocalMax) break;
            }
            if (isLocalMax) {
                corners(texPos.y, texPos.x) = curVal;
                // Skip ahead in the row, since we know this is a local maxima
                idx += (minDistance - 1);
            }
        }
    }
}

// Model of predicate that returns true if the compared value is greater than or equal to the
// predicate's value.
template <typename T>
struct GreaterOrEqual {
    T _value;
    __host__ __device__ GreaterOrEqual(T value) : _value(value) {}
    __host__ __device__ bool operator()(const T& comp) const {
        return comp >= _value;
    }
};

void harris::gpu::refineCorners(const cv::Mat& cornerResponse,
                                const double threshold,
                                const int minDistance,
                                cv::Mat& corners) {
    // Only support CV_32F right now
    assert(cornerResponse.type() == CV_32F);

    auto logger = spdlog::get(config::STDOUT_LOGGER);
    auto flogger = spdlog::get(config::FILE_LOGGER);

    corners = cv::Mat::zeros(cornerResponse.rows, cornerResponse.cols, cornerResponse.type());
    flogger->info(
        "Resized corners to match input size: {} rows x {} cols", corners.rows, corners.cols);

    // Allocate GPU memory and async stream
    cv::cuda::GpuMat d_cornerResponse(
        cornerResponse.rows, cornerResponse.cols, cornerResponse.type()),
        d_corners(corners.rows, corners.cols, corners.type());
    cv::cuda::Stream stream;
    stream.enqueueHostCallback(common::checkCopySuccess, nullptr);

    // Asynchronously copy memory
    d_cornerResponse.upload(cornerResponse, stream);
    d_corners.setTo(cv::Scalar(0), stream);

    // Bind GPU matrices to texture memory
    cv::cuda::device::bindTexture<float>(&texCornerResponse, d_cornerResponse);

    // Set up kernel call
    static constexpr size_t TILE_SIZE_X = 1;
    static constexpr size_t TILE_SIZE_Y = 32;

    dim3 blocks(common::divRoundUp(cornerResponse.cols, NMS_COLS_PER_THREAD),
                common::divRoundUp(cornerResponse.rows, TILE_SIZE_Y));
    dim3 threads(TILE_SIZE_X, TILE_SIZE_Y);

    flogger->info("Launching refineCornersKernel");
    GpuTimer timer;
    timer.start();

    refineCornersKernel<float><<<blocks, threads, 0, cv::cuda::StreamAccessor::getStream(stream)>>>(
        d_cornerResponse.rows, d_cornerResponse.cols, threshold, minDistance, d_corners);

    cudaDeviceSynchronize();
    checkCudaErrors(cudaGetLastError());

    timer.stop();
    logger->info("refineCornersKernel execution took {} ms", timer.getTime());

    // Copy back to CPU
    d_corners.download(corners, stream);
}