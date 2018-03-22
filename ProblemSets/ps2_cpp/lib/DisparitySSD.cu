#include <common/GpuTimer.h>
#include <common/CudaCommon.cuh>
#include "../include/DisparitySSD.h"

#include <spdlog/spdlog.h>

#include <opencv2/core/core.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/core/cuda_stream_accessor.hpp>

#define STEPS 3
#define BLOCKDIM_X 8
#define BLOCKDIM_Y 32
#define WINDOW_RAD 8

template <typename T>
struct Array2DWrapper {
    // Data array on device
    T* _data;
    size_t _rows, _cols;

    __device__ Array2DWrapper(T* data, const size_t rows, const size_t cols)
        : _data(data), _rows(rows), _cols(cols) {}

    // Access array at index with row value y and column value x
    __device__ T& operator()(const size_t y, const size_t x) {
        return _data[y * _cols + x];
    }
};

// Implementations

// Stereo block matcher kernel using sum of squared differences. Returns disparity values over the
// image.
// TODO: Use texture memory + shared memory instead of global mem
template <typename srcType, typename dispType>
__global__ void disparitySSDKernel(const cv::cuda::PtrStepSz<srcType> imgLeft,
                                   const cv::cuda::PtrStepSz<srcType> imgRight,
                                   const ReferenceFrame frame,
                                   const size_t windowRad,
                                   const size_t minDisparity,
                                   const size_t maxDisparity,
                                   cv::cuda::PtrStepSz<dispType> disparity) {
    const uint2 gThreadPos = make_uint2(blockIdx.x * blockDim.x + threadIdx.x + windowRad,
                                        blockIdx.y * blockDim.y + threadIdx.y + windowRad);

    if (gThreadPos.x > imgLeft.cols - windowRad || gThreadPos.y > imgLeft.rows - windowRad) {
        return;
    }

    // Copy to shared memory
    extern __shared__ srcType shmImgs[];
    size_t stepSize = blockDim.x + 2 * windowRad; // x-dimension of shared memory region
    size_t steps = 1 + 2 * windowRad;             // y-dimension of shared memory region

    // Determine location within the shared memory arrays
    const uint2 sThreadPos = make_uint2(threadIdx.x + windowRad, threadIdx.y + windowRad);

    // Create wrappers around the shared memory for left and right images for easy access
    Array2DWrapper<srcType> sharedLeft(shmImgs, steps, stepSize);
    Array2DWrapper<srcType> sharedRight(shmImgs + stepSize * steps, steps, stepSize);

    // Each thread copies itself and the windowRad pixels above/below it to sh mem
    for (int y = -windowRad; y <= int(windowRad); y++) {
        sharedLeft(sThreadPos.y + y, sThreadPos.x) = imgLeft(gThreadPos.y + y, gThreadPos.x);
        sharedRight(sThreadPos.y + y, sThreadPos.x) = imgRight(gThreadPos.y + y, gThreadPos.x);
    }

    // Leftmost pixel stores the left side pixels to shared memory
    if (threadIdx.x == 0) {
        for (int y = -windowRad; y <= int(windowRad); y++) {
            for (int x = -windowRad; x < 0; x++) {
                sharedLeft(sThreadPos.y + y, sThreadPos.x + x) =
                    imgLeft(gThreadPos.y + y, gThreadPos.x + x);
                sharedRight(sThreadPos.y + y, sThreadPos.x + x) =
                    imgRight(gThreadPos.y + y, gThreadPos.x + x);
            }
        }
    }

    // Rightmost pixel stores right side pixels to shared memory
    if (threadIdx.x == blockDim.x - 1) {
        for (int y = -windowRad; y <= int(windowRad); y++) {
            for (int x = 1; x <= windowRad; x++) {
                sharedLeft(sThreadPos.y + y, sThreadPos.x + x) =
                    imgLeft(gThreadPos.y + y, gThreadPos.x + x);
                sharedRight(sThreadPos.y + y, sThreadPos.x + x) =
                    imgRight(gThreadPos.y + y, gThreadPos.x + x);
            }
        }
    }

    __syncthreads();

    // Print shared memory
    if (threadIdx.x == 0 && blockIdx.x == 0 && blockIdx.y == 1) {
        for (int y = 0; y < sharedLeft._rows; y++) {
            for (int x = 0; x < sharedLeft._cols; x++) {
                // printf("sharedLeft @ %d, %d (y, x) = %f\n", y, x, sharedLeft(y, x));
                if (sharedLeft(y, x) != imgLeft(y, x)) {
                    printf("Difference found @ (%d, %d) (y, x): shared = %f, orig = %f\n",
                           y,
                           x,
                           sharedLeft(y, x),
                           imgLeft(y, x));
                }
            }
        }
    }

    int bestCost = 99999999;
    int bestDisparity = 0;
    // Iterate over the row to search for a matching window. If the reference frame is the
    // left image, then we search to the left; if it's the right image, then we search to
    // the right
    // Really hacky "zero" index
    int searchIndex = (frame == ReferenceFrame::LEFT)
                          ? max(int(windowRad), int(gThreadPos.x) - int(maxDisparity))
                          : gThreadPos.x;
    int maxSearchIndex = (frame == ReferenceFrame::LEFT)
                             ? gThreadPos.x - minDisparity
                             : min(size_t(imgLeft.cols), gThreadPos.x + maxDisparity);
    for (; searchIndex <= maxSearchIndex; searchIndex++) {
        int sum = 0;
        // Iterate over the window and compute sum of squared differences
        for (int winY = -windowRad; winY <= int(windowRad); winY++) {
            for (int winX = -windowRad; winX <= int(windowRad); winX++) {
                float rawCost = imgLeft(gThreadPos.y + winY, gThreadPos.x + winX) -
                                imgRight(gThreadPos.y + winY, searchIndex + winX);
                sum += roundf(rawCost * rawCost);
            }
        }
        if (sum < bestCost) {
            bestCost = sum;
            bestDisparity = searchIndex - gThreadPos.x;
        }
    }

    disparity(gThreadPos.y - windowRad, gThreadPos.x - windowRad) = bestDisparity;
}

// CUDA stream callback
void checkCopySuccess(int status, void* userData) {
    auto logger = spdlog::get("file_logger");
    if (status == cudaSuccess) {
        logger->info("Stream successfully copied data to GPU");
    } else {
        logger->error("Stream copy failed!");
    }
}

void cuda::disparitySSD(const cv::Mat& left,
                        const cv::Mat& right,
                        const ReferenceFrame frame,
                        const size_t windowRad,
                        const size_t minDisparity,
                        const size_t maxDisparity,
                        cv::Mat& disparity) {
    static constexpr size_t TILE_SIZE_X = 64;
    static constexpr size_t TILE_SIZE_Y = 1;

    // Set up file loggers
    auto logger = spdlog::get("file_logger");
    logger->info("Setting up CUDA kernel execution...");
    logger->info("Padding input images with {} pixels", windowRad);
    cv::Mat leftPadded, rightPadded;
    // Hacky way to use right image as reference frame vs the left - just make the left padded image
    // out of the right
    cv::copyMakeBorder((frame == ReferenceFrame::LEFT ? left : right),
                       leftPadded,
                       windowRad,
                       windowRad,
                       windowRad,
                       windowRad,
                       cv::BORDER_REPLICATE);
    cv::copyMakeBorder((frame == ReferenceFrame::LEFT ? right : left),
                       rightPadded,
                       windowRad,
                       windowRad,
                       windowRad,
                       windowRad,
                       cv::BORDER_REPLICATE);
    logger->info("Original image: rows={} cols={}; new image: rows={} cols={}",
                 left.rows,
                 left.cols,
                 leftPadded.rows,
                 leftPadded.cols);

    disparity.create(cv::Size(left.rows, left.cols), CV_8SC1);

    // Copy to GPU memory. cv::GpuMat::upload() calls are blocking, so copy in CUDA streams
    cv::cuda::GpuMat d_left, d_right, d_leftPadded, d_rightPadded, d_disparity;
    cv::cuda::Stream cpyStream;
    cpyStream.enqueueHostCallback(checkCopySuccess, NULL);

    d_leftPadded.upload(leftPadded, cpyStream);
    d_rightPadded.upload(rightPadded, cpyStream);
    d_disparity.upload(disparity, cpyStream);

    // Determine block and grid size
    dim3 blocks(max(1, (unsigned int)ceil(float(left.cols) / float(TILE_SIZE_X))),
                max(1, (unsigned int)ceil(float(left.rows) / float(TILE_SIZE_Y))));
    dim3 threads(TILE_SIZE_X, TILE_SIZE_Y);
    size_t shmSize = 2 * ((TILE_SIZE_X + 2 * windowRad) * (1 + 2 * windowRad)) * sizeof(float);

    // Time kernel execution
    logger->info("Launching disparitySSDKernel");
    GpuTimer timer;
    timer.start();

    disparitySSDKernel<float, unsigned char>
        <<<blocks, threads, shmSize, cv::cuda::StreamAccessor::getStream(cpyStream)>>>(
            d_leftPadded, d_rightPadded, frame, windowRad, minDisparity, maxDisparity, d_disparity);

    cudaDeviceSynchronize();
    checkCudaErrors(cudaGetLastError());

    timer.stop();
    logger->info("disparitySSDKernel execution took {} ms", timer.getTime());

    // Copy back to CPU
    d_disparity.download(disparity);
}