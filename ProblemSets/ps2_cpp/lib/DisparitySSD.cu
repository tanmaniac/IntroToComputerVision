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

// Implementations

// Stereo block matcher kernel using sum of squared differences. Returns disparity values over the
// image.
// TODO: Use texture memory + shared memory instead of global mem

__global__ void disparitySSDKernel(const cv::cuda::PtrStepSz<float> imgLeft,
                                   const cv::cuda::PtrStepSz<float> imgRight,
                                   const ReferenceFrame frame,
                                   const size_t windowRad,
                                   const size_t minDisparity,
                                   const size_t maxDisparity,
                                   cv::cuda::PtrStepSz<unsigned char> disparity) {
    const uint2 threadPos = make_uint2(blockIdx.x * blockDim.x + threadIdx.x + windowRad,
                                       blockIdx.y * blockDim.y + threadIdx.y + windowRad);

    if (threadPos.x > imgLeft.cols - windowRad || threadPos.y > imgLeft.rows - windowRad) {
        return;
    }

    int bestCost = 99999999;
    int bestDisparity = 0;
    // Iterate over the row to search for a matching window. If the reference frame is the
    // left image, then we search to the left; if it's the right image, then we search to
    // the right
    int searchIndex =
        (frame == ReferenceFrame::LEFT) ? max(0, int(threadPos.x) - int(maxDisparity)) : threadPos.x;
    int maxSearchIndex = (frame == ReferenceFrame::LEFT)
                             ? threadPos.x - minDisparity
                             : min(imgLeft.cols - 2 * windowRad, threadPos.x + maxDisparity);
    for (; searchIndex <= maxSearchIndex; searchIndex++) {
        int sum = 0;
        // Iterate over the window and compute sum of squared differences
        for (int winY = -windowRad; winY <= int(windowRad); winY++) {
            for (int winX = -windowRad; winX <= int(windowRad); winX++) {
                float rawCost = imgLeft(threadPos.y + winY, threadPos.x + winX) -
                                imgRight(threadPos.y + winY, searchIndex + winX);
                sum += roundf(rawCost * rawCost);
            }
        }
        if (sum < bestCost) {
            bestCost = sum;
            bestDisparity = searchIndex - threadPos.x;
        }
    }

    disparity(threadPos.y - windowRad, threadPos.x - windowRad) = bestDisparity;
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
    static constexpr size_t TILE_SIZE = 16;

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
    dim3 blocks(max(1, (unsigned int)ceil(float(left.cols) / float(TILE_SIZE))),
                max(1, (unsigned int)ceil(float(left.rows) / float(TILE_SIZE))));
    dim3 threads(TILE_SIZE, TILE_SIZE);
    
    // Time kernel execution
    logger->info("Launching disparitySSDKernel");
    GpuTimer timer;
    timer.start();

    disparitySSDKernel<<<blocks, threads, 0, cv::cuda::StreamAccessor::getStream(cpyStream)>>>(
        d_leftPadded, d_rightPadded, frame, windowRad, minDisparity, maxDisparity, d_disparity);

    timer.stop();
    logger->info("disparitySSDKernel execution took {} ms", timer.getTime());

    // Copy back to CPU
    d_disparity.download(disparity);
}