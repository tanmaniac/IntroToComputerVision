#include <common/GpuTimer.h>
#include <common/Utils.h>
#include <common/CudaCommon.cuh>
#include "../include/DisparitySSD.h"

#include <spdlog/spdlog.h>

#include <opencv2/core/core.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/core/cuda/common.hpp>
#include <opencv2/core/cuda_stream_accessor.hpp>
#include <opencv2/core/opengl.hpp>

// Minimum SSD value required to be counted as a disparity point
static constexpr float MIN_DISP_SSD = 5000000;
static constexpr size_t ROWS_PER_THREAD = 40; // Number of rows each thread will process

texture<float, cudaTextureType2D, cudaReadModeElementType> textureLeft;
texture<float, cudaTextureType2D, cudaReadModeElementType> textureRight;

// Implementations

// Stereo block matcher kernel using sum of squared differences. Returns disparity values over the
// image.
// Reference: https://goo.gl/SdEkVj
template <typename srcType, typename dispType>
__global__ void disparitySSDKernel(const cv::cuda::PtrStepSz<srcType> imgLeft,
                                   const cv::cuda::PtrStepSz<srcType> imgRight,
                                   const int windowRad,
                                   const int minDisparity,
                                   const int maxDisparity,
                                   cv::cuda::PtrStepSz<dispType> disparity,
                                   cv::cuda::PtrStepSz<float> disparityMinSSD) {
    const int2 gThreadPos =
        make_int2(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * ROWS_PER_THREAD);

    // Shared memory, which contains the SSD values for the current row
    extern __shared__ float ssdDiffs[];

    dispType disp;   // Disparity value
    float diff, ssd; // Temp difference value; total SSD for kernel
    int row, i;      // Current row in rolling window; index
    int2 texCoords;  // Texture coordinates for image lookup

    // For threads reading the extra border pixels, this is the offset into shared memory to store
    // the values
    int extraReadVal = 0;
    if (threadIdx.x < 2 * windowRad) {
        extraReadVal = blockDim.x + threadIdx.x;
    }

    // Iterate over the range of disparity values to be set
    if (gThreadPos.x < imgLeft.cols + windowRad && gThreadPos.y <= imgLeft.rows) {
        texCoords.x = gThreadPos.x - windowRad;
        for (disp = minDisparity; disp <= maxDisparity; disp++) {
            ssdDiffs[threadIdx.x] = 0;

            if (extraReadVal > 0) {
                ssdDiffs[extraReadVal] = 0;
            }
            __syncthreads();

            // Accumulate column sums for the first 2 * windowRad + 1 rows (vertically)
            texCoords.y = gThreadPos.y - windowRad;

            for (i = 0; i <= 2 * windowRad; i++) {
                diff = tex2D(textureLeft, texCoords.x, texCoords.y) -
                       tex2D(textureRight, texCoords.x + disp, texCoords.y);
                ssdDiffs[threadIdx.x] += diff * diff;

                if (extraReadVal > 0) {
                    diff = tex2D(textureLeft, texCoords.x + blockDim.x, texCoords.y) -
                           tex2D(textureRight, texCoords.x + blockDim.x + disp, texCoords.y);
                    ssdDiffs[extraReadVal] += diff * diff;
                }
                texCoords.y++;
            }
            __syncthreads();

            // Accumulate the total in the horizontal direction
            if (gThreadPos.x < imgLeft.cols && gThreadPos.y < imgLeft.rows) {
                ssd = 0;
                for (i = 0; i < 2 * windowRad; i++) {
                    ssd += ssdDiffs[i + threadIdx.x];
                }

                if (ssd < disparityMinSSD(gThreadPos.y, gThreadPos.x)) {
                    disparity(gThreadPos.y, gThreadPos.x) = disp;
                    disparityMinSSD(gThreadPos.y, gThreadPos.x) = ssd;
                }
            }
            __syncthreads();

            // Rolling window to compute the rest of the rows for this thread
            texCoords.y = gThreadPos.y - windowRad;
            for (row = 1; row < ROWS_PER_THREAD && row + gThreadPos.y < imgLeft.rows + windowRad;
                 row++) {
                // Subtract the value of the first row from the column sums
                diff = tex2D(textureLeft, texCoords.x, texCoords.y) -
                       tex2D(textureRight, texCoords.x + disp, texCoords.y);
                ssdDiffs[threadIdx.x] -= diff * diff;

                // Add in the value from the next row down
                diff = tex2D(textureLeft, texCoords.x, texCoords.y + 2 * windowRad + 1) -
                       tex2D(textureRight, texCoords.x + disp, texCoords.y + 2 * windowRad + 1);
                ssdDiffs[threadIdx.x] += diff * diff;

                // Handle edges
                if (extraReadVal > 0) {
                    diff = tex2D(textureLeft, texCoords.x + blockDim.x, texCoords.y) -
                           tex2D(textureRight, texCoords.x + disp + blockDim.x, texCoords.y);
                    ssdDiffs[threadIdx.x + blockDim.x] -= diff * diff;

                    diff = tex2D(textureLeft,
                                 texCoords.x + blockDim.x,
                                 texCoords.y + 2 * windowRad + 1) -
                           tex2D(textureRight,
                                 texCoords.x + disp + blockDim.x,
                                 texCoords.y + 2 * windowRad + 1);
                    ssdDiffs[extraReadVal] += diff * diff;
                }
                texCoords.y++;
                __syncthreads();

                // Accumulate the total
                if (gThreadPos.x < imgLeft.cols && gThreadPos.y + row < imgLeft.rows) {
                    ssd = 0;
                    for (i = 0; i < 2 * windowRad; i++) {
                        ssd += ssdDiffs[i + threadIdx.x];
                    }
                    if (ssd < disparityMinSSD(gThreadPos.y + row, gThreadPos.x)) {
                        disparity(gThreadPos.y + row, gThreadPos.x) = disp;
                        disparityMinSSD(gThreadPos.y + row, gThreadPos.x) = ssd;
                    }
                }
                __syncthreads();
            }
        }
    }
}

void cuda::disparitySSD(const cv::Mat& left,
                        const cv::Mat& right,
                        const size_t windowRad,
                        const int minDisparity,
                        const int maxDisparity,
                        cv::Mat& disparity) {
    // Don't really want to deal with templating this right now
    assert(left.type() == CV_32FC1 && right.type() == CV_32FC1);

    static constexpr size_t TILE_SIZE_X = 64;
    static constexpr size_t TILE_SIZE_Y = 1;

    // Set up file loggers
    auto logger = spdlog::get("file_logger");
    logger->info("Setting up CUDA kernel execution...");
    logger->info("Original image: rows={} cols={}", left.rows, left.cols);

    disparity.create(left.rows, left.cols, CV_8SC1);

    // Copy to GPU memory. cv::GpuMat::upload() calls are blocking, so copy in CUDA streams
    cv::cuda::GpuMat d_left, d_right, d_disparity, d_disparityMinSSD;
    cv::cuda::Stream cpyStream;
    // Set up a callback to log a confirmation that the stream copy has completed
    cpyStream.enqueueHostCallback(common::checkCopySuccess, nullptr);

    // Allocate GPU memory
    // Note that I don't use createContinuous since the texture binding operations required pitched
    // memory
    d_left.create(left.rows, left.cols, left.type());
    d_right.create(right.rows, right.cols, right.type());
    d_disparity.create(disparity.rows, disparity.cols, disparity.type());
    d_disparityMinSSD.create(disparity.rows, disparity.cols, CV_32FC1);
    d_left.upload(left, cpyStream);
    d_right.upload(right, cpyStream);
    d_disparity.setTo(cv::Scalar(-1), cpyStream);
    d_disparityMinSSD.setTo(cv::Scalar(MIN_DISP_SSD), cpyStream);

    // Set up texture memory
    cv::cuda::device::bindTexture<float>(&textureLeft, d_left);
    cv::cuda::device::bindTexture<float>(&textureRight, d_right);

    // Determine block and grid size
    dim3 blocks(common::divRoundUp(left.cols, TILE_SIZE_X),
                common::divRoundUp(left.rows, ROWS_PER_THREAD));
    dim3 threads(TILE_SIZE_X, TILE_SIZE_Y);
    size_t shmSize = (TILE_SIZE_X + 2 * windowRad) * sizeof(float);

    // Time kernel execution
    logger->info("Launching disparitySSDKernel");
    GpuTimer timer;
    timer.start();

    disparitySSDKernel<float, char>
        <<<blocks, threads, shmSize, cv::cuda::StreamAccessor::getStream(cpyStream)>>>(
            d_left, d_right, windowRad, minDisparity, maxDisparity, d_disparity, d_disparityMinSSD);

    cudaDeviceSynchronize();
    checkCudaErrors(cudaGetLastError());

    timer.stop();
    logger->info("disparitySSDKernel execution took {} ms", timer.getTime());

    // Copy back to CPU
    d_disparity.download(disparity);
}