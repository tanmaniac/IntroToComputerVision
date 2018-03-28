#include <common/GpuTimer.h>
#include <common/Utils.h>
#include <common/CudaCommon.cuh>
#include "../include/DisparityNCorr.h"

#include <spdlog/spdlog.h>

#include <opencv2/core/core.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/core/cuda/common.hpp>
#include <opencv2/core/cuda_stream_accessor.hpp>
#include <opencv2/core/opengl.hpp>

// Minimum SSD value required to be counted as a disparity point
static constexpr float MIN_DISP_SSD = 0;
static constexpr size_t ROWS_PER_THREAD = 40; // Number of rows each thread will process

texture<float, cudaTextureType2D, cudaReadModeElementType> textureLeft;
texture<float, cudaTextureType2D, cudaReadModeElementType> textureRight;

// Implementations

// Stereo block matcher kernel using normalized cross correlation. Returns disparity values over the
// two input images. Very similar in design to the sum-of-squared differences kernel
// (DisparitySSD.cu), but also keeps track of autocorrelation values for the template (the window
// surrounding the target pixel) and of the target image.
template <typename srcType, typename dispType>
__global__ void disparityNCorrKernel(const cv::cuda::PtrStepSz<srcType> imgLeft,
                                     const cv::cuda::PtrStepSz<srcType> imgRight,
                                     const int windowRad,
                                     const int minDisparity,
                                     const int maxDisparity,
                                     const size_t shmPitch,
                                     cv::cuda::PtrStepSz<dispType> disparity,
                                     cv::cuda::PtrStepSz<float> disparityMaxNCorr) {
    const int2 gThreadPos =
        make_int2(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * ROWS_PER_THREAD);

    // Shared memory, which contains the SSD values for the current row
    extern __shared__ float products[];
    float* acorrTemplProducts = products + shmPitch;
    float* acorrImgProducts = products + 2 * shmPitch;

    dispType disp;                    // Disparity value
    float nCorr;                      // Total normalized cross correlation for kernel
    float templVal, imgVal;           // Current values in the template and image, respectively
    int row, i;                       // Current row in rolling window; index
    int2 texCoords;                   // Texture coordinates for image lookup
    float autocorrTempl, autocorrImg; // Autocorrelations of template and image, respectively

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
            // Reset shared memory arrays
            products[threadIdx.x] = 0;
            acorrTemplProducts[threadIdx.x] = 0;
            acorrImgProducts[threadIdx.x] = 0;

            if (extraReadVal > 0) {
                products[extraReadVal] = 0;
                acorrTemplProducts[extraReadVal] = 0;
                acorrImgProducts[extraReadVal] = 0;
            }
            __syncthreads();

            // Accumulate column sums for the first 2 * windowRad + 1 rows (vertically)
            texCoords.y = gThreadPos.y - windowRad;

            for (i = 0; i <= 2 * windowRad; i++) {
                templVal = tex2D(textureLeft, texCoords.x, texCoords.y);
                imgVal = tex2D(textureRight, texCoords.x + disp, texCoords.y);
                products[threadIdx.x] += templVal * imgVal;
                acorrTemplProducts[threadIdx.x] += templVal * templVal;
                acorrImgProducts[threadIdx.x] += imgVal * imgVal;

                if (extraReadVal > 0) {
                    templVal = tex2D(textureLeft, texCoords.x + blockDim.x, texCoords.y);
                    imgVal = tex2D(textureRight, texCoords.x + blockDim.x + disp, texCoords.y);
                    products[extraReadVal] += templVal * imgVal;
                    acorrTemplProducts[extraReadVal] += templVal * templVal;
                    acorrImgProducts[extraReadVal] += imgVal * imgVal;
                }
                texCoords.y++;
            }
            __syncthreads();

            // Accumulate the total in the horizontal direction
            if (gThreadPos.x < imgLeft.cols && gThreadPos.y < imgLeft.rows) {
                nCorr = autocorrTempl = autocorrImg = 0;
                for (i = 0; i < 2 * windowRad; i++) {
                    nCorr += products[i + threadIdx.x];
                    autocorrTempl += acorrTemplProducts[i + threadIdx.x];
                    autocorrImg += acorrImgProducts[i + threadIdx.x];
                }

                // Normalize
                nCorr = nCorr / sqrtf(autocorrTempl * autocorrImg);

                if (nCorr > disparityMaxNCorr(gThreadPos.y, gThreadPos.x)) {
                    disparity(gThreadPos.y, gThreadPos.x) = disp;
                    disparityMaxNCorr(gThreadPos.y, gThreadPos.x) = nCorr;
                }
            }
            __syncthreads();

            // Rolling window to compute the rest of the rows for this thread
            texCoords.y = gThreadPos.y - windowRad;
            for (row = 1; row < ROWS_PER_THREAD && row + gThreadPos.y < imgLeft.rows + windowRad;
                 row++) {
                // Subtract the value of the first row from the column sums
                templVal = tex2D(textureLeft, texCoords.x, texCoords.y);
                imgVal = tex2D(textureRight, texCoords.x + disp, texCoords.y);
                products[threadIdx.x] -= templVal * imgVal;
                acorrTemplProducts[threadIdx.x] -= templVal * templVal;
                acorrImgProducts[threadIdx.x] -= imgVal * imgVal;

                // Add in the value from the next row down
                templVal = tex2D(textureLeft, texCoords.x, texCoords.y + 2 * windowRad + 1);
                imgVal = tex2D(textureRight, texCoords.x + disp, texCoords.y + 2 * windowRad + 1);
                products[threadIdx.x] += templVal * imgVal;
                acorrTemplProducts[threadIdx.x] += templVal * templVal;
                acorrImgProducts[threadIdx.x] += imgVal * imgVal;

                // Handle edges
                if (extraReadVal > 0) {
                    // Subtract old values
                    templVal = tex2D(textureLeft, texCoords.x + blockDim.x, texCoords.y);
                    imgVal = tex2D(textureRight, texCoords.x + disp + blockDim.x, texCoords.y);
                    products[threadIdx.x + blockDim.x] -= templVal * imgVal;
                    acorrTemplProducts[threadIdx.x + blockDim.x] -= templVal * templVal;
                    acorrImgProducts[threadIdx.x + blockDim.x] -= imgVal * imgVal;

                    templVal = tex2D(
                        textureLeft, texCoords.x + blockDim.x, texCoords.y + 2 * windowRad + 1);
                    imgVal = tex2D(textureRight,
                                   texCoords.x + disp + blockDim.x,
                                   texCoords.y + 2 * windowRad + 1);
                    products[extraReadVal] += templVal * imgVal;
                    acorrTemplProducts[extraReadVal] += templVal * templVal;
                    acorrImgProducts[extraReadVal] += imgVal * imgVal;
                }
                texCoords.y++;
                __syncthreads();

                // Accumulate the total
                if (gThreadPos.x < imgLeft.cols && gThreadPos.y + row < imgLeft.rows) {
                    nCorr = autocorrTempl = autocorrImg = 0;
                    for (i = 0; i < 2 * windowRad; i++) {
                        nCorr += products[i + threadIdx.x];
                        autocorrTempl += acorrTemplProducts[i + threadIdx.x];
                        autocorrImg += acorrImgProducts[i + threadIdx.x];
                    }

                    // Normalize
                    nCorr = nCorr / sqrtf(autocorrTempl * autocorrImg);

                    if (nCorr > disparityMaxNCorr(gThreadPos.y + row, gThreadPos.x)) {
                        disparity(gThreadPos.y + row, gThreadPos.x) = disp;
                        disparityMaxNCorr(gThreadPos.y + row, gThreadPos.x) = nCorr;
                    }
                }
                __syncthreads();
            }
        }
    }
}

void cuda::disparityNCorr(const cv::Mat& left,
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
    cv::cuda::GpuMat d_left, d_right, d_disparity, d_disparityMaxNCorr;
    cv::cuda::Stream cpyStream;
    // Set up a callback to log a confirmation that the stream copy has completed
    cpyStream.enqueueHostCallback(common::checkCopySuccess, nullptr);

    // Allocate GPU memory
    // Note that I don't use createContinuous since the texture binding operations requires pitched
    // memory
    d_left.create(left.rows, left.cols, left.type());
    d_right.create(right.rows, right.cols, right.type());
    d_disparity.create(disparity.rows, disparity.cols, disparity.type());
    d_disparityMaxNCorr.create(disparity.rows, disparity.cols, CV_32FC1);
    d_left.upload(left, cpyStream);
    d_right.upload(right, cpyStream);
    d_disparity.setTo(cv::Scalar(-1), cpyStream);
    d_disparityMaxNCorr.setTo(cv::Scalar(MIN_DISP_SSD), cpyStream);

    // Set up texture memory
    cv::cuda::device::bindTexture<float>(&textureLeft, d_left);
    cv::cuda::device::bindTexture<float>(&textureRight, d_right);

    // Determine block and grid size
    dim3 blocks(common::divRoundUp(left.cols, TILE_SIZE_X),
                common::divRoundUp(left.rows, ROWS_PER_THREAD));
    dim3 threads(TILE_SIZE_X, TILE_SIZE_Y);
    // Shared mem size is the width of one block, plus the window size. Allocate the same amount of
    // space for the autocorrelation accumulation for the template and the image window.
    const size_t shmPitch = TILE_SIZE_X + 2 * windowRad;
    size_t shmSize = 3 * shmPitch * sizeof(float);

    // Time kernel execution
    logger->info("Launching disparityNCorrKernel");
    GpuTimer timer;
    timer.start();

    disparityNCorrKernel<float, char>
        <<<blocks, threads, shmSize, cv::cuda::StreamAccessor::getStream(cpyStream)>>>(
            d_left,
            d_right,
            windowRad,
            minDisparity,
            maxDisparity,
            shmPitch,
            d_disparity,
            d_disparityMaxNCorr);

    cudaDeviceSynchronize();
    checkCudaErrors(cudaGetLastError());

    timer.stop();
    logger->info("disparityNCorrKernel execution took {} ms", timer.getTime());

    // Copy back to CPU
    d_disparity.download(disparity);
}