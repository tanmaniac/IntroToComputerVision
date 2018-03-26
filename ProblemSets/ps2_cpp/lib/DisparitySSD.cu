#include <common/GpuTimer.h>
#include <common/CudaCommon.cuh>
#include "../include/DisparitySSD.h"

#include <spdlog/spdlog.h>

#include <opencv2/core/core.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/core/cuda/common.hpp>
#include <opencv2/core/cuda_stream_accessor.hpp>
//#include <opencv2/cudev/ptr2d/texture.hpp>
#include <opencv2/core/opengl.hpp>

#define STEPS 3
#define BLOCKDIM_X 8
#define BLOCKDIM_Y 32
#define WINDOW_RAD 8

texture<float, cudaTextureType2D, cudaReadModeElementType> textureLeft;
texture<float, cudaTextureType2D, cudaReadModeElementType> textureRight;

// Does not clamp access!
template <typename T>
struct Array2DWrapper {
    // Data array on device
    T* _data;
    size_t _rows, _cols;

    __device__ Array2DWrapper(T* data, const size_t rows, const size_t cols)
        : _data(data), _rows(rows), _cols(cols) {}

    // Access array at index with row value y and column value x
    __device__ T& operator()(const int y, const int x) {
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
                                   const int windowRad,
                                   const int minDisparity,
                                   const int maxDisparity,
                                   cv::cuda::PtrStepSz<dispType> disparity) {
    const int2 gThreadPos = make_int2(blockIdx.x * blockDim.x + threadIdx.x,
                                      blockIdx.y * blockDim.y + threadIdx.y);

    if (gThreadPos.x > imgLeft.cols || gThreadPos.y > imgLeft.rows) {
        return;
    }

    // Shared memory indexing
    const uint2 sThreadPos = make_uint2(threadIdx.x + windowRad, threadIdx.y + windowRad);

    srcType imLeft, imRight;
    int rawCost;
    int bestCost = INT_MAX;
    int bestDisparity = 0;

    // Declare shared memory
    extern __shared__ dispType ssdDiffs[];
    Array2DWrapper<dispType> diffs(
        ssdDiffs, blockDim.y + 2 * windowRad, blockDim.x + 2 * windowRad);

    // Copy edge points from left image into local reigsters
    srcType imLeftA[STEPS];
    srcType imLeftB[STEPS];

    for (int i = 0; i < STEPS; i++) {
        int offset = -1 * int(windowRad) + i * windowRad;
        imLeftA[i] = tex2D(textureLeft, gThreadPos.x - int(windowRad), gThreadPos.y + offset);
        imLeftB[i] =
            tex2D(textureLeft, gThreadPos.x - int(windowRad) + blockDim.x, gThreadPos.y + offset);
    }

    for (int disp = minDisparity; disp <= maxDisparity; disp++) {
        // Left side
        for (int i = 0; i < STEPS; i++) {
            int offset = -1 * int(windowRad) + i * windowRad;
            imLeft = imLeftA[i];
            imRight =
                tex2D(textureRight, gThreadPos.x - int(windowRad) + disp, gThreadPos.y + offset);
            //rawCost = roundf(__powf(imLeft - imRight, 2.f));
            rawCost = fabsf(imLeft - imRight);
            diffs(sThreadPos.y + offset, sThreadPos.x - int(windowRad)) = rawCost;
        }

        // Right side
        for (int i = 0; i < STEPS; i++) {
            int offset = -1 * int(windowRad) + i * windowRad;
            if (threadIdx.x < 2 * windowRad) {
                imLeft = imLeftB[i];
                imRight = tex2D(textureRight,
                                gThreadPos.x - int(windowRad) + blockDim.x + disp,
                                gThreadPos.y + offset);
                //rawCost = roundf(__powf(imLeft - imRight, 2.f));
                rawCost = fabsf(imLeft - imRight);
                diffs(sThreadPos.y + offset, sThreadPos.x - int(windowRad) + blockDim.x) =
                    rawCost;
            }
        }

        __syncthreads();

        // Sum cost horizontally
        for (int j = 0; j < STEPS; j++) {
            int offset = -1 * int(windowRad) + j * windowRad;
            rawCost = 0;

            for (int i = -1 * int(windowRad); i <= windowRad; i++) {
                rawCost += diffs(sThreadPos.y + offset, sThreadPos.x + i);
            }

            __syncthreads();
            diffs(sThreadPos.y + offset, sThreadPos.x) = rawCost;
            __syncthreads();
        }

        // Sum cost vertically
        rawCost = 0;

        for (int i = -1 * int(windowRad); i <= windowRad; i++) {
            rawCost += diffs(sThreadPos.y + i, sThreadPos.x);
        }

        // Determine if rawCost is better than the previous best
        if (rawCost < bestCost) {
            bestCost = rawCost;
            bestDisparity = disp + 8;
        }

        __syncthreads();
    }

    if (gThreadPos.y < imgLeft.cols && gThreadPos.x < imgLeft.rows) {
        disparity(gThreadPos.y, gThreadPos.x) = bestDisparity;
    }
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
    // Don't really want to deal with templating this right now
    assert(left.type() == CV_32FC1 && right.type() == CV_32FC1);

    static constexpr size_t TILE_SIZE_X = 32;
    static constexpr size_t TILE_SIZE_Y = 8;

    // Set up file loggers
    auto logger = spdlog::get("file_logger");
    logger->info("Setting up CUDA kernel execution...");
    logger->info("Original image: rows={} cols={}", left.rows, left.cols);

    disparity.create(cv::Size(left.rows, left.cols), CV_8SC1);

    // Copy to GPU memory. cv::GpuMat::upload() calls are blocking, so copy in CUDA streams
    cv::cuda::GpuMat d_left, d_right, d_disparity;
    cv::cuda::Stream cpyStream;
    cpyStream.enqueueHostCallback(checkCopySuccess, NULL);

    d_left.create(left.rows, left.cols, left.type());
    d_right.create(right.rows, right.cols, right.type());
    d_disparity.create(disparity.rows, disparity.cols, disparity.type());
    d_left.upload(left, cpyStream);
    d_right.upload(right, cpyStream);
    d_disparity.upload(disparity, cpyStream);

    // Set up texture memory
    /*cudaChannelFormatDesc caDesc0 = cudaCreateChannelDesc<float>();
    cudaChannelFormatDesc caDesc1 = cudaCreateChannelDesc<float>();
    // Make texture memory clamp out-of-bounds indices
    textureLeft.addressMode[0] = cudaAddressModeClamp;
    textureLeft.addressMode[1] = cudaAddressModeClamp;
    textureLeft.filterMode = cudaFilterModePoint;
    textureLeft.normalized = false;
    textureRight.addressMode[0] = cudaAddressModeClamp;
    textureRight.addressMode[1] = cudaAddressModeClamp;
    textureRight.filterMode = cudaFilterModePoint;
    textureRight.normalized = false;

    float* buffer;
    static constexpr size_t N = 1024;
    checkCudaErrors(cudaMalloc(&buffer, N * sizeof(float)));
    cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeLinear;
    resDesc.res.linear.devPtr = buffer;
    resDesc.res.linear.desc.f = cudaChannelFormatKindFloat;
    resDesc.res.linear.desc.x = 32;
    resDesc.res.linear.sizeInBytes = N * sizeof(float);

    cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.readMode = cudaReadModeElementType;

    cudaTextureObject_t tex = 0;
    cudaCreateTextureObject(&tex, &resDesc, &texDesc, NULL);*/

    cv::cuda::device::bindTexture<float>(&textureLeft, d_left);
    cv::cuda::device::bindTexture<float>(&textureRight, d_right);

    // Determine block and grid size
    dim3 blocks(max(1, (unsigned int)ceil(float(left.cols) / float(TILE_SIZE_X))),
                max(1, (unsigned int)ceil(float(left.rows) / float(TILE_SIZE_Y))));
    dim3 threads(TILE_SIZE_X, TILE_SIZE_Y);
    size_t shmSize =
        ((TILE_SIZE_X + 2 * windowRad) * (TILE_SIZE_Y + 2 * windowRad)) * sizeof(char);

    // Time kernel execution
    logger->info("Launching disparitySSDKernel");
    GpuTimer timer;
    timer.start();

    disparitySSDKernel<float, unsigned char>
        <<<blocks, threads, shmSize, cv::cuda::StreamAccessor::getStream(cpyStream)>>>(
            d_left, d_right, frame, windowRad, minDisparity, maxDisparity, d_disparity);

    cudaDeviceSynchronize();
    checkCudaErrors(cudaGetLastError());

    timer.stop();
    logger->info("disparitySSDKernel execution took {} ms", timer.getTime());

    // Copy back to CPU
    d_disparity.download(disparity);
}