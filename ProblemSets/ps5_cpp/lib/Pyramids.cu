#include <common/GpuTimer.h>
#include <common/Utils.h>
#include <common/CudaCommon.cuh>
#include "../include/Config.h"
#include "../include/Pyramids.h"

#include <spdlog/spdlog.h>

#include <opencv2/core/cuda.hpp>
#include <opencv2/core/cuda/common.hpp>
#include <opencv2/core/cuda_stream_accessor.hpp>
#include <opencv2/cudafilters.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <iostream>

// Create separable 5-tap Gaussian
cv::Mat gaussKernel = (cv::Mat_<float>(1, 5) << 0.0625, 0.25, 0.375, 0.25, 0.0625);

template <typename T>
__global__ void pyrDownsampleKernel(const cv::cuda::PtrStepSz<T> src, cv::cuda::PtrStepSz<T> dst) {
    const uint2 gThreadPos = getPosition();

    if (gThreadPos.x >= src.cols || gThreadPos.y >= src.rows) {
        return;
    }

    // Get all the pixels whose row and col are both odd and place them at this thread's index in
    // the output image
    dst(gThreadPos.y, gThreadPos.x) = src(gThreadPos.y * 2 + 1, gThreadPos.x * 2 + 1);
}

void pyr::pyrDown(const cv::Mat& src, cv::Mat& dst) {
    assert(src.type() == CV_32F);

    auto flogger = spdlog::get(config::FILE_LOGGER);

    // Set up async stream
    cv::cuda::Stream stream;
    stream.enqueueHostCallback(common::checkCopySuccess, nullptr);

    // Set up GPU memory for gaussian
    cv::cuda::GpuMat d_src, d_blurred;
    d_src.upload(src, stream);
    d_blurred.create(src.size(), src.type());

    // Blur the input image
    auto gauss = cv::cuda::createSeparableLinearFilter(d_src.type(), -1, gaussKernel, gaussKernel);
    gauss->apply(d_src, d_blurred, stream);

    // Downsample
    cv::cuda::GpuMat d_down(src.rows / 2, src.cols / 2, src.type());
    static constexpr size_t TILE_SIZE = 16;

    // Set up kernel execution
    dim3 blocks(common::divRoundUp(src.cols / 2, TILE_SIZE),
                common::divRoundUp(src.rows / 2, TILE_SIZE));
    dim3 threads(TILE_SIZE, TILE_SIZE);

    GpuTimer timer;
    timer.start();

    // Launch kernel
    pyrDownsampleKernel<float>
        <<<blocks, threads, 0, cv::cuda::StreamAccessor::getStream(stream)>>>(d_src, d_down);

    timer.stop();
    flogger->info("pyrDownsampleKernel took {} ms", timer.getTime());

    dst.create(d_down.size(), d_down.type());
    d_down.download(dst, stream);
}

template <typename T>
__global__ void pyrUpsampleKernel(const cv::cuda::PtrStepSz<T> src, cv::cuda::PtrStepSz<T> dst) {
    assert(dst.rows == src.rows * 2 && dst.cols == src.cols * 2);
    const uint2 gThreadPos = getPosition();

    if (gThreadPos.x >= src.cols || gThreadPos.y >= src.rows) {
        return;
    }

    // Move each pixel to the index at two times its row and column, effectively padding each row
    // and column with 0s in between
    T px = src(gThreadPos.y, gThreadPos.x);
    uint2 destPos = make_uint2(gThreadPos.x * 2, gThreadPos.y * 2);
    dst(destPos.y, destPos.x) = px;
    dst(destPos.y + 1, destPos.x) = px;
    dst(destPos.y + 1, destPos.x + 1) = px;
    dst(destPos.y, destPos.x + 1) = px;
}

void pyr::pyrUp(const cv::Mat& src, cv::Mat& dst) {
    assert(src.type() == CV_32F);

    auto flogger = spdlog::get(config::FILE_LOGGER);

    // Set up async stream
    cv::cuda::Stream stream;
    stream.enqueueHostCallback(common::checkCopySuccess, nullptr);

    // Set up GPU memory for upsampling and gaussian
    cv::cuda::GpuMat d_src, d_blurred, d_up;
    d_src.upload(src, stream);
    d_up.create(src.rows * 2, src.cols * 2, src.type());
    d_blurred.create(d_up.size(), d_up.type());

    // Upsample
    // Set up kernel execution
    static constexpr size_t TILE_SIZE = 16;
    dim3 blocks(common::divRoundUp(src.cols, TILE_SIZE), common::divRoundUp(src.rows, TILE_SIZE));
    dim3 threads(TILE_SIZE, TILE_SIZE);

    GpuTimer timer;
    timer.start();

    // Launch kernel
    pyrUpsampleKernel<float>
        <<<blocks, threads, 0, cv::cuda::StreamAccessor::getStream(stream)>>>(d_src, d_up);

    timer.stop();
    flogger->info("pyrUpsampleKernel took {} ms", timer.getTime());

    // Blur the result of the upsample step
    auto gauss = cv::cuda::createSeparableLinearFilter(d_src.type(), -1, gaussKernel, gaussKernel);
    gauss->apply(d_up, d_blurred, stream);

    dst.create(d_up.size(), d_up.type());
    d_blurred.download(dst, stream);
}