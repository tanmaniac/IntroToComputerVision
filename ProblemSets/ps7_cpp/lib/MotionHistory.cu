#include <common/OpenCVThrustInterop.h>
#include <common/Utils.h>
#include <common/CudaCommon.cuh>

#include <spdlog/spdlog.h>

#include <opencv2/core/cuda.hpp>
#include <opencv2/core/cuda/common.hpp>
#include <opencv2/core/cuda_stream_accessor.hpp>

#include <thrust/execution_policy.h>
#include <thrust/transform.h>

#include <math.h>

template <typename T>
struct AbsThreshold : public thrust::unary_function<T, T> {
    const double _threshold;

    AbsThreshold(const double threshold) : _threshold(threshold) {}

    __host__ __device__ T operator()(T val) {
        return ((val >= _threshold || -val >= _threshold) ? 1 : 0);
    }
};

void thresholdDifference(const cv::cuda::GpuMat& src,
                         const double thresh,
                         cv::cuda::GpuMat& dst,
                         cv::cuda::Stream& stream) {
    // Only support single-channel uint8_t
    assert(src.type() == CV_8UC1 && dst.type() == src.type() && src.rows == dst.rows &&
           src.cols == dst.cols);

    // Wrap input matrices in thrust iterators
    auto srcBegin = GpuMatBeginItr<uint8_t>(src);
    auto srcEnd = GpuMatEndItr<uint8_t>(src);

    auto dstBegin = GpuMatBeginItr<uint8_t>(dst);

    // Threshold and push to output
    AbsThreshold<uint8_t> thresholdOp(thresh);
    thrust::transform(thrust::system::cuda::par.on(cv::cuda::StreamAccessor::getStream(stream)),
                      srcBegin,
                      srcEnd,
                      dstBegin,
                      thresholdOp);
}

// Functions for computing motion history

template <typename T>
__global__ void motionHistoryKernel(cv::cuda::PtrStepSz<T> history,
                                    const cv::cuda::PtrStepSz<T> binaryMask,
                                    const int tau) {
    const uint2 gThreadPos = getPosition();

    if (gThreadPos.x >= history.cols || gThreadPos.y >= history.rows) {
        return;
    }

    // Update history
    T historyVal = history(gThreadPos.y, gThreadPos.x);
    history(gThreadPos.y, gThreadPos.x) =
        (binaryMask(gThreadPos.y, gThreadPos.x) == 1) ? tau : max(historyVal - 1, 0);
}

void motionHistoryKernelCall(cv::cuda::GpuMat& history,
                             const cv::cuda::GpuMat& binaryMask,
                             const int tau) {
    // 16 x 16 thread blocks
    static constexpr size_t TILE_SIZE = 16;

    dim3 blocks(common::divRoundUp(history.cols, TILE_SIZE),
                common::divRoundUp(history.rows, TILE_SIZE));
    dim3 threads(TILE_SIZE, TILE_SIZE);

    // auto flogger->info("Launching motionHistoryKernel");

    motionHistoryKernel<uint8_t><<<blocks, threads>>>(history, binaryMask, tau);
    cudaDeviceSynchronize();
    checkCudaErrors(cudaGetLastError());
}