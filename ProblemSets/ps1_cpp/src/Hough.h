#pragma once

#include <opencv2/core/core.hpp>
#include <opencv2/gpu/gpu.hpp>

// C++ wrappers around CUDA kernels

void houghAccumulate(const cv::gpu::GpuMat& edgeMask,
                     const size_t rhoBinSize,
                     const size_t thetaBinSize,
                     cv::gpu::GpuMat& accumulator);

void houghAccumulate(const cv::Mat& edgeMask,
                     const size_t rhoBinSize,
                     const size_t thetaBinSize,
                     cv::Mat& accumulator);