#pragma once

#include <opencv2/core/core.hpp>

// Function definitions that are used between CUDA and C++
void rowConvolution(const cv::Mat& input, const cv::Mat& kernel, cv::Mat& dest);

void columnConvolution(const cv::Mat& input, const cv::Mat& kernel, cv::Mat& dest);

void rowConvolution(const cv::cuda::GpuMat& d_input,
                    const cv::cuda::GpuMat& d_kernel,
                    cv::cuda::GpuMat& d_dest);

void columnConvolution(const cv::cuda::GpuMat& d_input,
                       const cv::cuda::GpuMat& d_kernel,
                       cv::cuda::GpuMat& d_dest);

// Convolve will do both row and column convolution steps together, so there doesn't need to be
// a buffer between each step. Params:
//  input       OpenCV Matrix (CPU) of the input image. Must be grayscale, type CV_32FC1
//  rowKernel   First row of a separable filter kernel. Type CV_32FC1
//  colKernel   First column of the separable filter kernel. Type CV_32FC1
//  dest        Destination matrix to which the result should be written.
void separableConvolution(const cv::Mat& input,
                          const cv::Mat& rowKernel,
                          const cv::Mat& colKernel,
                          cv::Mat& dest);