#pragma once

#include <opencv2/core/core.hpp>

// Compute Harris corners of an image

namespace harris {
enum class GradientDirection { X, Y };

void getGradients(const cv::Mat& in, int kernelSize, cv::Mat& outX, cv::Mat& outY);

void getHarrisValues(const cv::Mat& in,
                     const size_t windowSize,
                     const double gaussianSigma,
                     cv::Mat& harrisVals);
}; // namespace harris