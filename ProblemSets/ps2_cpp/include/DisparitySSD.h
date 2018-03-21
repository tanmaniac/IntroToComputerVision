#pragma once

#include <opencv2/core/core.hpp>
#include <opencv2/core/cuda.hpp>

// Defines which frame to use as the reference frame
enum class ReferenceFrame { LEFT, RIGHT };

// Parallelized implementation of disparity using sum-of-squared-differences.
namespace cuda {

// Compute disparity with sum of squared differences. This overload is intended to be called by host
// code.
void disparitySSD(const cv::Mat& left,
                  const cv::Mat& right,
                  const ReferenceFrame frame,
                  const size_t windowRad,
                  const size_t minDisparity,
                  const size_t maxDisparity,
                  cv::Mat& disparity);
}; // namespace cuda

namespace serial {
void disparitySSD(const cv::Mat& left,
                  const cv::Mat& right,
                  const ReferenceFrame frame,
                  const size_t windowRad,
                  const size_t minDisparity,
                  const size_t maxDisparity,
                  cv::Mat& disparity);
}; // namespace serial