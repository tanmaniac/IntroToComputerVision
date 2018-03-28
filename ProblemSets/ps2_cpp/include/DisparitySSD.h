#pragma once

#include <opencv2/core/core.hpp>

// Parallelized implementation of disparity using sum-of-squared-differences.
namespace cuda {
/**
 * \brief Compute disparity on GPU with sum-of-squared-differences.
 * \param left "reference" image
 * \param right target image to compare against
 * \param windowRad Radius of the summing window, e.g. 5 = 11x11 window
 * \param minDisparity Minimum disparity value. Should be negative if a left side image is used as
 *        a reference image.
 * \param maxDisparity Maximum disparity value. Should be positive if a right side image is used as
 *        a reference image.
 * \param disparity Output disparity matrix.
 */
void disparitySSD(const cv::Mat& left,
                  const cv::Mat& right,
                  const size_t windowRad,
                  const int minDisparity,
                  const int maxDisparity,
                  cv::Mat& disparity);
}; // namespace cuda

namespace serial {
/**
 * \brief Compute disparity on CPU with sum-of-squared-differences.
 * \param left "reference" image
 * \param right target image to compare against
 * \param windowRad Radius of the summing window, e.g. 5 = 11x11 window
 * \param minDisparity Minimum disparity value. Should be negative if a left side image is used as
 *        a reference image.
 * \param maxDisparity Maximum disparity value. Should be positive if a right side image is used as
 *        a reference image.
 * \param disparity Output disparity matrix.
 */
void disparitySSD(const cv::Mat& left,
                  const cv::Mat& right,
                  const size_t windowRad,
                  const int minDisparity,
                  const int maxDisparity,
                  cv::Mat& disparity);
}; // namespace serial