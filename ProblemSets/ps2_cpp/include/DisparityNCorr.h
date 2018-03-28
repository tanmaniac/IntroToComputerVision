#pragma once

#include <opencv2/core/core.hpp>

// Functions for computing disparity using normalized correlation

namespace cuda {
/**
 * \brief Computes disparity between two images using normalized correlation.
 * \param left "reference" image
 * \param right target image to compare against
 * \param windowRad Radius of the summing window, e.g. 5 = 11x11 window
 * \param minDisparity Minimum disparity value. Should be negative if a left side image is used as
 *        a reference image.
 * \param maxDisparity Maximum disparity value. Should be positive if a right side image is used as
 *        a reference image.
 * \param disparity Output disparity matrix.
 */
void disparityNCorr(const cv::Mat& left,
                    const cv::Mat& right,
                    const size_t windowRad,
                    const int minDisparity,
                    const int maxDisparity,
                    cv::Mat& disparity);
}; // namespace cuda

namespace serial {
/**
 * \brief Computes disparity between two images using normalized correlation.
 * \param left "reference" image
 * \param right target image to compare against
 * \param windowRad Radius of the summing window, e.g. 5 = 11x11 window
 * \param minDisparity Minimum disparity value. Should be negative if a left side image is used as
 *        a reference image.
 * \param maxDisparity Maximum disparity value. Should be positive if a right side image is used as
 *        a reference image.
 * \param disparity Output disparity matrix.
 */
void disparityNCorr(const cv::Mat& left,
                    const cv::Mat& right,
                    const size_t windowRad,
                    const int minDisparity,
                    const int maxDisparity,
                    cv::Mat& disparity);
}; // namespace serial