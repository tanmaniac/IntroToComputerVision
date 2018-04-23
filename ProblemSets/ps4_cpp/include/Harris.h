#pragma once

#include <opencv2/core/core.hpp>

// Compute Harris corners of an image

namespace harris {
enum class GradientDirection { X, Y };

void getGradients(const cv::Mat& in, int kernelSize, cv::Mat& diffX, cv::Mat& diffY);

namespace cpu {
void getCornerResponse(const cv::Mat& gradX,
                       const cv::Mat& gradY,
                       const size_t windowSize,
                       const double gaussianSigma,
                       const float harrisScore,
                       cv::Mat& cornerResponse);

/**
 * \brief refineCorners Performs thresholding and non-maximum suppression on the values found in
 * getCornerResponse to find the strongest corners in an image.
 *
 * \param cornerResponse Corner response matrix found with getCornerResponse
 * \param threshold Minimum corner response threshold allowed to be called a corner
 * \param minDistance Minimum distance (x and y) between returned corners
 * \param corners Output with strongest corners marked
 */
void refineCorners(const cv::Mat& cornerResponse,
                   const double threshold,
                   const int minDistance,
                   cv::Mat& corners);
}; // namespace cpu

namespace gpu {
void getCornerResponse(const cv::Mat& gradX,
                       const cv::Mat& gradY,
                       const size_t windowSize,
                       const double gaussianSigma,
                       const float harrisScore,
                       cv::Mat& cornerResponse);
}
}; // namespace harris