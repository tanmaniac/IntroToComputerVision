#pragma once

#include <opencv2/core/core.hpp>

// Compute Harris corners of an image

namespace harris {
enum class GradientDirection { X, Y };

/**
 * \brief getGradients Computes the X and Y gradients of an input image
 *
 * \param in Input image
 * \param kernelSize Size of the Sobel operator used to compute the gradient. Must be 1, 3, 5, or 7
 * \param diffX Output gradient in the X direction
 * \param diffY Output gradient in the Y direction
 */
void getGradients(const cv::Mat& in, int kernelSize, cv::Mat& diffX, cv::Mat& diffY);

namespace cpu {
/**
 * \brief getCornerResponse Computes the corner response R for each pixel in an image, where
 *          R = det(M) - a * trace(M)^2
 *      and M is the second moment matrix
 *          M = sum(window_x,y * [I_x^2, I_x*I_y; I_x*I_y, I_y^2])
 *      window is the windowing function, in this case a Gaussian kernel, and I_x,y are the
 * gradients in the X and Y directions.
 *
 * \param gradX Gradient in X direction
 * \param gradY Gradient in Y direction
 * \param windowSize Size of the Gaussian windowing function. Must be odd.
 * \param gaussianSigma Sigma used in computing the Gaussian window.
 * \param alpha "a" in the R equation above
 * \param cornerResponse Output corner response matrix of the same size as gradX and gradY.
 */
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
/**
 * \brief getCornerResponse Computes the corner response R for each pixel in an image, where
 *          R = det(M) - a * trace(M)^2
 *      and M is the second moment matrix
 *          M = sum(window_x,y * [I_x^2, I_x*I_y; I_x*I_y, I_y^2])
 *      window is the windowing function, in this case a Gaussian kernel, and I_x,y are the
 * gradients in the X and Y directions.
 *
 * \param gradX Gradient in X direction
 * \param gradY Gradient in Y direction
 * \param windowSize Size of the Gaussian windowing function. Must be odd.
 * \param gaussianSigma Sigma used in computing the Gaussian window.
 * \param alpha "a" in the R equation above
 * \param cornerResponse Output corner response matrix of the same size as gradX and gradY.
 */
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
}; // namespace gpu
}; // namespace harris