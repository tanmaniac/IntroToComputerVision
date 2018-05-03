#pragma once

#include <opencv2/core/core.hpp>
#include <opencv2/core/types.hpp>

namespace sift {
// Create an "angle" image from X and Y gradients of an input image
void getAnglesFromGradients(const cv::Mat& gradX, const cv::Mat& gradY, cv::Mat& angles);

/**
 * \brief getKeypoints Computes gradient angles for a set of points in an image
 *
 * \param gradX Gradient in the x-direction of an image
 * \param gradY Gradient in the y-direction of an image
 * \param cornerLocs Set of (y, x) coordinates of points in the image to treat as keypoints
 * \param size Keypoint size
 * \param keypoints output vector of cv::KeyPoint objects
 */
void getKeypoints(const cv::Mat& gradX,
                  const cv::Mat& gradY,
                  const std::vector<std::pair<int, int>>& cornerLocs,
                  const size_t size,
                  std::vector<cv::KeyPoint>& keypoints);
}; // namespace sift