#pragma once

#include <Eigen/Dense>
#include <opencv2/core/core.hpp>

namespace fundamental {
/**
 * \brief  Compute the fundamental matrix estimating the mapping between points in two cameras,
 * solving the linear system with normal equations.
 *
 * \param pts2dA 2D points from camera A
 * \param pts2dB 2D points from camera B
 */
Eigen::MatrixXf solveLeastSquares(const Eigen::MatrixXf& pts2dA, const Eigen::MatrixXf& pts2d);

// Overload of the above function using cv::Mat instead of Eigen
cv::Mat solveLeastSquares(const cv::Mat& pts2dA, const cv::Mat& pts2dB);

/**
 * \brief Reduce rank of fundamental matrix from 3 to 2
 *
 * \param fMat input matrix
 */
Eigen::MatrixXf rankReduce(const Eigen::MatrixXf& fMat);

cv::Mat rankReduce(const cv::Mat& fMat);
}; // namespace fundamental