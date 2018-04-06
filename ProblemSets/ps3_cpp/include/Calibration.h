#include <Eigen/Dense>
#include <opencv2/core/core.hpp>

namespace calib {
// Solve system of linear equations using least squares normal equations
Eigen::MatrixXf solveLeastSquares(const Eigen::MatrixXf& pts2d, const Eigen::MatrixXf& pts3d);

// Overload of the above function using cv::Mat instead of Eigen
cv::Mat solveLeastSquares(const cv::Mat& pts2d, const cv::Mat& pts3d);

// Solve system of linear equations using singular value decomposition
Eigen::MatrixXf solveSVD(const Eigen::MatrixXf& pts2d, const Eigen::MatrixXf& pts3d);

// Overload of the above function using cv::Mat instead of Eigen
cv::Mat solveSVD(const cv::Mat& pts2d, const cv::Mat& pts3d);
}; // namespace calib