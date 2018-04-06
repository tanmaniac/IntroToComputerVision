#include "../include/Calibration.h"

#include <opencv2/core/eigen.hpp>

#include <iostream>

Eigen::MatrixXf calib::solveLeastSquares(const Eigen::MatrixXf& pts2d,
                                         const Eigen::MatrixXf& pts3d) {
    assert(pts2d.cols() == pts3d.cols() && pts2d.rows() == 2 && pts3d.rows() == 3);
    // Set up A and b matrices.
    const size_t rows = pts3d.cols() * 2;
    const size_t cols = 11;
    Eigen::MatrixXf A(rows, cols);
    Eigen::MatrixXf b(rows, 1);

    // Build A and b matrices
    for (int i = 0; i < rows; i += 2) {
        float X = pts3d(0, i / 2);
        float Y = pts3d(1, i / 2);
        float Z = pts3d(2, i / 2);
        float x = pts2d(0, i / 2);
        float y = pts2d(1, i / 2);
        A.row(i) << X, Y, Z, 1, Eigen::MatrixXf::Zero(1, 4), -x * X, -x * Y, -x * Z;
        A.row(i + 1) << Eigen::MatrixXf::Zero(1, 4), X, Y, Z, 1, -y * X, -y * Y, -y * Z;
        b.row(i) << x;
        b.row(i + 1) << y;
    }

    // Solve least squares
    Eigen::MatrixXf sol = (A.transpose() * A).ldlt().solve(A.transpose() * b);
    return sol;
}

cv::Mat calib::solveLeastSquares(const cv::Mat& pts2d, const cv::Mat& pts3d) {
    // Wrap OpenCV matrices in Eigen::Map
    Eigen::MatrixXf eigenPts2d, eigenPts3d;
    cv::cv2eigen(pts2d, eigenPts2d);
    cv::cv2eigen(pts3d, eigenPts3d);

    auto eigenSol = solveLeastSquares(eigenPts2d, eigenPts3d);
    cv::Mat cvSol;
    cv::eigen2cv(eigenSol, cvSol);
    return cvSol;
}

Eigen::MatrixXf calib::solveSVD(const Eigen::MatrixXf& pts2d, const Eigen::MatrixXf& pts3d) {
    assert(pts2d.cols() == pts3d.cols() && pts2d.rows() == 2 && pts3d.rows() == 3);
    // Set up A and b matrices.
    const size_t rows = pts3d.cols() * 2;
    const size_t cols = 12;
    Eigen::MatrixXf A(rows, cols);

    // Build A and b matrices
    for (int i = 0; i < rows; i += 2) {
        float X = pts3d(0, i / 2);
        float Y = pts3d(1, i / 2);
        float Z = pts3d(2, i / 2);
        float x = pts2d(0, i / 2);
        float y = pts2d(1, i / 2);
        A.row(i) << X, Y, Z, 1, Eigen::MatrixXf::Zero(1, 4), -x * X, -x * Y, -x * Z, -x;
        A.row(i + 1) << Eigen::MatrixXf::Zero(1, 4), X, Y, Z, 1, -y * X, -y * Y, -y * Z, -y;
    }

    // Compute the orthogonal matrix of eigenvectors of A_T*A
    Eigen::MatrixXf eigenvectors = A.jacobiSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).matrixV();
    // Get the eigenvector with the smallest eigenvalue (the last column of the V matrix)
    Eigen::MatrixXf smallest = eigenvectors.col(eigenvectors.cols() - 1);
    // std::cout << "sol = \n" << smallest << std::endl;
    return smallest;
}

cv::Mat calib::solveSVD(const cv::Mat& pts2d, const cv::Mat& pts3d) {
    // Convert OpenCV matrices to Eigen matrices
    Eigen::MatrixXf eigenPts2d, eigenPts3d;
    cv::cv2eigen(pts2d, eigenPts2d);
    cv::cv2eigen(pts3d, eigenPts3d);

    auto eigenSol = solveSVD(eigenPts2d, eigenPts3d);
    cv::Mat cvSol;
    cv::eigen2cv(eigenSol, cvSol);
    return cvSol;
}