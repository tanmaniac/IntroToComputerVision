#include "../include/Fundamental.h"

#include <opencv2/core/eigen.hpp>

#include <iostream>

Eigen::MatrixXf fundamental::solveLeastSquares(const Eigen::MatrixXf& pts2dA,
                                               const Eigen::MatrixXf& pts2dB) {
    assert(pts2dA.cols() == pts2dB.cols() && pts2dA.rows() == pts2dB.rows() && pts2dA.rows() == 2);
    // Set up A and b matrices.
    const size_t rows = pts2dB.cols();
    const size_t cols = 8;
    Eigen::MatrixXf A(rows, cols);
    Eigen::MatrixXf b = Eigen::MatrixXf::Constant(rows, 1, -1);

    // Build A and b matrices
    for (int i = 0; i < rows; i++) {
        float u = pts2dA(0, i);
        float v = pts2dA(1, i);
        float u_p = pts2dB(0, i);
        float v_p = pts2dB(1, i);
        A.row(i) << u * u_p, v * u_p, u_p, u * v_p, v * v_p, v_p, u, v;
    }

    // Solve least squares
    Eigen::MatrixXf sol = (A.transpose() * A).ldlt().solve(A.transpose() * b);
    // Append a 1 to the end since scale is constant
    sol.conservativeResize(sol.rows() + 1, sol.cols());
    sol(sol.rows() - 1, 0) = 1;
    return sol;
}

cv::Mat fundamental::solveLeastSquares(const cv::Mat& pts2dA, const cv::Mat& pts2dB) {
    // Wrap OpenCV matrices in Eigen::Map
    Eigen::MatrixXf eigenPts2dA, eigenPts2dB;
    cv::cv2eigen(pts2dA, eigenPts2dA);
    cv::cv2eigen(pts2dB, eigenPts2dB);

    auto eigenSol = solveLeastSquares(eigenPts2dA, eigenPts2dB);
    cv::Mat cvSol;
    cv::eigen2cv(eigenSol, cvSol);
    return cvSol;
}

Eigen::MatrixXf fundamental::rankReduce(const Eigen::MatrixXf& fMat) {
    // Make sure this is a valid input matrix
    assert(fMat.cols() == 3 && fMat.rows() == 3);

    // Compute SVD of fundamental matrix
    auto svd = fMat.jacobiSvd(Eigen::ComputeFullU | Eigen::ComputeFullV);

    // Set smallest singular value to zero
    auto sigma2 = svd.singularValues();
    sigma2(sigma2.rows() - 1, sigma2.cols() - 1) = 0;

    Eigen::MatrixXf r2FMat = svd.matrixU() * sigma2.asDiagonal() * svd.matrixV().transpose();

    return r2FMat;
}

cv::Mat fundamental::rankReduce(const cv::Mat& fMat) {
    // Convert OpenCV matrix to Eigen
    Eigen::MatrixXf eigenFMat;
    cv::cv2eigen(fMat, eigenFMat);

    auto eigenSol = rankReduce(eigenFMat);
    cv::Mat cvSol;
    cv::eigen2cv(eigenSol, cvSol);
    return cvSol;
}