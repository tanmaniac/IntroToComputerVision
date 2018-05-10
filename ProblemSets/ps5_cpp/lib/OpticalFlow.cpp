#include "../include/OpticalFlow.h"
#include <common/Utils.h>
#include "../include/Pyramids.h"

#include <opencv2/core/cuda.hpp>
#include <opencv2/cudafilters.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <iostream>

void computeGradients(const cv::Mat& in, cv::Mat& outX, cv::Mat& outY) {
    assert(in.type() == CV_32F);

    // Set up async stream
    cv::cuda::Stream stream;
    stream.enqueueHostCallback(common::checkCopySuccess, nullptr);
    // Use a sobel operator to get the gradient
    float scale = 1.f / 9.f;
    auto sobelX = cv::cuda::createSobelFilter(in.type(), -1, 1, 0, 3, scale);
    auto sobelY = cv::cuda::createSobelFilter(in.type(), -1, 0, 1, 3, scale);

    outX.create(in.size(), in.type());
    outY.create(in.size(), in.type());

    // Copy to GPU memory
    cv::cuda::GpuMat d_in, d_diffX, d_diffY, d_blurX, d_blurY;
    d_in.upload(in, stream);
    d_diffX.create(in.size(), in.type());
    d_diffY.create(in.size(), in.type());

    // Apply sobel operator in X and Y direction
    sobelX->apply(d_in, d_diffX, stream);
    sobelY->apply(d_in, d_diffY, stream);

    // Copy back to CPU
    d_diffX.download(outX, stream);
    d_diffY.download(outY, stream);
}

void lk::calcOpticalFlow(const cv::Mat& prevImg,
                         const cv::Mat& nextImg,
                         cv::Mat& u,
                         cv::Mat& v,
                         const size_t winSize) {
    // Make sure input images are the correct size and type
    assert(prevImg.size() == nextImg.size() && prevImg.type() == nextImg.type());
    assert(winSize % 2 == 1);

    cv::Mat prev, next;
    prevImg.convertTo(prev, CV_32F);
    nextImg.convertTo(next, CV_32F);
    u.create(prev.size(), CV_32F);
    v.create(prev.size(), CV_32F);

    // Compute gradients
    cv::Mat Sxx, Sxy, Syy, Sxt, Syt;
    {
        cv::Mat nextIx, nextIy, prevIx, prevIy;
        computeGradients(prev, prevIx, prevIy);
        computeGradients(next, nextIx, nextIy);
        cv::Mat Ix = (nextIx + prevIx) / 2.f;
        cv::Mat Iy = (nextIy + prevIy) / 2.f;
        cv::Mat It = next - prev;

        Sxx = Ix.mul(Ix);
        Sxy = Ix.mul(Iy);
        Syy = Iy.mul(Iy);
        Sxt = Ix.mul(It);
        Syt = Iy.mul(It);

        // Sum up values in the windowed area. Convolution is just a weighted sum
        cv::GaussianBlur(Sxx, Sxx, cv::Size(winSize, winSize), float(winSize) / 3.f);
        cv::GaussianBlur(Sxy, Sxy, cv::Size(winSize, winSize), float(winSize) / 3.f);
        cv::GaussianBlur(Syy, Syy, cv::Size(winSize, winSize), float(winSize) / 3.f);
        cv::GaussianBlur(Sxt, Sxt, cv::Size(winSize, winSize), float(winSize) / 3.f);
        cv::GaussianBlur(Syt, Syt, cv::Size(winSize, winSize), float(winSize) / 3.f);
    }

    // Threshold for A matrix determining movement. If the determinant of A is sufficiently small,
    // then the matrix is (or is close to) singular, so there is no motion.
    static constexpr double tau = 0.1;

    // Iterate over the whole image and solve the least-squares equation for [u; v]
    for (int y = 0; y < prev.rows; y++) {
        for (int x = 0; x < prev.cols; x++) {
            cv::Mat A = (cv::Mat_<float>(2, 2) << Sxx.at<float>(y, x),
                         Sxy.at<float>(y, x),
                         Sxy.at<float>(y, x),
                         Syy.at<float>(y, x));
            cv::Mat b = (cv::Mat_<float>(2, 1) << -Sxt.at<float>(y, x), -Syt.at<float>(y, x));

            // If the determinant of A is less than the threshold, just set its displacement to 0
            cv::Mat uv;
            if (cv::determinant(A) < tau) {
                uv = (cv::Mat_<float>(2, 1) << 0, 0);
            } else {
                cv::solve(A, b, uv);
            }
            u.at<float>(y, x) = uv.at<float>(0, 0);
            v.at<float>(y, x) = uv.at<float>(1, 0);
        }
    }
}

void lk::warp(const cv::Mat& src, const cv::Mat& du, const cv::Mat& dv, cv::Mat& dst) {
    assert(du.size() == dv.size() && src.size() == du.size());
    assert(du.type() == CV_32F && du.type() == dv.type());
    // Iterate over the u and v displacements and determine where these pixels should move to
    cv::Mat newU(du.size(), du.type()), newV(dv.size(), dv.type());
    for (int y = 0; y < du.rows; y++) {
        for (int x = 0; x < du.cols; x++) {
            newU.at<float>(y, x) = x + du.at<float>(y, x);
            newV.at<float>(y, x) = y + dv.at<float>(y, x);
        }
    }

    // Remap to warp these points to the correct locations
    cv::remap(src, dst, newU, newV, cv::INTER_LINEAR);
}

void lk::calcOpticalFlowPyr(const cv::Mat& prevImg,
                            const cv::Mat& nextImg,
                            cv::Mat& u,
                            cv::Mat& v,
                            const size_t winSize) {
    const size_t pyrDepth = 4;
    std::vector<cv::Mat> prevPyr = pyr::makeGaussianPyramid(prevImg, pyrDepth);
    std::vector<cv::Mat> nextPyr = pyr::makeGaussianPyramid(nextImg, pyrDepth);

    // Initialize displacement matrices
    cv::Mat du = cv::Mat::zeros(prevPyr[pyrDepth - 1].size(), CV_32F);
    cv::Mat dv = cv::Mat::zeros(prevPyr[pyrDepth - 1].size(), CV_32F);

    for (int level = 0; level < pyrDepth; level++) {
        cv::Mat& prevK = prevPyr[pyrDepth - level - 1];
        cv::Mat& nextK = nextPyr[pyrDepth - level - 1];

        // EXPAND du and dv after the first iteration
        if (level > 0) {
            pyr::pyrUp(du, du);
            du = 2 * du;
            pyr::pyrUp(dv, dv);
            dv = 2 * dv;
        }

        // Resize the du and dv displacements if they're not the correct size
        if (du.rows != nextK.rows || du.cols != nextK.cols) {
            cv::resize(du, du, nextK.size());
            cv::resize(dv, dv, nextK.size());
        }

        // Warp the "next" img back to the prev img using the displacement
        cv::Mat warped;
        lk::warp(nextK, du, dv, warped);

        // Compute optical flow
        cv::Mat dx, dy;
        lk::calcOpticalFlow(prevK, warped, dx, dy, winSize);

        du = du + dx;
        dv = dv + dy;
    }

    u = du;
    v = dv;
}