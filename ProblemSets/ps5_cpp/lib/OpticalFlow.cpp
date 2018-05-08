#include "../include/OpticalFlow.h"
#include <common/Utils.h>

#include <opencv2/core/cuda.hpp>
#include <opencv2/cudafilters.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <iostream>

void computeGradientsAndBlur(const cv::Mat& in,
                             cv::Mat& outX,
                             cv::Mat& outY,
                             const size_t gaussSize = 25,
                             const double gaussSigma = 15) {
    assert(in.type() == CV_32F);

    // Set up async stream
    cv::cuda::Stream stream;
    stream.enqueueHostCallback(common::checkCopySuccess, nullptr);
    // Use a sobel operator to get the gradient
    auto scharrX = cv::cuda::createScharrFilter(in.type(), in.type(), 1, 0);
    auto scharrY = cv::cuda::createScharrFilter(in.type(), in.type(), 0, 1);

    auto gauss = cv::cuda::createGaussianFilter(
        in.type(), in.type(), cv::Size(gaussSize, gaussSize), gaussSigma);

    outX.create(in.size(), in.type());
    outY.create(in.size(), in.type());

    // Copy to GPU memory
    cv::cuda::GpuMat d_in, d_diffX, d_diffY, d_blurX, d_blurY;
    d_in.upload(in, stream);
    d_diffX.create(in.size(), in.type());
    d_diffY.create(in.size(), in.type());
    d_blurX.create(in.size(), in.type());
    d_blurY.create(in.size(), in.type());

    // Apply sobel operator in X and Y direction
    scharrX->apply(d_in, d_diffX, stream);
    scharrY->apply(d_in, d_diffY, stream);
    gauss->apply(d_diffX, d_blurX, stream);
    gauss->apply(d_diffY, d_blurY, stream);

    d_blurX.download(outX, stream);
    d_blurY.download(outY, stream);
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
    cv::Mat Ix(prev.size(), prev.type()), Iy(prev.size(), prev.type());
    computeGradientsAndBlur(prev, Ix, Iy);
    cv::Mat It = next - prev;

    // Build up matrix of weights, which is a diagonalized gaussian
    cv::Mat weights = cv::Mat::eye(winSize * winSize, winSize * winSize, CV_32F);
    cv::Mat gauss = cv::getGaussianKernel(winSize * winSize, -1, CV_32F);
    for (int i = 0; i < winSize * winSize; i++) {
        weights.at<float>(i, i) = gauss.at<float>(i, 0);
    }
    // std::cout << "weights:\n" << weights << std::endl;

    int winRad = winSize / 2;
    // Iterate over the whole image
    for (int y = 0; y < prev.rows; y++) {
        for (int x = 0; x < prev.cols; x++) {
            cv::Mat A(winSize * winSize, 2, CV_32F);
            cv::Mat b(winSize * winSize, 1, CV_32F);
            size_t row = 0;
            for (int winY = -winRad; winY <= winRad; winY++) {
                for (int winX = -winRad; winX <= winRad; winX++) {
                    cv::Point2i idx(std::min(std::max(0, x + winX), Ix.cols - 1),
                                    std::min(std::max(0, y + winY), Ix.rows - 1));
                    A.at<float>(row, 0) = Ix.at<float>(idx);
                    A.at<float>(row, 1) = Iy.at<float>(idx);
                    b.at<float>(row, 0) = It.at<float>(idx);
                    row++;
                }
            }
            cv::Mat uv, Atrans;
            cv::transpose(A, Atrans);
            // cv::solve(Atrans * weights * A, Atrans * weights * b, uv);
            cv::solve(Atrans * A, -Atrans * b, uv);
            // if (uv.at<float>(0, 0) != 0.f && uv.at<float>(1, 0) != 0.f) {
            //     std::cout << "@(" << y << "," << x << "): uv=\n" << uv << std::endl;
            // }
            u.at<float>(y, x) = uv.at<float>(0, 0);
            v.at<float>(y, x) = uv.at<float>(1, 0);
        }
    }
}