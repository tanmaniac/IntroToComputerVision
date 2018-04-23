#include "../include/Harris.h"
#include <common/Utils.h>
#include "../include/Config.h"

#include <spdlog/spdlog.h>

#include <opencv2/core/cuda.hpp>
#include <opencv2/cudafilters.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <algorithm>

void harris::getGradients(const cv::Mat& in, int kernelSize, cv::Mat& diffX, cv::Mat& diffY) {
    // Kernel size must be 1, 3, 5, or 7
    assert(kernelSize == 1 || kernelSize == 3 || kernelSize == 5 || kernelSize == 7);
    assert(in.type() == CV_32F);

    // Set up async stream
    cv::cuda::Stream stream;
    stream.enqueueHostCallback(common::checkCopySuccess, nullptr);
    // Use a sobel operator to get the gradient, since it combines the gaussian with the
    // derivative
    auto sobelX = cv::cuda::createSobelFilter(in.type(), in.type(), 1, 0, kernelSize);
    auto sobelY = cv::cuda::createSobelFilter(in.type(), in.type(), 0, 1, kernelSize);

    diffX.create(in.rows, in.cols, in.type());
    diffY.create(in.rows, in.cols, in.type());

    // Copy to GPU memory
    cv::cuda::GpuMat d_in, d_outX, d_outY;
    d_in.upload(in, stream);
    d_outX.upload(diffX, stream);
    d_outY.upload(diffY, stream);

    // Apply sobel operator in X and Y direction
    sobelX->apply(d_in, d_outX, stream);
    sobelY->apply(d_in, d_outY, stream);
    d_outX.download(diffX, stream);
    d_outY.download(diffY, stream);
}

void harris::cpu::getCornerResponse(const cv::Mat& gradX,
                                    const cv::Mat& gradY,
                                    const size_t windowSize,
                                    const double gaussianSigma,
                                    const float harrisScore,
                                    cv::Mat& cornerResponse) {
    assert(gradX.rows == gradY.rows && gradX.cols == gradY.cols && gradX.type() == CV_32F &&
           gradX.type() == gradY.type());
    assert(windowSize % 2 == 1);

    auto logger = spdlog::get(config::STDOUT_LOGGER);
    auto flogger = spdlog::get(config::FILE_LOGGER);

    cornerResponse = cv::Mat::zeros(gradX.rows, gradX.cols, CV_32F);
    flogger->info("Resized cornerResponse to match gradient sizes: {} rows x {} cols",
                  cornerResponse.rows,
                  cornerResponse.cols);
    // Get a 1D Gaussian kernel with a given size and sigma
    cv::Mat gauss = cv::getGaussianKernel(windowSize, gaussianSigma, gradX.type());
    // Outer product for a 2D matrix
    gauss = gauss * gauss.t();
    // Iterate over each pixel in the image and compute the second moment matrix for each pixel,
    // where the weights are the Gaussian kernel
    int windowRad = windowSize / 2;
    for (int y = 0; y < gradX.rows; y++) {
        for (int x = 0; x < gradX.cols; x++) {
            cv::Mat secondMoment = cv::Mat::zeros(2, 2, CV_32F);
            for (int wy = -windowRad; wy <= windowRad; wy++) {
                for (int wx = -windowRad; wx <= windowRad; wx++) {
                    // Get the gradient values
                    float gradXVal = gradX.at<float>(std::min(std::max(0, y + wy), gradX.rows - 1),
                                                     std::min(std::max(0, x + wx), gradX.cols - 1));
                    float gradYVal = gradY.at<float>(std::min(std::max(0, y + wy), gradY.rows - 1),
                                                     std::min(std::max(0, x + wx), gradY.cols - 1));

                    float weight = gauss.at<float>(wy + windowRad, wx + windowRad);

                    // Build up gradient matrix
                    cv::Mat gradVals = (cv::Mat_<float>(2, 2) << gradXVal * gradXVal,
                                        gradXVal * gradYVal,
                                        gradXVal * gradYVal,
                                        gradYVal * gradYVal);

                    // Add to second moment matrix sum
                    secondMoment = secondMoment + weight * gradVals;
                }
            }
            // Compute the corner response value, R
            float trace = (cv::trace(secondMoment))[0];
            float R = cv::determinant(secondMoment) - harrisScore * trace * trace;

            cornerResponse.at<float>(y, x) = R;
        }
    }
}

void harris::cpu::refineCorners(const cv::Mat& cornerResponse,
                                const double threshold,
                                const int minDistance,
                                cv::Mat& corners) {
    // Only support CV_32F right now
    assert(cornerResponse.type() == CV_32F);

    auto logger = spdlog::get(config::STDOUT_LOGGER);
    auto flogger = spdlog::get(config::FILE_LOGGER);

    corners = cv::Mat::zeros(cornerResponse.rows, cornerResponse.cols, cornerResponse.type());
    flogger->info(
        "Resized corners to match input size: {} rows x {} cols", corners.rows, corners.cols);

    // Iterate over the matrix. For each pixel, check if it's above the threshold value; if it is,
    // check the minDistance pixels around it to verify that there is nothing higher than it.
    for (int y = 0; y < cornerResponse.rows; y++) {
        for (int x = 0; x < cornerResponse.cols; x++) {
            float crVal = cornerResponse.at<float>(y, x);
            if (crVal >= threshold) {
                // Iterate around the surrounding pixels to see if it's a local maxima
                bool isLocalMax = true;
                for (int wy = -minDistance; wy <= minDistance; wy++) {
                    for (int wx = -minDistance; wx <= minDistance; wx++) {
                        // Skip if the comparison point is this point
                        int compY = std::min(std::max(0, y + wy), cornerResponse.rows - 1);
                        int compX = std::min(std::max(0, x + wx), cornerResponse.cols - 1);
                        if (y == compY && x == compX) continue;

                        if (crVal <= cornerResponse.at<float>(compY, compX)) {
                            isLocalMax = false;
                            break;
                        }
                    }
                    if (!isLocalMax) break;
                }
                if (isLocalMax) {
                    corners.at<float>(y, x) = crVal;
                    // This is a local maxima, so we can skip ahead in the row search
                    x += (minDistance - 1);
                }
            }
        }
    }
}