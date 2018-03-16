#include "Solution.h"
#include "Convolution.h"
#include "Hough.h"

#include <yaml-cpp/yaml.h>

#include <opencv2/core/cuda.hpp>
#include <opencv2/cudafilters.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <cmath>
#include <iostream>

static constexpr float PI = 3.14159265;

// Finds edges using a Canny edge detector.
// TODO: Don't do gaussian blurring within this function, just expect a pre-blurred image @tanmaniac
void sol::generateEdge(const cv::Mat& input, const Config::EdgeDetect& config, cv::Mat& output) {
    cv::cuda::GpuMat d_input(input);
    cv::cuda::GpuMat d_blurredFlt(input.size(), input.type());

    // Run Gaussian blur
    cv::Ptr<cv::cuda::Filter> gaussian =
        cv::cuda::createGaussianFilter(d_input.type(),
                                       d_blurredFlt.type(),
                                       cv::Size(config._gaussianSize, config._gaussianSize),
                                       config._gaussianSigma);

    gaussian->apply(d_input, d_blurredFlt);

    // Compute edges
    // First convert to uchar8 matrix
    cv::cuda::GpuMat d_blurred(input.size(), CV_8UC1);
    d_blurredFlt.convertTo(d_blurred, CV_8UC1);
    cv::Ptr<cv::cuda::CannyEdgeDetector> canny = cv::cuda::createCannyEdgeDetector(
        config._lowerThreshold, config._upperThreshold, config._sobelApertureSize);

    cv::cuda::GpuMat d_output(input.size(), CV_8UC1);
    canny->detect(d_blurred, d_output);

    // Copy result back to output
    d_output.download(output);
    // d_blurred.download(output);
}

void sol::gaussianBlur(const cv::Mat& input, const Config::EdgeDetect& config, cv::Mat& output) {
    /*cv::Mat colGaussian =
        cv::getGaussianKernel(config._gaussianSize, config._gaussianSigma, CV_32F);
    cv::Mat rowGaussian;
    cv::transpose(colGaussian, rowGaussian);
    // Running CUDA convolution
    separableConvolution(input, rowGaussian, colGaussian, output);*/
    cv::cuda::GpuMat d_input(input);
    cv::cuda::GpuMat d_blurred(input.size(), input.type());
    cv::Ptr<cv::cuda::Filter> gaussian =
        cv::cuda::createGaussianFilter(d_input.type(),
                                       d_blurred.type(),
                                       cv::Size(config._gaussianSize, config._gaussianSize),
                                       config._gaussianSigma);
    gaussian->apply(d_input, d_blurred);

    // Copy back to output
    d_blurred.download(output);
}

void sol::houghLinesAccumulate(const cv::Mat& edgeMask,
                               const Config::Hough& config,
                               cv::Mat& accumulator) {
    cuda::houghLinesAccumulate(edgeMask, config._rhoBinSize, config._thetaBinSize, accumulator);
}

void sol::houghCirclesAccumulate(const cv::Mat& edgeMask,
                                 const size_t radius,
                                 cv::Mat& accumulator) {
    cuda::houghCirclesAccumulate(edgeMask, radius, accumulator);
}

void sol::findLocalMaxima(const cv::Mat& accumulator,
                          const Config::Hough& config,
                          std::vector<std::pair<unsigned int, unsigned int>>& localMaxima) {
    cuda::findLocalMaxima(accumulator, config._numPeaks, config._threshold, localMaxima);
}

std::pair<int, int> sol::rowColToRhoTheta(const std::pair<unsigned int, unsigned int>& coordinates,
                                          const cv::Mat& inputImage,
                                          const Config::Hough& config) {
    const size_t diagDist =
        ceil(sqrt(inputImage.rows * inputImage.rows + inputImage.cols * inputImage.cols));
    int rho = coordinates.first * config._rhoBinSize - diagDist;
    int theta = coordinates.second * config._thetaBinSize + MIN_THETA;
    return std::make_pair(rho, theta);
}

void sol::drawLineParametric(cv::Mat& image,
                             const float rho,
                             const float theta,
                             const cv::Scalar color) {
    float thetaRad = theta * PI / 180.f;
    cv::Point2f start, end;
    // Make sure line isn't vertical
    if (thetaRad != 0) {
        start.x = 0;
        end.x = image.cols;

        float slope = -1.f * cos(thetaRad) / sin(thetaRad);
        float c = rho / sin(thetaRad);

        start.y = slope * start.x + c;
        end.y = slope * end.x + c;
    } else {
        // Line is vertical
        start.y = 0;
        end.y = image.rows;
        start.x = end.x = rho / cos(thetaRad);
    }

    cv::line(image, start, end, color, 1);
}
