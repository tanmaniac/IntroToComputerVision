#include "Solution.h"
#include "Hough.h"

#include <yaml-cpp/yaml.h>

#include <opencv2/core/cuda.hpp>
#include <opencv2/cudafilters.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <cmath>
#include <cstdint>
#include <iostream>
#include <unordered_map>

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
                               const Config::HoughLines& config,
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
                                          const Config::HoughLines& config) {
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

void sol::drawLinesParametric(cv::Mat& image,
                              const std::vector<std::pair<int, int>>& rhoTheta,
                              const cv::Scalar color) {
    for (const auto& val : rhoTheta) {
        drawLineParametric(image, val.first, val.second, color);
    }
}

void sol::drawCircles(cv::Mat& image,
                      const std::vector<std::pair<unsigned int, unsigned int>>& centers,
                      const size_t radius,
                      const cv::Scalar color) {
    for (const auto& center : centers) {
        cv::circle(image, cv::Point(center.second, center.first), radius, color);
    }
}

void sol::findParallelLines(const std::vector<std::pair<uint32_t, uint32_t>>& rhoTheta,
                            const size_t deltaTheta,
                            const size_t deltaRho,
                            std::vector<std::pair<uint32_t, uint32_t>>& parallelRhoThetas) {
    // Key = combination of rho and theta bin; Value = index of rhoTheta pair
    typedef std::unordered_multimap<uint64_t, size_t> MapType;
    MapType lines;
    parallelRhoThetas.clear();

    // Bin each line by rho and theta values and place them in a map
    for (size_t idx = 0; idx < rhoTheta.size(); idx++) {
        auto line = rhoTheta[idx];
        uint32_t rhoBin = line.first / deltaRho * deltaRho;
        uint32_t thetaBin = line.second / deltaTheta * deltaTheta;

        uint64_t key = 0;
        key = ((key | rhoBin) << 32) | thetaBin;

        lines.insert(std::make_pair(key, idx));
    }

    // Iterate over the map
    for (auto iter = lines.begin(); iter != lines.end();) {
        auto key = iter->first;
        // See if this key has more than one value in its bucket
        if (lines.count(key) > 1) {
            auto range = lines.equal_range(key);
            // Push each resulting pair onto the output vector
            std::for_each(range.first,
                          range.second,
                          [&parallelRhoThetas, &rhoTheta](MapType::value_type& value) {
                              parallelRhoThetas.push_back(rhoTheta[value.second]);
                          });
        }

        // Advance to next unique key
        do {
            ++iter;
        } while (iter != lines.end() && key == iter->first);
    }
}