#include "../include/Pyramids.h"

#include <opencv2/imgproc/imgproc.hpp>

std::vector<cv::Mat> pyr::makeGaussianPyramid(const cv::Mat& src, const size_t levels) {
    std::vector<cv::Mat> pyramid;

    // PyrUp/down only supports CV_32F since I'm too lazy to support anything else
    cv::Mat grey = src.clone();
    if (grey.channels() > 1) {
        cv::cvtColor(grey, grey, cv::COLOR_RGB2GRAY);
    }
    if (grey.type() != CV_32F) {
        grey.convertTo(grey, CV_32F);
    }

    // Build Gaussian pyramid. The first image in the pyramid is the original image
    pyramid.push_back(grey);
    for (int i = 1; i < levels; i++) {
        cv::Mat down;
        pyr::pyrDown(pyramid[i - 1], down);
        pyramid.push_back(down);
    }

    return pyramid;
}