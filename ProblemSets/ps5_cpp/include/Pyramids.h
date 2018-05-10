#pragma once

#include <opencv2/core/core.hpp>

#include <vector>

namespace pyr {
void pyrDown(const cv::Mat& src, cv::Mat& dst);
void pyrUp(const cv::Mat& src, cv::Mat& dst);

std::vector<cv::Mat> makeGaussianPyramid(const cv::Mat& src, const size_t levels);
} // namespace pyr