#pragma once

#include <opencv2/core/core.hpp>

#include <utility>
#include <vector>

namespace moments {
// Compute the central moments of an image given the input image and a vector of pairs describing
// the desired orders of the central moments <p, q>
std::vector<std::pair<float, float>>
    centralMoment(const cv::Mat& img, const std::vector<std::pair<int, int>>& momentOrders);
} // namespace moments