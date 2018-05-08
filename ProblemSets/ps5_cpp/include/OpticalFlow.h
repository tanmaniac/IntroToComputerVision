#pragma once

#include <opencv2/core/core.hpp>

namespace lk {
void calcOpticalFlow(const cv::Mat& prevImg,
                     const cv::Mat& nextImg,
                     cv::Mat& u,
                     cv::Mat& v,
                     const size_t winSize = 9);
}
