#pragma once

#include <opencv2/core/core.hpp>

namespace lk {
void calcOpticalFlow(const cv::Mat& prevImg,
                     const cv::Mat& nextImg,
                     cv::Mat& u,
                     cv::Mat& v,
                     const size_t winSize = 21);

void warp(const cv::Mat& src, const cv::Mat& du, const cv::Mat& dv, cv::Mat& dst);

void calcOpticalFlowPyr(const cv::Mat& prevImg,
                        const cv::Mat& nextImg,
                        cv::Mat& u,
                        cv::Mat& v,
                        const size_t winSize = 21);
} // namespace lk
