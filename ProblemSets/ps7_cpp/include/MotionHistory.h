#pragma once

#include <spdlog/spdlog.h>

#include <opencv2/core/core.hpp>

namespace mhi {
// Compute difference between two frames. Returns a binary image with a value of 1 where the
// difference between the two frames is greater than the given threshold, and 0 if not.
void frameDifference(const cv::Mat& f1,
                     const cv::Mat& f2,
                     const double thresh,
                     cv::Mat& diff,
                     const cv::Size& blurSize = cv::Size(3, 3),
                     const double blurSigma = 1.0);

// Build a motion history image from the current motion history image, tau, and a binary mask
// representing pixels that have moved in this frame (see frameDifference)
void calcMotionHistory(cv::Mat& history, const cv::Mat& binaryMask, const int tau);

// Compute motion energy images (MEIs) from a motion history image. This thresholds the MHIs such
// that any nonzero value is set to 1.
void energyFromHistory(const cv::Mat& mhi, cv::Mat& mei);

// This is an overloaded namespace function provided for convenience. It differs from the above
// function only in what arguments it accepts.
void energyFromHistory(const std::vector<cv::Mat>& mhis, std::vector<cv::Mat>& meis);
} // namespace mhi