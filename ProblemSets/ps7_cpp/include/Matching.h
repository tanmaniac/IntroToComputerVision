#pragma once

#include <opencv2/core/core.hpp>

namespace matching {
// Computes confusion matrix by training KNN classifier from the given features and labels, removing
// one pair each iteration as a cross-validation sample.
void naiveConfusionMatrix(const cv::Mat& features, const cv::Mat& labels, cv::Mat& confusion);

// More "correct" confusion matrix. Builds a training sets by removing each person from the feature
// data and using that for cross validation. Returns a vector of N mats, where mats 0 through N-1
// are the confusion matrices for persons 1 through N; mat N is the average of mats 0 through N-1.
void confusionMatrix(const cv::Mat& features,
                     const cv::Mat& labels,
                     const cv::Mat& people,
                     const size_t numPeople,
                     std::vector<cv::Mat>& confusions);

// Plots confusion matrix. fileName is an optional parameter; if it is set, then the plot will be
// saved to that file.
void plotConfusionMatrix(const cv::Mat& confusion,
                         const std::string& title,
                         const std::string& fileName = "");
} // namespace matching