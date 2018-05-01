#include "../include/Descriptors.h"

#include <cmath>

static constexpr float PI = 3.1415921636;

void sift::getAnglesFromGradients(const cv::Mat& gradX, const cv::Mat& gradY, cv::Mat& angles) {
    // Make sure our input images are the right size, then resize the output to the correct size
    assert(gradX.rows == gradY.rows && gradX.cols == gradY.cols && gradX.type() == gradY.type() &&
           gradX.type() == CV_32F);

    // Each angle will be represented in degrees
    angles.create(gradX.rows, gradX.cols, CV_32F);

    // Iterate over the input matrices and compute the angle
    float Ix, Iy;
    for (int y = 0; y < gradX.rows; y++) {
        for (int x = 0; x < gradX.cols; x++) {
            Ix = gradX.at<float>(y, x);
            Iy = gradY.at<float>(y, x);

            angles.at<float>(y, x) = std::atan2(Iy, Ix);
        }
    }
}

void sift::getKeypoints(const cv::Mat& gradX,
                        const cv::Mat& gradY,
                        const std::vector<std::pair<int, int>>& cornerLocs,
                        const size_t size,
                        std::vector<cv::KeyPoint>& keypoints) {
    // Make sure our input images are the right size, then resize the output to the correct size
    assert(gradX.rows == gradY.rows && gradX.cols == gradY.cols && gradX.type() == gradY.type() &&
           gradX.type() == CV_32F);

    keypoints.clear();

    // Iterate over corners and create keypoints
    for (const auto& corner : cornerLocs) {
        float Ix = gradX.at<float>(corner.first, corner.second);
        float Iy = gradY.at<float>(corner.first, corner.second);

        float angle = std::atan2(Iy, Ix) * 180.f / PI;

        keypoints.emplace_back(corner.second, corner.first, size, angle, 0);
    }
}