#include "Solution.h"
#include "../include/OpticalFlow.h"

#include <spdlog/spdlog.h>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

void sol::runProblem1(const Config& config) {
    std::vector<cv::Mat> imgsGrey;
    for (const auto& img : config._shift) {
        cv::Mat grey;
        cv::cvtColor(img, grey, cv::COLOR_RGB2GRAY);
        imgsGrey.push_back(grey);
    }

    cv::Mat u, v;
    lk::calcOpticalFlow(imgsGrey[0], imgsGrey[2], u, v);
    cv::Mat velocityVectors = cv::Mat::zeros(u.size(), CV_8U);
    // Draw lil lines
    for (int y = 0; y < u.rows; y += 10) {
        for (int x = 0; x < u.cols; x += 10) {
            float uVal = u.at<float>(y, x);
            float vVal = v.at<float>(y, x);
            cv::arrowedLine(velocityVectors,
                            cv::Point2f(x, y),
                            cv::Point2f(x + uVal, y + vVal),
                            cv::Scalar(255, 255, 255, 255));
        }
    }

    cv::imwrite("wow.png", velocityVectors);

    cv::Mat uNorm, vNorm;
    cv::normalize(u, uNorm, 0, 255, cv::NORM_MINMAX, CV_8U);
    cv::normalize(v, vNorm, 0, 255, cv::NORM_MINMAX, CV_8U);

    cv::imwrite("u.png", uNorm);
    cv::imwrite("v.png", vNorm);
}