#include "../include/DisparitySSD.h"

#include <spdlog/spdlog.h>

#include <cmath>
#include <iostream>

void serial::disparitySSD(const cv::Mat& left,
                          const cv::Mat& right,
                          const size_t windowRad,
                          const int minDisparity,
                          const int maxDisparity,
                          cv::Mat& disparity) {
    assert(left.type() == CV_32FC1 && right.type() == CV_32FC1);
    // Set up file loggers
    auto logger = spdlog::get("file_logger");
    logger->info("disparitySSD (CPU): Padding input images with {} pixels", windowRad);
    cv::Mat leftPadded, rightPadded;
    cv::copyMakeBorder(
        left, leftPadded, windowRad, windowRad, windowRad, windowRad, cv::BORDER_REPLICATE);
    cv::copyMakeBorder(
        right, rightPadded, windowRad, windowRad, windowRad, windowRad, cv::BORDER_REPLICATE);

    logger->info(
        "disparitySSD (CPU): Original image: rows={} cols={}; padded image: rows={} cols={}",
        left.rows,
        left.cols,
        leftPadded.rows,
        leftPadded.cols);

    disparity.create(left.rows, left.cols, CV_8SC1);

    // iterate over every pixel in the left image
    for (int y = windowRad; y < leftPadded.rows - windowRad; y++) {
        for (int x = windowRad; x < leftPadded.cols - windowRad; x++) {
            int bestCost = 99999999;
            int bestDisparity = 0;
            // Iterate over the row to search for a matching window. If the reference frame is the
            // left image, then we search to the left; if it's the right image, then we search to
            // the right
            int searchIndex = fmax(0, x + minDisparity);
            int maxSearchIndex = fmin(leftPadded.cols - 1, x + maxDisparity);
            for (searchIndex; searchIndex <= maxSearchIndex; searchIndex++) {
                int sum = 0;
                // Iterate over the window and compute sum of squared differences
                for (int winY = -1 * int(windowRad); winY <= int(windowRad); winY++) {
                    for (int winX = -1 * int(windowRad); winX <= int(windowRad); winX++) {
                        float rawCost = leftPadded.at<float>(y + winY, x + winX) -
                                        rightPadded.at<float>(y + winY, searchIndex + winX);
                        sum += round(rawCost * rawCost);
                    }
                }
                if (sum < bestCost) {
                    bestCost = sum;
                    bestDisparity = searchIndex - x;
                }
            }
            disparity.at<char>(y - windowRad, x - windowRad) = bestDisparity;
        }
    }
}