#include "../include/DisparitySSD.h"

#include <spdlog/spdlog.h>

#include <opencv2/highgui/highgui.hpp>

#include <cmath>
#include <iostream>

void serial::disparitySSD(const cv::Mat& left,
                          const cv::Mat& right,
                          const ReferenceFrame frame,
                          const size_t windowRad,
                          const size_t minDisparity,
                          const size_t maxDisparity,
                          cv::Mat& disparity) {
    // Set up file loggers
    auto logger = spdlog::get("file_logger");
    logger->info("Padding input images with {} pixels", windowRad);
    cv::Mat leftPadded, rightPadded;
    // Hacky way to use right image as reference frame vs the left - just make the left padded image
    // out of the right
    cv::copyMakeBorder((frame == ReferenceFrame::LEFT ? left : right),
                       leftPadded,
                       windowRad,
                       windowRad,
                       windowRad,
                       windowRad,
                       cv::BORDER_REPLICATE);
    cv::copyMakeBorder((frame == ReferenceFrame::LEFT ? right : left),
                       rightPadded,
                       windowRad,
                       windowRad,
                       windowRad,
                       windowRad,
                       cv::BORDER_REPLICATE);
    // cv::imwrite("ps2_output/left1.png", leftPadded);
    // cv::imwrite("ps2_output/right1.png", rightPadded);
    logger->info("Original image: rows={} cols={}; new image: rows={} cols={}",
                 left.rows,
                 left.cols,
                 leftPadded.rows,
                 leftPadded.cols);

    disparity.create(cv::Size(left.rows, left.cols), CV_8SC1);

    // iterate over every pixel in the left image
    for (int y = windowRad; y < leftPadded.rows - windowRad; y++) {
        for (int x = windowRad; x < leftPadded.cols - windowRad; x++) {
            int bestCost = 99999999;
            int bestDisparity = 0;
            // Iterate over the row to search for a matching window. If the reference frame is the
            // left image, then we search to the left; if it's the right image, then we search to
            // the right
            int searchIndex = (frame == ReferenceFrame::LEFT) ? fmax(0, x - maxDisparity) : x;
            int maxSearchIndex = (frame == ReferenceFrame::LEFT)
                                     ? x - minDisparity
                                     : fmin(left.cols, x + maxDisparity);
            for (searchIndex; searchIndex <= maxSearchIndex; searchIndex++) {
                // Holy shit another nested loop
                int sum = 0;
                // Iterate over the window and compute sum of squared differences
                for (int winY = -windowRad; winY <= int(windowRad); winY++) {
                    for (int winX = -windowRad; winX <= int(windowRad); winX++) {
                        float rawCost = leftPadded.at<float>(y + winY, x + winX) -
                                        rightPadded.at<float>(y + winY, searchIndex + winX);
                        // logger->info("rawCost = {}", rawCost);
                        sum += round(rawCost * rawCost);
                    }
                }
                if (sum < bestCost) {
                    bestCost = sum;
                    bestDisparity = searchIndex - x;
                }
            }
            if (bestDisparity != 0) {
                logger->info("Found disparity {} at ({}, {}) with cost {}",
                             bestDisparity,
                             y - windowRad,
                             x - windowRad,
                             bestCost);
            }
            disparity.at<unsigned char>(y - windowRad, x - windowRad) = bestDisparity;
        }
    }
}