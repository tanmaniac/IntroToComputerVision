#include "../include/DisparityNCorr.h"
#include "../include/Config.h"

#include <spdlog/spdlog.h>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <cmath>
#include <iostream>

void serial::disparityNCorr(const cv::Mat& left,
                            const cv::Mat& right,
                            const size_t windowRad,
                            const int minDisparity,
                            const int maxDisparity,
                            cv::Mat& disparity) {
    assert(left.type() == CV_32FC1 && right.type() == CV_32FC1);
    // Set up file loggers
    auto logger = spdlog::get(config::FILE_LOGGER);
    logger->info("DisparityNCorr: windowRad={}, minDisparity={}, maxDisparity={}",
                 windowRad,
                 minDisparity,
                 maxDisparity);
    // Pad input images with replicated borders so we can do the edge pixels
    logger->info("DisparityNCorr: padding input images with {} pixels", windowRad);
    cv::Mat leftPadded, rightPadded;
    cv::copyMakeBorder(
        left, leftPadded, windowRad, windowRad, windowRad, windowRad, cv::BORDER_REPLICATE);
    cv::copyMakeBorder(
        right, rightPadded, windowRad, windowRad, windowRad, windowRad, cv::BORDER_REPLICATE);

    logger->info("DisparityNCorr: original image: rows={} cols={}; new image: rows={} cols={}",
                 left.rows,
                 left.cols,
                 leftPadded.rows,
                 leftPadded.cols);

    disparity.create(left.rows, left.cols, CV_8SC1);

    size_t windowSize = 2 * windowRad + 1;

    // Iterate over the image
    for (int y = windowRad; y < leftPadded.rows - windowRad; y++) {
        for (int x = windowRad; x < leftPadded.cols - windowRad; x++) {
            // Get the template around this pixel that we'll use for the block matcher
            cv::Rect bbox(x - windowRad, y - windowRad, windowSize, windowSize);
            cv::Mat templ = leftPadded(bbox).clone();
            // Get the search window.
            int startX = std::max(0, x + minDisparity - int(windowRad));
            int endX = std::min(int(leftPadded.cols), x + maxDisparity + 1 + int(windowRad));
            bbox = cv::Rect(startX, y - windowRad, endX - startX, windowSize);
            cv::Mat search = rightPadded(bbox).clone();
            // Result of template matching is stored in a matrix of size (W - w + 1 * H - h + 1),
            // where W,H = width and height of the search image, w,h = width and height of the
            // template
            cv::Mat result(search.rows - windowSize + 1, search.cols - windowSize + 1, CV_32FC1);

            // Match templates
            cv::matchTemplate(search, templ, result, cv::TM_CCORR_NORMED);
            // Find maxima
            double minVal, maxVal;
            cv::Point minLoc, maxLoc;
            cv::minMaxLoc(result, &minVal, &maxVal, &minLoc, &maxLoc);

            // Get disparity value
            int disp = maxLoc.x - (minDisparity <= 0 && maxDisparity <= 0 ? result.cols - 1 : 0);
            disparity.at<char>(y - windowRad, x - windowRad) = disp;
        }
    }
}