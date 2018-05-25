#include "../include/Moments.h"

#include <cmath>
#include <iostream>

std::vector<std::pair<float, float>>
    moments::centralMoment(const cv::Mat& img,
                           const std::vector<std::pair<int, int>>& momentOrders) {
    assert(img.channels() == 1);
    // Compute the image moments of the input image
    cv::Mat x_i, y_j;
    cv::Mat fltImg;
    img.convertTo(fltImg, CV_32FC1);

    size_t numRows = fltImg.rows;
    size_t numCols = fltImg.cols;
    std::vector<float> rows(numRows);
    std::vector<float> cols(numCols);
    std::iota(rows.begin(), rows.end(), 0);
    std::iota(cols.begin(), cols.end(), 0);

    x_i = cv::Mat(cols);     // Nx1 matrix
    cv::transpose(x_i, x_i); // 1xN matrix
    y_j = cv::Mat(rows);     // Mx1 matrix

    // Build up x and y matrices so that we can multiply them
    cv::Mat xFull = x_i.clone();
    cv::Mat yFull = y_j.clone();

    for (int r = 1; r < numRows; r++) {
        xFull.push_back(x_i);
    }
    for (int c = 1; c < numCols; c++) {
        // yFull.push_back(y_j);
        cv::hconcat(yFull, y_j, yFull);
    }

    // std::cout << "xFull is " << xFull.rows << "x" << xFull.cols << "; yFull is " << yFull.rows
    //           << "x" << yFull.cols << "; fltImg is " << fltImg.rows << "x" << fltImg.cols
    //           << std::endl;

    // Calculate image moments
    float M00 = cv::sum(fltImg)[0];
    float M01 = cv::sum(yFull.mul(fltImg))[0];
    float M10 = cv::sum(xFull.mul(fltImg))[0];

    float xBar = M10 / M00;
    float yBar = M01 / M00;

    // Iterate over desired moment orders and compute central moments
    std::vector<std::pair<float, float>> centralMoments;
    for (const auto& order : momentOrders) {
        int p = order.first;
        int q = order.second;

        cv::Mat xPow, yPow;
        cv::pow(xFull - xBar, p, xPow);
        cv::pow(xFull - yBar, q, yPow);

        float mu = cv::sum(yPow.mul(xPow.mul(fltImg)))[0];
        float eta = mu / (std::pow(M00, 1.0 + float(p + q) / 2.0));
        // float eta = std::pow(M00, 1.0 + float(p + q) / 2.0);
        centralMoments.emplace_back(mu, eta);
    }

    return centralMoments;
}