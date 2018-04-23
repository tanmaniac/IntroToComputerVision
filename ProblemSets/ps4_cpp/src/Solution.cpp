#include "Solution.h"
#include "../include/Harris.h"

#include <spdlog/spdlog.h>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

void solution::runProblem1(const Config& config) {
    // Time runtime
    auto logger = spdlog::get(config::STDOUT_LOGGER);
    logger->info("Problem 1 begins");
    auto start = std::chrono::high_resolution_clock::now();

    // For the transA image
    {
        cv::Mat gradientX, gradientY;
        harris::getGradients(
            config._images._transA, config._harrisTrans._sobelKernelSize, gradientX, gradientY);
        // Concatenate X and Y gradients for output
        cv::Mat gradCombined(gradientX.rows + gradientY.rows, gradientX.cols, CV_8UC1),
            gradXNorm(gradientX.rows, gradientX.cols, CV_8UC1),
            gradYNorm(gradientY.rows, gradientY.cols, CV_8UC1);
        cv::normalize(gradientX, gradXNorm, 0, 255, cv::NORM_MINMAX);
        cv::normalize(gradientY, gradYNorm, 0, 255, cv::NORM_MINMAX);
        cv::hconcat(gradXNorm, gradYNorm, gradCombined);
        cv::imwrite(config._outputPathPrefix + "/ps4-1-a-1.png", gradCombined);
    }
    // For the simA image
    {
        cv::Mat gradientX, gradientY;
        harris::getGradients(
            config._images._simA, config._harrisSim._sobelKernelSize, gradientX, gradientY);
        // Concatenate X and Y gradients for output
        cv::Mat gradCombined(gradientX.rows + gradientY.rows, gradientX.cols, CV_8UC1),
            gradXNorm(gradientX.rows, gradientX.cols, CV_8UC1),
            gradYNorm(gradientY.rows, gradientY.cols, CV_8UC1);
        cv::normalize(gradientX, gradXNorm, 0, 255, cv::NORM_MINMAX);
        cv::normalize(gradientY, gradYNorm, 0, 255, cv::NORM_MINMAX);
        cv::hconcat(gradXNorm, gradYNorm, gradCombined);
        cv::imwrite(config._outputPathPrefix + "/ps4-1-a-2.png", gradCombined);
    }

    auto finish = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> runtime = finish - start;
    logger->info("Problem 1a runtime = {} ms", runtime.count());
}