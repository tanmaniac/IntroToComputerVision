#include "Solution.h"
#include "../include/OpticalFlow.h"
#include "../include/Pyramids.h"

#include <spdlog/spdlog.h>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

enum class LKMode { NAIVE, HEIRARCHICAL };

void drawVelocityVectors(cv::Mat& inputImg,
                         const cv::Mat& u,
                         const cv::Mat& v,
                         const cv::Scalar& color) {
    assert(u.size() == v.size() && u.size() == inputImg.size());

    // If the input image is single-channel, convert it to RGB first
    if (inputImg.channels() < 3) {
        cv::cvtColor(inputImg, inputImg, cv::COLOR_GRAY2RGB);
    }

    // Number of arrows per row and column
    static constexpr size_t ARROWS_PER_RC = 30;
    const size_t rowStride = u.rows / ARROWS_PER_RC;
    const size_t colStride = u.cols / ARROWS_PER_RC;

    // Iterate over input image and add arrows
    for (int y = 0; y < u.rows; y += rowStride) {
        for (int x = 0; x < u.cols; x += colStride) {
            float uVal = u.at<float>(y, x);
            float vVal = v.at<float>(y, x);
            cv::arrowedLine(inputImg, cv::Point2f(x, y), cv::Point2f(x + uVal, y + vVal), color);
        }
    }
}

// Wraps the parameters required for optical flow calculation for this assignment
std::pair<cv::Mat, cv::Mat> denseLKWrapper(const cv::Mat& prevImg,
                                           const cv::Mat& nextImg,
                                           const LKMode mode,
                                           const size_t windowSize,
                                           const std::string& filePrefix,
                                           const std::string& outputImg,
                                           bool saveColorMaps = true) {
    // Convert input images to grey
    cv::Mat prevGrey, nextGrey;
    if (prevImg.channels() >= 3)
        cv::cvtColor(prevImg, prevGrey, cv::COLOR_RGB2GRAY);
    else
        prevGrey = prevImg;

    if (nextImg.channels() >= 3)
        cv::cvtColor(nextImg, nextGrey, cv::COLOR_RGB2GRAY);
    else
        nextGrey = nextImg;

    cv::Mat u, v;
    if (mode == LKMode::NAIVE) {
        lk::calcOpticalFlow(prevGrey, nextGrey, u, v, windowSize);
    } else {
        lk::calcOpticalFlowPyr(prevImg, nextImg, u, v, windowSize);
    }

    cv::Mat velocityVectors = prevImg.clone();
    drawVelocityVectors(velocityVectors, u, v, cv::Scalar(0, 255, 0, 255));

    cv::imwrite(filePrefix + "/" + outputImg + ".png", velocityVectors);

    // Draw color maps of u and v movement
    if (saveColorMaps) {
        cv::Mat uNorm, vNorm;
        cv::normalize(u, uNorm, 0, 255, cv::NORM_MINMAX, CV_8U);
        cv::normalize(v, vNorm, 0, 255, cv::NORM_MINMAX, CV_8U);
        cv::applyColorMap(uNorm, uNorm, cv::COLORMAP_JET);
        cv::applyColorMap(vNorm, vNorm, cv::COLORMAP_JET);

        cv::imwrite(filePrefix + "/" + outputImg + "-uColorMap.png", uNorm);
        cv::imwrite(filePrefix + "/" + outputImg + "-vColorMap.png", vNorm);
    }

    return std::make_pair(u.clone(), v.clone());
}

void savePyramid(const std::vector<cv::Mat>& pyramid, const std::string& filename) {
    // Scale pyramid images back up and save them
    cv::Mat half, quarter, eighth;
    cv::resize(pyramid[1], half, pyramid[0].size(), 2, 2, cv::INTER_NEAREST);
    cv::resize(pyramid[2], quarter, pyramid[0].size(), 4, 4, cv::INTER_NEAREST);
    cv::resize(pyramid[3], eighth, pyramid[0].size(), 8, 8, cv::INTER_NEAREST);

    cv::Mat origAndHalf, quarterAndEighth, all;
    cv::hconcat(pyramid[0], half, origAndHalf);
    cv::hconcat(quarter, eighth, quarterAndEighth);
    cv::vconcat(origAndHalf, quarterAndEighth, all);

    cv::imwrite(filename, all);
}

void warpHelper(const std::vector<std::vector<cv::Mat>>& pyramids,
                const int pyrLevel,
                const size_t winSize,
                const std::string& filePrefix,
                const std::string& filename) {
    // iterate over all the images in the dataset
    for (int imIdx = 1; imIdx < pyramids.size(); imIdx++) {
        // Compute u and v displacement
        auto displacement = denseLKWrapper(pyramids[imIdx - 1][pyrLevel],
                                           pyramids[imIdx][pyrLevel],
                                           LKMode::NAIVE,
                                           winSize,
                                           filePrefix,
                                           filename + "-" + std::to_string(imIdx));
        cv::Mat& du = displacement.first;
        cv::Mat& dv = displacement.second;

        // Warp back to the previous image
        cv::Mat warped;
        lk::warp(pyramids[imIdx][pyrLevel], du, dv, warped);

        // Save the differences between the warped image and the previous image
        cv::Mat diff = pyramids[imIdx - 1][pyrLevel] - warped;
        cv::normalize(diff, diff, 0, 255, cv::NORM_MINMAX, CV_8U);
        cv::imwrite(filePrefix + "/" + filename + "-" + std::to_string(imIdx) + "-warped-diff.png",
                    diff);
    }
}

void sol::runProblem1(const Config& config) {
    // Time runtime
    auto logger = spdlog::get(config::STDOUT_LOGGER);
    logger->info("Problem 1 begins");
    auto start = std::chrono::high_resolution_clock::now();

    // Problem 1 a. Index 2 in the _shift image set is ShiftR2 and index 5 is ShiftR5U5
    const size_t winSize = config._lkWinSize1;
    denseLKWrapper(config._shift[0],
                   config._shift[2],
                   LKMode::NAIVE,
                   winSize,
                   config._outputPathPrefix,
                   "ps5-1-a-1");
    denseLKWrapper(config._shift[0],
                   config._shift[5],
                   LKMode::NAIVE,
                   winSize,
                   config._outputPathPrefix,
                   "ps5-1-a-2");

    // Problem 2 a. Index 1 is ShiftR10, 3 is R20, 4 is R40
    denseLKWrapper(config._shift[0],
                   config._shift[1],
                   LKMode::NAIVE,
                   winSize,
                   config._outputPathPrefix,
                   "ps5-1-b-1");
    denseLKWrapper(config._shift[0],
                   config._shift[3],
                   LKMode::NAIVE,
                   winSize,
                   config._outputPathPrefix,
                   "ps5-1-b-2");
    denseLKWrapper(config._shift[0],
                   config._shift[4],
                   LKMode::NAIVE,
                   winSize,
                   config._outputPathPrefix,
                   "ps5-1-b-3");

    auto finish = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> runtime = finish - start;
    logger->info("Problem 1 runtime = {} ms", runtime.count());
}

void sol::runProblem2(const Config& config) {
    // Time runtime
    auto logger = spdlog::get(config::STDOUT_LOGGER);
    logger->info("Problem 2 begins");
    auto start = std::chrono::high_resolution_clock::now();

    const size_t pyrDepth = 4;
    std::vector<cv::Mat> gaussianPyramid = pyr::makeGaussianPyramid(config._yosemite[0], pyrDepth);

    savePyramid(gaussianPyramid, config._outputPathPrefix + "/ps5-2-a-1.png");

    // Build Laplacian pyramid
    std::vector<cv::Mat> laplacianPyramid;
    for (int i = 0; i < pyrDepth - 1; i++) {
        cv::Mat next;
        pyr::pyrUp(gaussianPyramid[i + 1], next);
        // Sometimes the expanded image is 1 row/col smaller than the one in the gaussian pyramid,
        // so resize if necessary
        if (next.rows < gaussianPyramid[i].rows || next.cols < gaussianPyramid[i].cols) {
            cv::resize(next, next, gaussianPyramid[i].size());
        }
        laplacianPyramid.push_back(gaussianPyramid[i] - next);
    }
    // Copy the "base case" from the gaussian pyramid, which is the smallest one
    laplacianPyramid.push_back(gaussianPyramid[pyrDepth - 1]);

    savePyramid(laplacianPyramid, config._outputPathPrefix + "/ps5-2-b-1.png");

    auto finish = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> runtime = finish - start;
    logger->info("Problem 2 runtime = {} ms", runtime.count());
}

void sol::runProblem3(const Config& config) {
    // Time runtime
    auto logger = spdlog::get(config::STDOUT_LOGGER);
    logger->info("Problem 3 begins");
    auto start = std::chrono::high_resolution_clock::now();

    std::vector<std::vector<cv::Mat>> yosemitePyramids;
    std::vector<std::vector<cv::Mat>> pupperPyramids;

    // Iterate over all the images in the Yosemite set and build Gaussian pyramids
    const int pyrDepth = 4;
    for (const auto& img : config._yosemite) {
        yosemitePyramids.push_back(pyr::makeGaussianPyramid(img, pyrDepth));
    }

    // Do the same for the dataset with the dog
    for (const auto& img : config._pupper) {
        pupperPyramids.push_back(pyr::makeGaussianPyramid(img, pyrDepth));
    }

    // Compute the optical flow from the first image to the second image of the Yosemite set
    // Use g1, the second image in the Gaussian pyramid, for this
    warpHelper(yosemitePyramids,
               config._pyrLevel3a,
               config._lkWinSize3,
               config._outputPathPrefix,
               "ps5-3-a-1");
    // Use g2 for the dog dataset
    warpHelper(pupperPyramids,
               config._pyrLevel3b,
               config._lkWinSize3,
               config._outputPathPrefix,
               "ps5-3-a-2");

    auto finish = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> runtime = finish - start;
    logger->info("Problem 3 runtime = {} ms", runtime.count());
}

void sol::runProblem4(const Config& config) {
    // Time runtime
    auto logger = spdlog::get(config::STDOUT_LOGGER);
    logger->info("Problem 4 begins");
    auto start = std::chrono::high_resolution_clock::now();

    size_t winSize = config._lkWinSize4;
    // Yosemite data set
    denseLKWrapper(config._yosemite[0],
                   config._yosemite[1],
                   LKMode::HEIRARCHICAL,
                   winSize,
                   config._outputPathPrefix,
                   "ps5-4-a-1",
                   true);
    denseLKWrapper(config._yosemite[1],
                   config._yosemite[2],
                   LKMode::HEIRARCHICAL,
                   winSize,
                   config._outputPathPrefix,
                   "ps5-4-a-2",
                   true);

    // Dog data set
    denseLKWrapper(config._pupper[0],
                   config._pupper[1],
                   LKMode::HEIRARCHICAL,
                   winSize,
                   config._outputPathPrefix,
                   "ps5-4-b-1",
                   true);
    denseLKWrapper(config._pupper[1],
                   config._pupper[2],
                   LKMode::HEIRARCHICAL,
                   winSize,
                   config._outputPathPrefix,
                   "ps5-4-b-2",
                   true);

    auto finish = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> runtime = finish - start;
    logger->info("Problem 4 runtime = {} ms", runtime.count());
}