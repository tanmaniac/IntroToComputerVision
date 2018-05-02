#include "Solution.h"
#include "../include/Descriptors.h"
#include "../include/Harris.h"

#include <spdlog/fmt/ostr.h>
#include <spdlog/spdlog.h>

#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/xfeatures2d.hpp>

#include <vector>

// Constructor
Solution::Solution(const Config& config) : _config(config) {
    // Emplace back configurations
    _featConts.emplace_back(config._images._transA,
                            config._harrisTrans,
                            config._useGpu,
                            config._outputPathPrefix,
                            "/ps4-1-a-1.png",
                            "/ps4-1-b-1.png",
                            "/ps4-1-c-1.png",
                            "/ps4-2-a-1.png",
                            "/ps4-2-b-1.png");
    // transB image
    _featConts.emplace_back(config._images._transB,
                            config._harrisTrans,
                            config._useGpu,
                            config._outputPathPrefix,
                            "/ps4-1-a-3.png",
                            "/ps4-1-b-2.png",
                            "/ps4-1-c-2.png",
                            "/ps4-2-a-3.png",
                            "/ps4-2-b-3.png");
    // simA image
    _featConts.emplace_back(config._images._simA,
                            config._harrisSim,
                            config._useGpu,
                            config._outputPathPrefix,
                            "/ps4-1-a-2.png",
                            "/ps4-1-b-3.png",
                            "/ps4-1-c-3.png",
                            "/ps4-2-a-2.png",
                            "/ps4-2-b-2.png");
    // simB image
    _featConts.emplace_back(config._images._simB,
                            config._harrisSim,
                            config._useGpu,
                            config._outputPathPrefix,
                            "/ps4-1-a-4.png",
                            "/ps4-1-b-4.png",
                            "/ps4-1-c-4.png",
                            "/ps4-2-a-4.png",
                            "/ps4-2-b-4.png");
}

void Solution::drawDots(const cv::Mat& mask, const cv::Mat& img, cv::Mat& dottedImg) {
    // Convert the image to rgb
    cv::Mat img3Ch;
    img.convertTo(img3Ch, CV_8UC3);
    dottedImg.create(img3Ch.rows, img3Ch.cols, img3Ch.type());
    cv::cvtColor(img, dottedImg, cv::COLOR_GRAY2RGB);

    cv::Mat maskNorm(mask.rows, mask.cols, CV_8U);
    cv::normalize(mask, maskNorm, 0, 255, cv::NORM_MINMAX, CV_8U);
    dottedImg.setTo(cv::Scalar(0, 0, 255), maskNorm);
}

void Solution::harrisHelper(FeaturesContainer& conf) {
    auto logger = spdlog::get(config::STDOUT_LOGGER);
    cv::Mat input;
    conf._input.convertTo(input, CV_32F);

    // Compute gradients of the input image
    harris::getGradients(input, conf._config._sobelKernelSize, conf._gradientX, conf._gradientY);

    // Concatenate X and Y gradients for output
    cv::Mat gradCombined(
        conf._gradientX.rows + conf._gradientY.rows, conf._gradientX.cols, CV_8UC1),
        gradXNorm(conf._gradientX.rows, conf._gradientX.cols, CV_8UC1),
        gradYNorm(conf._gradientY.rows, conf._gradientY.cols, CV_8UC1);
    cv::normalize(conf._gradientX, gradXNorm, 0, 255, cv::NORM_MINMAX);
    cv::normalize(conf._gradientY, gradYNorm, 0, 255, cv::NORM_MINMAX);
    cv::hconcat(gradXNorm, gradYNorm, gradCombined);
    cv::imwrite(conf._outPrefix + conf._gradImgPath, gradCombined);
    logger->info("Computed gradients and wrote to {}", conf._gradImgPath);

    // Find Harris values
    if (conf._useGpu) {
        harris::gpu::getCornerResponse(conf._gradientX,
                                       conf._gradientY,
                                       conf._config._windowSize,
                                       conf._config._gaussianSigma,
                                       conf._config._alpha,
                                       conf._cornerResponse);
    } else {
        harris::cpu::getCornerResponse(conf._gradientX,
                                       conf._gradientY,
                                       conf._config._windowSize,
                                       conf._config._gaussianSigma,
                                       conf._config._alpha,
                                       conf._cornerResponse);
    }

    cv::Mat cornerResponseNorm;
    cv::normalize(conf._cornerResponse, cornerResponseNorm, 0, 255, cv::NORM_MINMAX, CV_8UC1);
    cv::imwrite(conf._outPrefix + conf._crImgPath, cornerResponseNorm);
    logger->info("Computed corner responses and wrote to {}", conf._crImgPath);

    // Refine corners
    if (conf._useGpu) {
        harris::gpu::refineCorners(conf._cornerResponse,
                                   conf._config._responseThresh,
                                   conf._config._minDistance,
                                   conf._corners,
                                   conf._cornerLocs);
    } else {
        harris::cpu::refineCorners(conf._cornerResponse,
                                   conf._config._responseThresh,
                                   conf._config._minDistance,
                                   conf._corners,
                                   conf._cornerLocs);
    }

    // Draw dots
    cv::Mat dottedImg;
    drawDots(conf._corners, input, dottedImg);
    cv::imwrite(conf._outPrefix + conf._cornersImgPath, dottedImg);
    logger->info("Computed Harris corners and and wrote to {}", conf._cornersImgPath);
}

void Solution::siftHelper(FeaturesContainer& img1, FeaturesContainer& img2) {
    auto logger = spdlog::get(config::STDOUT_LOGGER);
    auto flogger = spdlog::get(config::FILE_LOGGER);

    static constexpr size_t SIFT_WINDOW_SIZE = 10;

    // Compute angles for each keypoint
    sift::getKeypoints(
        img1._gradientX, img1._gradientY, img1._cornerLocs, SIFT_WINDOW_SIZE, img1._keypoints);
    sift::getKeypoints(
        img2._gradientX, img2._gradientY, img2._cornerLocs, SIFT_WINDOW_SIZE, img2._keypoints);

    // Draw keypoints on each image
    img1._drawnKeypoints = img1._input.clone();
    img2._drawnKeypoints = img2._input.clone();
    cv::drawKeypoints(img1._drawnKeypoints,
                      img1._keypoints,
                      img1._drawnKeypoints,
                      cv::Scalar::all(-1),
                      cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    cv::drawKeypoints(img2._drawnKeypoints,
                      img2._keypoints,
                      img2._drawnKeypoints,
                      cv::Scalar::all(-1),
                      cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    // Concatenate the two images and draw the output
    cv::Mat keypointsCombined;
    cv::hconcat(img1._drawnKeypoints, img2._drawnKeypoints, keypointsCombined);
    cv::imwrite(img1._outPrefix + img1._keypointsImgPath, keypointsCombined);
    logger->info("Found keypoints and drew to {}", img1._keypointsImgPath);

    // Compute SIFT descriptors
    auto sift = cv::xfeatures2d::SIFT::create();
    cv::Mat descriptorsImg1, descriptorsImg2;
    sift->compute(img1._input, img1._keypoints, descriptorsImg1);
    sift->compute(img2._input, img2._keypoints, descriptorsImg2);

    // Match descriptors
    auto matcher = cv::BFMatcher::create();

    img1._goodMatches.clear();
    img2._goodMatches.clear();
    // Use KNN to find 2 matches for each point so we can apply the ratio test from the original
    // SIFT paper (https://people.eecs.berkeley.edu/~malik/cs294/lowe-ijcv04.pdf)
    std::vector<std::vector<cv::DMatch>> rawMatches;
    matcher->knnMatch(descriptorsImg1, descriptorsImg2, rawMatches, 2);
    for (const auto& matchPair : rawMatches) {
        if (matchPair[0].distance < 0.75 * matchPair[1].distance) {
            img1._goodMatches.push_back(matchPair[0]);
        }
    }
    // Copy good matches from img1 to img2
    img2._goodMatches = img1._goodMatches;

    // Create image with lines drawn between matched points. As we iterate through each point, log
    // its info
    cv::Mat combinedSrc;
    cv::hconcat(img1._drawnKeypoints, img2._drawnKeypoints, combinedSrc);
    std::stringstream ss;
    ss << "\nMatches:";
    cv::RNG rng(12345);
    for (const auto& match : img1._goodMatches) {
        cv::KeyPoint k1 = img1._keypoints[match.queryIdx];
        cv::KeyPoint k2 = img2._keypoints[match.trainIdx];
        int xOffset = img1._drawnKeypoints.cols;
        cv::line(combinedSrc,
                 k1.pt,
                 cv::Point2f(k2.pt.x + xOffset, k2.pt.y),
                 cv::Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255)));
        ss << "\nqueryIdx=" << match.queryIdx << "; trainIdx=" << match.trainIdx
           << "; distance=" << match.distance;
    }
    flogger->info("{}", ss.str());
    cv::imwrite(img1._outPrefix + img1._matchesImgPath, combinedSrc);
    logger->info("Found {} good matches; drew match pairs to {}",
                 img1._goodMatches.size(),
                 img1._matchesImgPath);
}

std::tuple<cv::Mat, std::vector<int>, double>
    Solution::ransacHelper(FeaturesContainer& img1,
                           FeaturesContainer& img2,
                           ransac::TransformType whichRansac,
                           const Config::RANSAC& settings,
                           const std::string& outputPath) {
    auto logger = spdlog::get(config::STDOUT_LOGGER);
    // Iterate over the good matches in the trans images and build up a list of matching points
    std::vector<cv::Point2f> transAPts, transBPts;
    for (const auto& match : img1._goodMatches) {
        transAPts.emplace_back(img1._keypoints[match.queryIdx].pt);
        transBPts.emplace_back(img2._keypoints[match.trainIdx].pt);
    }

    cv::Mat transform;
    std::vector<int> consensusSet;
    double consensusRatio;
    std::tie(transform, consensusSet, consensusRatio) = ransac::solve(transAPts,
                                                                      transBPts,
                                                                      whichRansac,
                                                                      settings._reprojThresh,
                                                                      settings._maxIters,
                                                                      settings._minConsensusRatio);

    logger->info("\nBest transform =\n{}\nwith consensus ratio {}", transform, consensusRatio);

    // Draw that shit
    cv::Mat combinedSrc;
    cv::hconcat(img1._input, img2._input, combinedSrc);
    cv::cvtColor(combinedSrc, combinedSrc, cv::COLOR_GRAY2RGB);
    cv::RNG rng(12345);
    for (const int& idx : consensusSet) {
        cv::line(combinedSrc,
                 transAPts[idx],
                 cv::Point2f(transBPts[idx].x + img1._input.cols, transBPts[idx].y),
                 cv::Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255)));
    }
    cv::imwrite(_config._outputPathPrefix + outputPath, combinedSrc);

    return std::make_tuple(transform, consensusSet, consensusRatio);
}

void Solution::runProblem1() {
    // Time runtime
    auto logger = spdlog::get(config::STDOUT_LOGGER);
    logger->info("Problem 1 begins");
    auto start = std::chrono::high_resolution_clock::now();

    // Iterate over the containers and run the problem set
    for (auto& container : _featConts) {
        harrisHelper(container);
    }

    auto finish = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> runtime = finish - start;
    logger->info("Problem 1 runtime = {} ms", runtime.count());
}

void Solution::runProblem2() {
    // Time runtime
    auto logger = spdlog::get(config::STDOUT_LOGGER);
    logger->info("Problem 2 begins");
    auto start = std::chrono::high_resolution_clock::now();

    siftHelper(_featConts[0], _featConts[1]);
    siftHelper(_featConts[2], _featConts[3]);

    auto finish = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> runtime = finish - start;
    logger->info("Problem 2 runtime = {} ms", runtime.count());
}

void Solution::runProblem3() {
    // Time runtime
    auto logger = spdlog::get(config::STDOUT_LOGGER);
    logger->info("Problem 3 begins");
    auto start = std::chrono::high_resolution_clock::now();

    // Seed RANSAC's RNG first
    ransac::seed(_config._mersenneSeed);

    logger->info("RANSAC for a translation:");
    ransacHelper(_featConts[0],
                 _featConts[1],
                 ransac::TransformType::TRANSLATION,
                 _config._trans,
                 "/ps4-3-a-1.png");

    {
        // Now compute transform for sim images
        cv::Mat transform;
        std::vector<int> consensusSet;
        double consensusRatio;
        logger->info("RANSAC for a similarity transformation:");
        std::tie(transform, consensusSet, consensusRatio) =
            ransacHelper(_featConts[2],
                         _featConts[3],
                         ransac::TransformType::SIMILARITY,
                         _config._sim,
                         "/ps4-3-b-1.png");

        // Invert the transform to so we can warp the simB image back to simA
        cv::invertAffineTransform(transform, transform);

        cv::Mat simA = _featConts[2]._input.clone();
        cv::Mat simB = _featConts[3]._input.clone();
        cv::Mat reverseWarp = cv::Mat::zeros(simB.rows, simB.cols, simB.type());

        // Do the warp
        cv::warpAffine(simB, reverseWarp, transform, reverseWarp.size());

        // Blend with original
        cv::Mat blended = simA * 0.5 + reverseWarp * 0.5;

        cv::imwrite(_config._outputPathPrefix + "/ps4-3-d-1.png", blended);
    }

    {
        // Try with affine transform
        cv::Mat transform;
        std::vector<int> consensusSet;
        double consensusRatio;
        logger->info("RANSAC for an affine transformation:");
        std::tie(transform, consensusSet, consensusRatio) =
            ransacHelper(_featConts[2],
                         _featConts[3],
                         ransac::TransformType::AFFINE,
                         _config._affine,
                         "/ps4-3-c-1.png");

        // Invert the transform to so we can warp the simB image back to simA
        cv::invertAffineTransform(transform, transform);

        cv::Mat simA = _featConts[2]._input.clone();
        cv::Mat simB = _featConts[3]._input.clone();
        cv::Mat reverseWarp = cv::Mat::zeros(simB.rows, simB.cols, simB.type());

        // Do the warp
        cv::warpAffine(simB, reverseWarp, transform, reverseWarp.size());

        // Blend with original
        cv::Mat blended = simA * 0.5 + reverseWarp * 0.5;

        cv::imwrite(_config._outputPathPrefix + "/ps4-3-e-1.png", blended);
    }

    auto finish = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> runtime = finish - start;
    logger->info("Problem 3 runtime = {} ms", runtime.count());
}