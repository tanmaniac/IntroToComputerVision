#include "Solution.h"
#include "../include/Harris.h"

#include <spdlog/spdlog.h>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <vector>

void drawDots(const cv::Mat& mask, const cv::Mat& img, cv::Mat& dottedImg) {
    // Convert the image to rgb
    cv::Mat img3Ch;
    img.convertTo(img3Ch, CV_8UC3);
    dottedImg.create(img3Ch.rows, img3Ch.cols, img3Ch.type());
    cv::cvtColor(img, dottedImg, cv::COLOR_GRAY2RGB);

    cv::Mat maskNorm(mask.rows, mask.cols, CV_8U);
    cv::normalize(mask, maskNorm, 0, 255, cv::NORM_MINMAX, CV_8U);
    dottedImg.setTo(cv::Scalar(0, 0, 255), maskNorm);
}

// Helpful struct to hold problem set configurations for Harris corners problems, so that we don't
// have to copy/paste a bunch of code
struct HarrisContainer {
    // Reference to input image
    const cv::Mat& _input;
    // Reference to runtime configuration for a Harris corner detector
    const Config::Harris& _config;
    // Flags to use CUDA for computation
    const bool _useGpu;
    const std::string _outPrefix, _gradImgPath, _crImgPath, _cornersImgPath;

    // Filled when Harris functions are run
    cv::Mat _gradientX, _gradientY, _cornerResponse, _corners;

    HarrisContainer(const cv::Mat& input,
                    const Config::Harris& config,
                    const bool useGpu,
                    const std::string& outPrefix,
                    const std::string& gradImgPath,
                    const std::string& crImgPath,
                    const std::string& cornersImgPath)
        : _input(input), _config(config), _useGpu(useGpu), _outPrefix(outPrefix),
          _gradImgPath(gradImgPath), _crImgPath(crImgPath), _cornersImgPath(cornersImgPath) {}
};

void harrisHelper(HarrisContainer& conf) {
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
    harris::cpu::refineCorners(conf._cornerResponse,
                               conf._config._responseThresh,
                               conf._config._minDistance,
                               conf._corners);

    // Draw dots
    cv::Mat dottedImg;
    drawDots(conf._corners, input, dottedImg);
    cv::imwrite(conf._outPrefix + conf._cornersImgPath, dottedImg);
    logger->info("Computed Harris corners and and wrote to {}", conf._cornersImgPath);
}

void solution::runProblem1(const Config& config) {
    // Time runtime
    auto logger = spdlog::get(config::STDOUT_LOGGER);
    logger->info("Problem 1 begins");
    auto start = std::chrono::high_resolution_clock::now();

    // Create a vector to hold all the various Mats that are created in this problem
    std::vector<HarrisContainer> containers;

    // Emplace back configurations
    containers.emplace_back(config._images._transA,
                            config._harrisTrans,
                            config._useGpu,
                            config._outputPathPrefix,
                            "/ps4-1-a-1.png",
                            "/ps4-1-b-1.png",
                            "/ps4-1-c-1.png");
    // transB image
    containers.emplace_back(config._images._transB,
                            config._harrisTrans,
                            config._useGpu,
                            config._outputPathPrefix,
                            "/ps4-1-a-3.png",
                            "/ps4-1-b-2.png",
                            "/ps4-1-c-2.png");
    // simA image
    containers.emplace_back(config._images._simA,
                            config._harrisSim,
                            config._useGpu,
                            config._outputPathPrefix,
                            "/ps4-1-a-2.png",
                            "/ps4-1-b-3.png",
                            "/ps4-1-c-3.png");
    // simB image
    containers.emplace_back(config._images._simB,
                            config._harrisSim,
                            config._useGpu,
                            config._outputPathPrefix,
                            "/ps4-1-a-4.png",
                            "/ps4-1-b-4.png",
                            "/ps4-1-c-4.png");

    // Iterate over the containers and run the problem set
    for (auto& container : containers) {
        harrisHelper(container);
    }

    auto finish = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> runtime = finish - start;
    logger->info("Problem 1 runtime = {} ms", runtime.count());
}