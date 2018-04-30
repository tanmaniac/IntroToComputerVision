#include "../include/Config.h"
#include "../include/DisparityNCorr.h"
#include "../include/DisparitySSD.h"

#include <common/CudaWarmup.h>

#include <spdlog/spdlog.h>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <chrono>
#include <memory>

// YAML file containing input parameters
static constexpr char CONFIG_FILE_PATH[] = "../config/ps2.yaml";

std::shared_ptr<spdlog::logger> _logger, _fileLogger;

// Computes left and right reference disparities using sum of square difference.
void disparitySSDPair(const cv::Mat& left,
                      const cv::Mat& right,
                      const bool useGpuDisparity,
                      const Config::DisparitySSD& config,
                      cv::Mat& leftDisparity,
                      cv::Mat& rightDisparity) {
    assert(left.type() == CV_32FC1 && right.type() == CV_32FC1);
    assert(left.rows == right.rows && left.cols == right.cols);
    // Run with left side as reference first
    leftDisparity = 0;
    if (useGpuDisparity) {
        cuda::disparitySSD(
            left, right, config._windowRadius, -config._disparityRange, 0, leftDisparity);
    } else {
        serial::disparitySSD(
            left, right, config._windowRadius, -config._disparityRange, 0, leftDisparity);
    }

    // Run with right side as reference
    rightDisparity = 0;
    if (useGpuDisparity) {
        cuda::disparitySSD(
            right, left, config._windowRadius, 0, config._disparityRange, rightDisparity);
    } else {
        serial::disparitySSD(
            right, left, config._windowRadius, 0, config._disparityRange, rightDisparity);
    }
}

// Computes left and right reference disparities using normalized cross correlation.
void disparityNCorrPair(const cv::Mat& left,
                        const cv::Mat& right,
                        const bool useGpuDisparity,
                        const Config::DisparitySSD& config,
                        cv::Mat& leftDisparity,
                        cv::Mat& rightDisparity) {
    assert(left.type() == CV_32FC1 && right.type() == CV_32FC1);
    assert(left.rows == right.rows && left.cols == right.cols);
    // Run with left side as reference first
    leftDisparity = 0;
    if (useGpuDisparity) {
        cuda::disparityNCorr(
            left, right, config._windowRadius, -config._disparityRange, 0, leftDisparity);
    } else {
        serial::disparityNCorr(
            left, right, config._windowRadius, -config._disparityRange, 0, leftDisparity);
    }

    // Run with right side as reference
    rightDisparity = 0;
    if (useGpuDisparity) {
        cuda::disparityNCorr(
            right, left, config._windowRadius, 0, config._disparityRange, rightDisparity);
    } else {
        serial::disparityNCorr(
            right, left, config._windowRadius, 0, config._disparityRange, rightDisparity);
    }
}

void runProblem1(const Config& config) {
    // Time runtime
    _logger->info("Problem 1 begins");
    auto start = std::chrono::high_resolution_clock::now();

    // Convert images to floating point
    cv::Mat left, right, leftDisparity, rightDisparity;
    config._images._pair0.first.convertTo(left, CV_32FC1);
    config._images._pair0.second.convertTo(right, CV_32FC1);

    disparitySSDPair(
        left, right, config._useGpuDisparity, config._p1disp, leftDisparity, rightDisparity);

    // Normalize for display
    cv::normalize(leftDisparity, leftDisparity, 0, 255, cv::NORM_MINMAX, CV_8UC1);
    cv::imwrite(config._outputPathPrefix + "/ps2-1-a-1.png", leftDisparity);

    // Normalize for display
    cv::normalize(rightDisparity, rightDisparity, 0, 255, cv::NORM_MINMAX, CV_8UC1);
    cv::imwrite(config._outputPathPrefix + "/ps2-1-a-2.png", rightDisparity);

    // Record end time
    auto finish = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> runtime = finish - start;
    _logger->info("Problem 1 runtime = {} ms", runtime.count());
}

void runProblem2(const Config& config) {
    // Time runtime
    _logger->info("Problem 2 begins");
    auto start = std::chrono::high_resolution_clock::now();

    // Convert images to greyscale floating point
    cv::Mat left, right, leftDisparity, rightDisparity;
    cv::cvtColor(config._images._pair1.first, left, cv::COLOR_RGB2GRAY, 1);
    left.convertTo(left, CV_32FC1);
    cv::cvtColor(config._images._pair1.second, right, cv::COLOR_RGB2GRAY, 1);
    right.convertTo(right, CV_32FC1);

    disparitySSDPair(
        left, right, config._useGpuDisparity, config._p2disp, leftDisparity, rightDisparity);

    // Normalize for display
    cv::normalize(leftDisparity, leftDisparity, 0, 255, cv::NORM_MINMAX, CV_8UC1);
    cv::imwrite(config._outputPathPrefix + "/ps2-2-a-1.png", leftDisparity);
    // Invert colors, since darker means further away but we have negative disparity values
    cv::Mat ones = cv::Mat::ones(leftDisparity.size(), leftDisparity.type()) * 255;
    leftDisparity = ones - leftDisparity;
    cv::imwrite(config._outputPathPrefix + "/ps2-2-a-1-inverted.png", leftDisparity);

    // Normalize for display
    cv::normalize(rightDisparity, rightDisparity, 0, 255, cv::NORM_MINMAX, CV_8UC1);
    cv::imwrite(config._outputPathPrefix + "/ps2-2-a-2.png", rightDisparity);

    // Record end time
    auto finish = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> runtime = finish - start;
    _logger->info("Problem 2 runtime = {} ms", runtime.count());
}

void addNoise(const cv::Mat& first,
              const cv::Mat& second,
              const float mean,
              const float sigma,
              cv::Mat& firstNoisy,
              cv::Mat& secondNoisy) {
    cv::Mat noise(first.size(), first.type());
    cv::randn(noise, mean, sigma);
    firstNoisy = first + noise;

    noise.create(second.size(), second.type());
    cv::randn(noise, mean, sigma);
    secondNoisy = second + noise;
}

void runProblem3(const Config& config) {
    // Time runtime
    _logger->info("Problem 3 begins");
    auto start = std::chrono::high_resolution_clock::now();

    // Convert images to greyscale floating point
    cv::Mat left, right, leftDisparity, rightDisparity;
    cv::cvtColor(config._images._pair1.first, left, cv::COLOR_RGB2GRAY, 1);
    left.convertTo(left, CV_32FC1);
    cv::cvtColor(config._images._pair1.second, right, cv::COLOR_RGB2GRAY, 1);
    right.convertTo(right, CV_32FC1);

    // Add noise
    cv::Mat leftNoisy, rightNoisy;
    addNoise(left, right, 0, 10, leftNoisy, rightNoisy);

    disparitySSDPair(leftNoisy,
                     rightNoisy,
                     config._useGpuDisparity,
                     config._p3disp,
                     leftDisparity,
                     rightDisparity);

    // Normalize for display
    cv::normalize(leftDisparity, leftDisparity, 0, 255, cv::NORM_MINMAX, CV_8UC1);
    cv::imwrite(config._outputPathPrefix + "/ps2-3-a-1.png", leftDisparity);
    // Invert colors, since darker means further away but we have negative disparity values
    cv::Mat ones = cv::Mat::ones(leftDisparity.size(), leftDisparity.type()) * 255;
    leftDisparity = ones - leftDisparity;
    cv::imwrite(config._outputPathPrefix + "/ps2-3-a-1-inverted.png", leftDisparity);

    // Normalize for display
    cv::normalize(rightDisparity, rightDisparity, 0, 255, cv::NORM_MINMAX, CV_8UC1);
    cv::imwrite(config._outputPathPrefix + "/ps2-3-a-2.png", rightDisparity);

    //-------------- Part 2: Increasing contrast instead of adding noise --------------
    float contrastFactor = 1.1;
    cv::Mat leftContrasty = left * contrastFactor;
    cv::Mat rightContrasty = right * contrastFactor;

    disparitySSDPair(leftContrasty,
                     rightContrasty,
                     config._useGpuDisparity,
                     config._p3disp,
                     leftDisparity,
                     rightDisparity);

    // Normalize for display
    cv::normalize(leftDisparity, leftDisparity, 0, 255, cv::NORM_MINMAX, CV_8UC1);
    cv::imwrite(config._outputPathPrefix + "/ps2-3-b-1.png", leftDisparity);
    // Invert colors, since darker means further away but we have negative disparity values
    leftDisparity = ones - leftDisparity;
    cv::imwrite(config._outputPathPrefix + "/ps2-3-b-1-inverted.png", leftDisparity);

    // Normalize for display
    cv::normalize(rightDisparity, rightDisparity, 0, 255, cv::NORM_MINMAX, CV_8UC1);
    cv::imwrite(config._outputPathPrefix + "/ps2-3-b-2.png", rightDisparity);

    // Record end time
    auto finish = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> runtime = finish - start;
    _logger->info("Problem 3 runtime = {} ms", runtime.count());
}

void runProblem4(const Config& config) {
    // Time runtime
    _logger->info("Problem 4 begins");
    auto start = std::chrono::high_resolution_clock::now();

    // Convert images to greyscale floating point
    cv::Mat left, right, leftDisparity, rightDisparity;
    cv::cvtColor(config._images._pair1.first, left, cv::COLOR_RGB2GRAY, 1);
    left.convertTo(left, CV_32FC1);
    cv::cvtColor(config._images._pair1.second, right, cv::COLOR_RGB2GRAY, 1);
    right.convertTo(right, CV_32FC1);

    disparityNCorrPair(
        left, right, config._useGpuDisparity, config._p4disp, leftDisparity, rightDisparity);

    // Normalize for display
    cv::normalize(leftDisparity, leftDisparity, 0, 255, cv::NORM_MINMAX, CV_8UC1);
    cv::imwrite(config._outputPathPrefix + "/ps2-4-a-1.png", leftDisparity);
    // Invert colors, since darker means further away but we have negative disparity values
    cv::Mat ones = cv::Mat::ones(leftDisparity.size(), leftDisparity.type()) * 255;
    leftDisparity = ones - leftDisparity;
    cv::imwrite(config._outputPathPrefix + "/ps2-4-a-1-inverted.png", leftDisparity);

    // Normalize for display
    cv::normalize(rightDisparity, rightDisparity, 0, 255, cv::NORM_MINMAX, CV_8UC1);
    cv::imwrite(config._outputPathPrefix + "/ps2-4-a-2.png", rightDisparity);

    //--------------- Normalized cross correlation on noisy images ---------------
    // Add noise
    cv::Mat leftNoisy, rightNoisy;
    addNoise(left, right, 0, 10, leftNoisy, rightNoisy);

    disparityNCorrPair(leftNoisy,
                       rightNoisy,
                       config._useGpuDisparity,
                       config._p4disp,
                       leftDisparity,
                       rightDisparity);
    // Normalize for display
    cv::normalize(leftDisparity, leftDisparity, 0, 255, cv::NORM_MINMAX, CV_8UC1);
    cv::imwrite(config._outputPathPrefix + "/ps2-4-b-1.png", leftDisparity);
    // Invert colors, since darker means further away but we have negative disparity values
    leftDisparity = ones - leftDisparity;
    cv::imwrite(config._outputPathPrefix + "/ps2-4-b-1-inverted.png", leftDisparity);

    // Normalize for display
    cv::normalize(rightDisparity, rightDisparity, 0, 255, cv::NORM_MINMAX, CV_8UC1);
    cv::imwrite(config._outputPathPrefix + "/ps2-4-b-2.png", rightDisparity);

    //--------------- Normalized cross correlation on contrast-boosted images ---------------
    float contrastFactor = 1.1;
    cv::Mat leftContrasty = left * contrastFactor;
    cv::Mat rightContrasty = right * contrastFactor;

    disparityNCorrPair(leftContrasty,
                       rightContrasty,
                       config._useGpuDisparity,
                       config._p4disp,
                       leftDisparity,
                       rightDisparity);
    // Normalize for display
    cv::normalize(leftDisparity, leftDisparity, 0, 255, cv::NORM_MINMAX, CV_8UC1);
    cv::imwrite(config._outputPathPrefix + "/ps2-4-c-1.png", leftDisparity);
    // Invert colors, since darker means further away but we have negative disparity values
    leftDisparity = ones - leftDisparity;
    cv::imwrite(config._outputPathPrefix + "/ps2-4-c-1-inverted.png", leftDisparity);

    // Normalize for display
    cv::normalize(rightDisparity, rightDisparity, 0, 255, cv::NORM_MINMAX, CV_8UC1);
    cv::imwrite(config._outputPathPrefix + "/ps2-4-c-2.png", rightDisparity);

    // Record end time
    auto finish = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> runtime = finish - start;
    _logger->info("Problem 4 runtime = {} ms", runtime.count());
}

void runProblem5(const Config& config) {
    // Time runtime
    _logger->info("Problem 5 begins");
    auto start = std::chrono::high_resolution_clock::now();

    // Convert images to greyscale floating point
    cv::Mat left, right, leftDisparity, rightDisparity;
    cv::cvtColor(config._images._pair2.first, left, cv::COLOR_RGB2GRAY, 1);
    left.convertTo(left, CV_32FC1);
    cv::cvtColor(config._images._pair2.second, right, cv::COLOR_RGB2GRAY, 1);
    right.convertTo(right, CV_32FC1);

    disparityNCorrPair(
        left, right, config._useGpuDisparity, config._p5disp, leftDisparity, rightDisparity);

    // Normalize for display
    cv::normalize(leftDisparity, leftDisparity, 0, 255, cv::NORM_MINMAX, CV_8UC1);
    cv::imwrite(config._outputPathPrefix + "/ps2-5-a-1.png", leftDisparity);
    // Invert colors, since darker means further away but we have negative disparity values
    cv::Mat ones = cv::Mat::ones(leftDisparity.size(), leftDisparity.type()) * 255;
    leftDisparity = ones - leftDisparity;
    cv::imwrite(config._outputPathPrefix + "/ps2-5-a-1-inverted.png", leftDisparity);

    // Normalize for display
    cv::normalize(rightDisparity, rightDisparity, 0, 255, cv::NORM_MINMAX, CV_8UC1);
    cv::imwrite(config._outputPathPrefix + "/ps2-5-a-2.png", rightDisparity);

    // Record end time
    auto finish = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> runtime = finish - start;
    _logger->info("Problem 5 runtime = {} ms", runtime.count());
}

int main() {
    // Set up loggers
    std::vector<spdlog::sink_ptr> sinks;
    auto colorStdoutSink = std::make_shared<spdlog::sinks::ansicolor_stdout_sink_mt>();
    auto fileSink = std::make_shared<spdlog::sinks::simple_file_sink_mt>("ps2.log");
    sinks.push_back(colorStdoutSink);
    sinks.push_back(fileSink);
    _logger = std::make_shared<spdlog::logger>(config::STDOUT_LOGGER, begin(sinks), end(sinks));
    // File logger is just for CUDA kernel outputs
    _fileLogger = std::make_shared<spdlog::logger>(config::FILE_LOGGER, fileSink);
    spdlog::register_logger(_logger);
    spdlog::register_logger(_fileLogger);

    Config config(CONFIG_FILE_PATH);

    // If we're using the GPU, warm it up by running a small kernel that does nothing. Required for
    // accurate timing data.
    if (config._useGpuDisparity) {
        common::warmup();
        _fileLogger->info("GPU warmup done");
    }

    auto start = std::chrono::high_resolution_clock::now();
    // Run problems
    runProblem1(config);
    runProblem2(config);
    runProblem3(config);
    runProblem4(config);
    runProblem5(config);
    auto finish = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> runtime = finish - start;
    _logger->info("Total runtime: {} ms", runtime.count());
}
