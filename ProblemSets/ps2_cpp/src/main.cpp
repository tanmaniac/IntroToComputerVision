#include "../include/Config.h"
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

void runProblem1(const Config& config) {
    // Time runtime
    _logger->info("Problem 1 begins");
    auto start = std::chrono::high_resolution_clock::now();

    // Convert images to floating point
    cv::Mat left, right, disparity;
    config._images._pair0.first.convertTo(left, CV_32FC1);
    config._images._pair0.second.convertTo(right, CV_32FC1);

    // Left side is reference
    if (!config._useGpuDisparity) {
        // Run CPU version
        serial::disparitySSD(
            left, right, config._p1ssd._windowRadius, -config._p1ssd._disparityRange, 0, disparity);
    } else {
        // Run CUDA version
        cuda::disparitySSD(
            left, right, config._p1ssd._windowRadius, -config._p1ssd._disparityRange, 0, disparity);
    }
    // Normalize for display
    cv::normalize(disparity, disparity, 0, 255, cv::NORM_MINMAX, CV_8UC1);
    cv::imwrite(config._outputPathPrefix + "/ps2-1-a-1.png", disparity);

    // Right side is reference
    disparity = 0;
    if (!config._useGpuDisparity) {
        serial::disparitySSD(
            right, left, config._p1ssd._windowRadius, 0, config._p1ssd._disparityRange, disparity);
    } else {
        cuda::disparitySSD(
            right, left, config._p1ssd._windowRadius, 0, config._p1ssd._disparityRange, disparity);
    }
    // Normalize for display
    cv::normalize(disparity, disparity, 0, 255, cv::NORM_MINMAX, CV_8UC1);
    cv::imwrite(config._outputPathPrefix + "/ps2-1-a-2.png", disparity);

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
    cv::Mat left, right, disparity;
    cv::cvtColor(config._images._pair1.first, left, cv::COLOR_RGB2GRAY, 1);
    left.convertTo(left, CV_32FC1);
    cv::cvtColor(config._images._pair1.second, right, cv::COLOR_RGB2GRAY, 1);
    right.convertTo(right, CV_32FC1);

    // Left side is reference
    if (!config._useGpuDisparity) {
        // Run CPU version
        serial::disparitySSD(
            left, right, config._p2ssd._windowRadius, -config._p2ssd._disparityRange, 0, disparity);
    } else {
        // Run CUDA version
        cuda::disparitySSD(
            left, right, config._p2ssd._windowRadius, -config._p2ssd._disparityRange, 0, disparity);
    }
    // Normalize for display
    cv::normalize(disparity, disparity, 0, 255, cv::NORM_MINMAX, CV_8UC1);
    cv::imwrite(config._outputPathPrefix + "/ps2-2-a-1.png", disparity);
    // Invert colors, since darker means further away but we have negative disparity values
    cv::Mat ones = cv::Mat::ones(disparity.size(), disparity.type()) * 255;
    disparity = ones - disparity;
    cv::imwrite(config._outputPathPrefix + "/ps2-2-a-1-inverted.png", disparity);

    // Right side is reference
    disparity = 0;
    if (!config._useGpuDisparity) {
        serial::disparitySSD(
            right, left, config._p2ssd._windowRadius, 0, config._p2ssd._disparityRange, disparity);
    } else {
        cuda::disparitySSD(
            right, left, config._p2ssd._windowRadius, 0, config._p2ssd._disparityRange, disparity);
    }
    // Normalize for display
    cv::normalize(disparity, disparity, 0, 255, cv::NORM_MINMAX, CV_8UC1);
    cv::imwrite(config._outputPathPrefix + "/ps2-2-a-2.png", disparity);

    // Record end time
    auto finish = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> runtime = finish - start;
    _logger->info("Problem 2 runtime = {} ms", runtime.count());
}

void runProblem3(const Config& config) {
    // Time runtime
    _logger->info("Problem 3 begins");
    auto start = std::chrono::high_resolution_clock::now();

    // Convert images to greyscale floating point
    cv::Mat left, right, disparity;
    cv::cvtColor(config._images._pair1.first, left, cv::COLOR_RGB2GRAY, 1);
    left.convertTo(left, CV_32FC1);
    cv::cvtColor(config._images._pair1.second, right, cv::COLOR_RGB2GRAY, 1);
    right.convertTo(right, CV_32FC1);

    // Add noise
    cv::Mat noise(left.size(), left.type());
    float mean = 0;
    float sigma = 10;
    cv::randn(noise, mean, sigma);
    cv::Mat leftNoisy = left + noise;
    cv::randn(noise, mean, sigma);
    cv::Mat rightNoisy = right + noise;

    // Left side is reference
    if (!config._useGpuDisparity) {
        // Run CPU version
        serial::disparitySSD(leftNoisy,
                             rightNoisy,
                             config._p3ssd._windowRadius,
                             -config._p3ssd._disparityRange,
                             0,
                             disparity);
    } else {
        // Run CUDA version
        cuda::disparitySSD(leftNoisy,
                           rightNoisy,
                           config._p3ssd._windowRadius,
                           -config._p3ssd._disparityRange,
                           0,
                           disparity);
    }
    // Normalize for display
    cv::normalize(disparity, disparity, 0, 255, cv::NORM_MINMAX, CV_8UC1);
    cv::imwrite(config._outputPathPrefix + "/ps2-3-a-1.png", disparity);
    // Invert colors, since darker means further away but we have negative disparity values
    cv::Mat ones = cv::Mat::ones(disparity.size(), disparity.type()) * 255;
    disparity = ones - disparity;
    cv::imwrite(config._outputPathPrefix + "/ps2-3-a-1-inverted.png", disparity);

    // Right side is reference
    disparity = 0;
    if (!config._useGpuDisparity) {
        serial::disparitySSD(rightNoisy,
                             leftNoisy,
                             config._p3ssd._windowRadius,
                             0,
                             config._p3ssd._disparityRange,
                             disparity);
    } else {
        cuda::disparitySSD(rightNoisy,
                           leftNoisy,
                           config._p3ssd._windowRadius,
                           0,
                           config._p3ssd._disparityRange,
                           disparity);
    }
    // Normalize for display
    cv::normalize(disparity, disparity, 0, 255, cv::NORM_MINMAX, CV_8UC1);
    cv::imwrite(config._outputPathPrefix + "/ps2-3-a-2.png", disparity);

    //-------------- Part 2: Increasing contrast instead of adding noise --------------
    float contrastFactor = 1.1;
    cv::Mat leftContrasty = left * contrastFactor;
    cv::Mat rightContrasty = right * contrastFactor;

    // Left side is reference
    if (!config._useGpuDisparity) {
        // Run CPU version
        serial::disparitySSD(leftContrasty,
                             rightContrasty,
                             config._p3ssd._windowRadius,
                             -config._p3ssd._disparityRange,
                             0,
                             disparity);
    } else {
        // Run CUDA version
        cuda::disparitySSD(leftContrasty,
                           rightContrasty,
                           config._p3ssd._windowRadius,
                           -config._p3ssd._disparityRange,
                           0,
                           disparity);
    }
    // Normalize for display
    cv::normalize(disparity, disparity, 0, 255, cv::NORM_MINMAX, CV_8UC1);
    cv::imwrite(config._outputPathPrefix + "/ps2-3-b-1.png", disparity);
    // Invert colors, since darker means further away but we have negative disparity values
    disparity = ones - disparity;
    cv::imwrite(config._outputPathPrefix + "/ps2-3-b-1-inverted.png", disparity);

    // Right side is reference
    disparity = 0;
    if (!config._useGpuDisparity) {
        serial::disparitySSD(rightContrasty,
                             leftContrasty,
                             config._p3ssd._windowRadius,
                             0,
                             config._p3ssd._disparityRange,
                             disparity);
    } else {
        cuda::disparitySSD(rightContrasty,
                           leftContrasty,
                           config._p3ssd._windowRadius,
                           0,
                           config._p3ssd._disparityRange,
                           disparity);
    }
    // Normalize for display
    cv::normalize(disparity, disparity, 0, 255, cv::NORM_MINMAX, CV_8UC1);
    cv::imwrite(config._outputPathPrefix + "/ps2-3-b-2.png", disparity);

    // Record end time
    auto finish = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> runtime = finish - start;
    _logger->info("Problem 3 runtime = {} ms", runtime.count());
}

int main() {
    std::vector<spdlog::sink_ptr> sinks;
    auto colorStdoutSink = std::make_shared<spdlog::sinks::ansicolor_stdout_sink_mt>();
    auto fileSink = std::make_shared<spdlog::sinks::simple_file_sink_mt>("ps2.log");
    sinks.push_back(colorStdoutSink);
    sinks.push_back(fileSink);
    _logger = std::make_shared<spdlog::logger>("logger", begin(sinks), end(sinks));
    // File logger is just for CUDA kernel outputs
    _fileLogger = std::make_shared<spdlog::logger>("file_logger", fileSink);
    spdlog::register_logger(_logger);
    spdlog::register_logger(_fileLogger);

    Config config(CONFIG_FILE_PATH);

    // If we're using the GPU, warm it up by running a small kernel that does nothing. Required for
    // accurate timing data.
    if (config._useGpuDisparity) {
        common::warmup();
        _fileLogger->info("GPU warmup done");
    }

    // Run problems
    runProblem1(config);
    runProblem2(config);
    runProblem3(config);
}
