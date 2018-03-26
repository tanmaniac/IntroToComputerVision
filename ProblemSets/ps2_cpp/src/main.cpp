#include "../include/Config.h"
#include "../include/DisparitySSD.h"

#include <spdlog/spdlog.h>

#include <opencv2/highgui/highgui.hpp>

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

    if (!config._useGpuDisparity) {
        serial::disparitySSD(left, right, ReferenceFrame::LEFT, 6, 0, 3, disparity);
        // Normalize for display
        cv::normalize(disparity, disparity, 0, 255, cv::NORM_MINMAX, CV_8UC1);
        cv::imwrite(config._outputPathPrefix + "/ps2-1-a-1.png", disparity);
        disparity = 0;
        serial::disparitySSD(left, right, ReferenceFrame::RIGHT, 6, 0, 3, disparity);
        // Normalize for display
        cv::normalize(disparity, disparity, 0, 255, cv::NORM_MINMAX, CV_8UC1);
        cv::imwrite(config._outputPathPrefix + "/ps2-1-a-2.png", disparity);
    } else {
        // Run CUDA version
        disparity = 0;
        cuda::disparitySSD(left, right, ReferenceFrame::LEFT, 10, -3, 0, disparity);
        // Normalize for display
        cv::normalize(disparity, disparity, 0, 255, cv::NORM_MINMAX, CV_8UC1);
        cv::imwrite(config._outputPathPrefix + "/ps2-1-a-1.png", disparity);
        disparity = 0;
        cuda::disparitySSD(right, left, ReferenceFrame::RIGHT, 8, 0, 5, disparity);
        // Normalize for display
        cv::normalize(disparity, disparity, 0, 255, cv::NORM_MINMAX, CV_8UC1);
        cv::imwrite(config._outputPathPrefix + "/ps2-1-a-2.png", disparity);
    }

    // Record end time
    auto finish = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> runtime = finish - start;
    _logger->info("Problem 1 runtime = {} ms", runtime.count());
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

    runProblem1(config);
}