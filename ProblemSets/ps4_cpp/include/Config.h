#pragma once

// Structures to hold runtime configurations.
#include <common/BasicConfig.h>
#include <common/Utils.h>

#include <spdlog/spdlog.h>
#include <yaml-cpp/yaml.h>

#include <opencv2/core/core.hpp>

#include <cstddef>
#include <iostream>
#include <memory>
#include <random>

namespace config {
static constexpr char FILE_LOGGER[] = "file_logger";
static constexpr char STDOUT_LOGGER[] = "logger";
}; // namespace config

class Config {
private:
    std::shared_ptr<spdlog::logger> _logger, _fileLogger;

public:
    // Structures
    struct Images : BasicConfig {
        // Input images
        cv::Mat _transA, _transB, _simA, _simB, _check, _checkRot;

        Images() = default;
        Images(const YAML::Node& imagesNode);
    };

    struct Harris : BasicConfig {
        int _sobelKernelSize;
        size_t _windowSize, _minDistance;
        float _gaussianSigma, _alpha, _responseThresh;

        Harris() = default;
        Harris(const YAML::Node& harrisNode);
    };

    Images _images;
    bool _useGpu = false;
    // Settings for Harris operator for the transA and simA images, respectively
    Harris _harrisTrans, _harrisSim;

    // Path to which output images will be written
    std::string _outputPathPrefix;
    // Seed for mersenne twister engine
    std::unique_ptr<std::seed_seq> _mersenneSeed;

    bool _configDone = false;

    Config(const std::string& configFilePath);

    bool loadConfig(const YAML::Node& config);
};