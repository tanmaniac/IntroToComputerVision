#pragma once

// Structures to hold runtime configurations.
#include <common/BasicConfig.h>
#include <common/Utils.h>

#include <spdlog/spdlog.h>
#include <yaml-cpp/yaml.h>

#include <opencv2/core/core.hpp>

#include <cstddef>
#include <iostream>

namespace config {
static constexpr char FILE_LOGGER[] = "file_logger";
static constexpr char STDOUT_LOGGER[] = "logger";
}; // namespace config

class Config {
private:
    std::shared_ptr<spdlog::logger> _logger;

public:
    // Structures
    struct Images : BasicConfig {
        std::pair<cv::Mat, cv::Mat> _pair0, _pair1, _pair2, _pair1GT, _pair2GT;

        Images() = default;
        Images(const YAML::Node& imagesNode);

    private:
        // Load an left-right image pair from a YAML file and return whether or not it succeeded.
        bool loadImgPair(const YAML::Node& node,
                         const std::string& keyLeft,
                         const std::string& keyRight,
                         std::pair<cv::Mat, cv::Mat>& imgs);
    };

    struct DisparitySSD : BasicConfig {
        size_t _windowRadius;
        size_t _disparityRange;

        DisparitySSD() = default;
        DisparitySSD(const YAML::Node& ssdNode);
    };

    Images _images;
    bool _useGpuDisparity = false;
    DisparitySSD _p1disp, _p2disp, _p3disp, _p4disp, _p5disp;

    std::string _outputPathPrefix;

    bool _configDone = false;

    Config(const std::string& configFilePath);

    bool loadConfig(const YAML::Node& config);
};