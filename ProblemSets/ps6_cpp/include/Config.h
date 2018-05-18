#pragma once

// Structures to hold runtime configurations.
#include <common/BasicConfig.h>
#include <common/Utils.h>

#include <spdlog/spdlog.h>
#include <yaml-cpp/yaml.h>

#include <opencv2/core/core.hpp>
#include <opencv2/videoio/videoio.hpp>

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
    struct Tracking : BasicConfig {
        cv::VideoCapture _cap; // Video capture stream
        cv::Point2f _bbox;     // (x, y) coordinates of the initial bounding box
        cv::Size2f _bboxSize;  // width and height of the initial bounding box

        Tracking() = default;
        Tracking(const YAML::Node& node);

    private:
        // Load bounding box (x, y) coordinates and width/height from a text file
        bool loadBBox(const std::string& filename);
    };

    struct PFConf : BasicConfig {
        double _mseSigma, _dynamicsSigma, _alpha;
        size_t _numParticles;

        PFConf() = default;
        PFConf(const YAML::Node& node);
    };

    Tracking _debate, _noisyDebate, _pedestrians;
    PFConf _pfConf1, _pfConf1Noisy, _pfConf2, _pfConf2Noisy, _pfConf3Head, _pfConf3Hand;
    bool _useGpu = false;

    // Path to which output images will be written
    std::string _outputPathPrefix;

    bool _configDone = false;

    Config(const std::string& configFilePath);

    bool loadConfig(const YAML::Node& config);
};