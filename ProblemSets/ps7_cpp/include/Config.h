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
#include <unordered_map>

namespace config {
static constexpr char FILE_LOGGER[] = "file_logger";
static constexpr char STDOUT_LOGGER[] = "logger";
}; // namespace config

class Config {
public:
    Config(const std::string& configFilePath);

    bool loadConfig(const YAML::Node& config);

    // Expects short names, e.g. "PS7A1P1T1"
    cv::VideoCapture openVid(const std::string& name);

    // Expects short names, e.g. "PS7A1P1T1"
    size_t lastFrameOfAction(const std::string& name);

    // Path to which output images will be written
    std::string _outputPathPrefix;

    struct MHI : BasicConfig {
        double _threshold, _preBlurSigma;
        cv::Size _preBlurSize;
        int _tau;

        MHI() = default;
        MHI(const YAML::Node& node);
    };

    MHI _mhiAction1, _mhiAction2, _mhiAction3;

private:
    std::shared_ptr<spdlog::logger> _logger, _fileLogger;

    // Get the full paths for all the video files in a directory and place them in the map of video
    // files
    bool getVidFilesFromDir(const std::string& dir);

    // Load actions lengths for each sequence into a map
    bool loadActionLengths(const YAML::Node& actions);

    // Map matching file names with absolute file paths
    std::unordered_map<std::string, std::string> _vidMap;
    // Map matching file names with final frames of each action
    std::unordered_map<std::string, size_t> _lastFrames;
    // Set as true when configuration is done
    bool _configDone = false;
};