#pragma once

// Structures to hold runtime configurations.
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

    struct BasicConfig {
        bool configDone();

    protected:
        bool _configDone = false;

        template <typename T>
        bool loadParam(const YAML::Node& node, const std::string& key, T& val) {
            if (node[key]) {
                val = node[key].as<T>();
                return true;
            }
            auto tmpLogger = spdlog::get(config::STDOUT_LOGGER);
            tmpLogger->error("Could not load param \"{}\"", key);
            return false;
        }
    };

public:
    // Structures
    struct Images : BasicConfig {
        // Input images
        cv::Mat _transA, _transB, _simA, _simB, _check, _checkRot;

        Images() = default;
        Images(const YAML::Node& imagesNode);

    private:
        // Load image from a YAML file and return whether or not it succeeded.
        bool loadImg(const YAML::Node& node, const std::string& key, cv::Mat& img);
    };

    struct Harris : BasicConfig {
        int _sobelKernelSize;

        Harris() = default;
        Harris(const YAML::Node& harrisNode);
    };

    Images _images;
    // Settings for Harris operator for the transA and simA images, respectively
    Harris _harrisTrans, _harrisSim;

    // Path to which output images will be written
    std::string _outputPathPrefix;
    // Seed for mersenne twister engine
    std::unique_ptr<std::seed_seq> _mersenneSeed;

    bool _configDone = false;

    Config(const std::string& configFilePath);

    bool loadConfig(const YAML::Node& config);

private:
    bool makeDir(const std::string& dirPath);
};