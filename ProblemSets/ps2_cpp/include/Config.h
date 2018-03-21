#pragma once

// Structures to hold runtime configurations.
#include <spdlog/spdlog.h>
#include <yaml-cpp/yaml.h>

#include <opencv2/core/core.hpp>

#include <cstddef>
#include <iostream>

class Config {
private:
    std::shared_ptr<spdlog::logger> _logger;

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
            auto tmpLogger = spdlog::get(_loggerName);
            tmpLogger->error("Could not load param \"{}\"", key);
            return false;
        }
    };

public:
    // Structures
    struct Images : BasicConfig {
        // Paths to each of the input images
        std::pair<std::string, std::string> _pair0Paths, _pair1Paths, _pair2Paths, _pair1GTPaths,
            _pair2GTPaths;
        std::pair<cv::Mat, cv::Mat> _pair0, _pair1, _pair2, _pair1GT, _pair2GT;

        Images() = default;
        Images(const YAML::Node& imagesNode);

    private:
        // Load an left-right image pair from a YAML file and return whether or not it succeeded.
        bool loadImgPair(const YAML::Node& node,
                         const std::string& keyLeft,
                         const std::string& keyRight,
                         std::pair<std::string, std::string>& imgPaths,
                         std::pair<cv::Mat, cv::Mat>& imgs);
    };

    const std::string _loggerName = "logger";

    Images _images;

    std::string _outputPathPrefix;

    bool _configDone = false;

    Config(const std::string& configFilePath);

    bool loadConfig(const YAML::Node& config);

private:
    bool makeDir(const std::string& dirPath);
};