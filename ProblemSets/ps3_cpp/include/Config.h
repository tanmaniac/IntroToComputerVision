#pragma once

// Structures to hold runtime configurations.
#include "FParse.h"

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
            auto tmpLogger = spdlog::get("logger");
            tmpLogger->error("Could not load param \"{}\"", key);
            return false;
        }
    };

public:
    // Structures
    struct Images : BasicConfig {
        // Paths to each of the input images
        std::string _picAPath, _picBPath;
        cv::Mat _picA, _picB;

        Images() = default;
        Images(const YAML::Node& imagesNode);

    private:
        // Load an left-right image pair from a YAML file and return whether or not it succeeded.
        bool loadImg(const YAML::Node& node,
                     const std::string& key,
                     std::string& imgPath,
                     cv::Mat& img);
    };

    struct Points : BasicConfig {
        // Vectors of column-vectors for each of the point sets
        std::vector<cv::Mat> _picA, _picB, _picANorm, _pts3D, _pts3DNorm;

        Points() = default;
        Points(const YAML::Node& node);

    private:
        // Load points from a text file
        template <typename T>
        bool loadPoints(const YAML::Node& node,
                        const std::string& key,
                        std::vector<cv::Mat>& points) {
            auto tmpLogger = spdlog::get("logger");
            if (node[key]) {
                std::string path = node[key].as<std::string>();
                points = FParse::parseAs<T>(path);
                if (!points.empty()) {
                    tmpLogger->info("Loaded points from {}", path);
                    return true;
                }
            }

            tmpLogger->error("Could not load points from param \"{}\"", key);
            return false;
        }
    };

    Images _images;
    Points _points;

    std::string _outputPathPrefix;

    bool _configDone = false;

    Config(const std::string& configFilePath);

    bool loadConfig(const YAML::Node& config);

private:
    bool makeDir(const std::string& dirPath);
};