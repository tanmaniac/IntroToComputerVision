#pragma once

// Structures to hold runtime configurations.
#include <common/BasicConfig.h>
#include <common/Utils.h>

#include <yaml-cpp/yaml.h>
#include "spdlog/spdlog.h"

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
        cv::Mat _input0, _input0Noise, _input1, _input2, _input3;

        Images() = default;
        Images(const YAML::Node& imagesNode);
    };

    struct EdgeDetect : BasicConfig {
        size_t _gaussianSize;
        float _gaussianSigma;
        double _lowerThreshold, _upperThreshold;
        int _sobelApertureSize;

        EdgeDetect() = default;
        EdgeDetect(const YAML::Node& edgeDetectNode);
    };

    struct Hough : BasicConfig {
        unsigned int _numPeaks;
        int _threshold;
    };

    struct HoughLines : Hough {
        unsigned int _rhoBinSize, _thetaBinSize;
        HoughLines() = default;
        HoughLines(const YAML::Node& houghNode);
    };

    struct HoughCircles : Hough {
        unsigned int _minRadius, _maxRadius;
        HoughCircles() = default;
        HoughCircles(const YAML::Node& houghNode);
    };

    Images _images;
    EdgeDetect _p2Edge, _p3Edge, _p4Edge, _p5Edge, _p6Edge, _p7Edge, _p8Edge;
    HoughLines _p2Hough, _p3Hough, _p4Hough, _p6Hough, _p8HoughLines;
    HoughCircles _p5Hough, _p7Hough, _p8HoughCircles;

    std::string _outputPathPrefix;

    bool _configDone = false;

    Config(const std::string& configFilePath);

    bool loadConfig(const YAML::Node& config);
};