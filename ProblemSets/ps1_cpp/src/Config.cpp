#include "Config.h"

#include <opencv2/imgcodecs.hpp>

#include <errno.h>
#include <sys/stat.h>

Config::Images::Images(const YAML::Node& imagesNode) {
    bool loadSuccess = true;
    auto logger = spdlog::get(config::STDOUT_LOGGER);
    loadSuccess = loadImg(imagesNode, "input0", _input0, logger);
    _input0.convertTo(_input0, CV_32FC1);
    loadSuccess = loadImg(imagesNode, "input0_noise", _input0Noise, logger);
    _input0Noise.convertTo(_input0Noise, CV_32FC1);
    loadSuccess = loadImg(imagesNode, "input1", _input1, logger);
    loadSuccess = loadImg(imagesNode, "input2", _input2, logger);
    loadSuccess = loadImg(imagesNode, "input3", _input3, logger);

    _configDone = loadSuccess;
}

Config::EdgeDetect::EdgeDetect(const YAML::Node& edgeDetectNode) {
    bool loadSuccess = true;
    auto logger = spdlog::get(config::STDOUT_LOGGER);
    loadSuccess = loadParam(edgeDetectNode, "gaussian_size", _gaussianSize, logger);
    loadSuccess = loadParam(edgeDetectNode, "gaussian_sigma", _gaussianSigma, logger);
    loadSuccess = loadParam(edgeDetectNode, "lower_threshold", _lowerThreshold, logger);
    loadSuccess = loadParam(edgeDetectNode, "upper_threshold", _upperThreshold, logger);
    loadSuccess = loadParam(edgeDetectNode, "sobel_aperture_size", _sobelApertureSize, logger);

    _configDone = loadSuccess;
}

Config::HoughLines::HoughLines(const YAML::Node& houghNode) {
    bool loadSuccess = true;
    auto logger = spdlog::get(config::STDOUT_LOGGER);
    loadSuccess = loadParam(houghNode, "rho_bin_size", _rhoBinSize, logger);
    loadSuccess = loadParam(houghNode, "theta_bin_size", _thetaBinSize, logger);
    loadSuccess = loadParam(houghNode, "threshold", _threshold, logger);
    loadSuccess = loadParam(houghNode, "num_peaks", _numPeaks, logger);

    _configDone = loadSuccess;
}

Config::HoughCircles::HoughCircles(const YAML::Node& houghNode) {
    bool loadSuccess = true;
    auto logger = spdlog::get(config::STDOUT_LOGGER);
    loadSuccess = loadParam(houghNode, "min_radius", _minRadius, logger);
    loadSuccess = loadParam(houghNode, "max_radius", _maxRadius, logger);
    loadSuccess = loadParam(houghNode, "threshold", _threshold, logger);
    loadSuccess = loadParam(houghNode, "num_peaks", _numPeaks, logger);

    _configDone = loadSuccess;
}

Config::Config(const std::string& configFilePath) {
    _logger = spdlog::get(config::STDOUT_LOGGER);
    YAML::Node config = YAML::LoadFile(configFilePath);
    if (config.IsNull()) {
        _logger->error("Could not load input file (was looking for {})", configFilePath);
        exit(-1);
    }

    if (loadConfig(config)) {
        _logger->info("Loaded runtime configuration from {}", configFilePath);
    } else {
        _logger->error("Configuration load failed!");
        exit(-1);
    }
}

bool Config::loadConfig(const YAML::Node& config) {
    // Load images
    if (YAML::Node imagesNode = config["images"]) {
        _images = Images(imagesNode);
    }

    bool madeOutputDir = false;
    // Set output path prefix
    if (config["output_dir"]) {
        _outputPathPrefix = config["output_dir"].as<std::string>();
        if (common::makeDir(_outputPathPrefix)) {
            _logger->info("Created output directory at \"{}\"", _outputPathPrefix);
            madeOutputDir = true;
        }
    }
    if (!madeOutputDir) {
        _outputPathPrefix = "./";
        _logger->warn(
            "No output path specified or could not make new directory; using current directory");
    }

    if (YAML::Node edgeDetectNode = config["edge_detector_p2"]) {
        _p2Edge = EdgeDetect(edgeDetectNode);
    }

    if (YAML::Node houghNode = config["hough_transform_p2"]) {
        _p2Hough = HoughLines(houghNode);
    }

    if (YAML::Node edgeDetectNode = config["edge_detector_p3"]) {
        _p3Edge = EdgeDetect(edgeDetectNode);
    }

    if (YAML::Node houghNode = config["hough_transform_p3"]) {
        _p3Hough = HoughLines(houghNode);
    }

    if (YAML::Node edgeDetectNode = config["edge_detector_p4"]) {
        _p4Edge = EdgeDetect(edgeDetectNode);
    }

    if (YAML::Node houghNode = config["hough_transform_p4"]) {
        _p4Hough = HoughLines(houghNode);
    }

    if (YAML::Node edgeDetectNode = config["edge_detector_p5"]) {
        _p5Edge = EdgeDetect(edgeDetectNode);
    }

    if (YAML::Node houghNode = config["hough_circle_transform_p5"]) {
        _p5Hough = HoughCircles(houghNode);
    }

    if (YAML::Node edgeDetectNode = config["edge_detector_p6"]) {
        _p6Edge = EdgeDetect(edgeDetectNode);
    }

    if (YAML::Node houghNode = config["hough_transform_p6"]) {
        _p6Hough = HoughLines(houghNode);
    }

    if (YAML::Node edgeDetectNode = config["edge_detector_p7"]) {
        _p7Edge = EdgeDetect(edgeDetectNode);
    }

    if (YAML::Node houghNode = config["hough_circle_transform_p7"]) {
        _p7Hough = HoughCircles(houghNode);
    }

    if (YAML::Node edgeDetectNode = config["edge_detector_p8"]) {
        _p8Edge = EdgeDetect(edgeDetectNode);
    }

    if (YAML::Node houghNode = config["hough_circle_transform_p8"]) {
        _p8HoughCircles = HoughCircles(houghNode);
    }

    if (YAML::Node houghNode = config["hough_line_transform_p8"]) {
        _p8HoughLines = HoughLines(houghNode);
    }

    bool configSuccess = true;
    // Verify that configurations were successful
    if (!_images.configDone()) {
        _logger->error("Loading images failed!");
        configSuccess = false;
    }
    if (!_p2Edge.configDone()) {
        _logger->error("Loading edge detection parameters for Problem 2 failed!");
        configSuccess = false;
    }
    if (!_p2Hough.configDone()) {
        _logger->error("Loading Hough transform parameters for Problem 2 failed!");
        configSuccess = false;
    }
    if (!_p3Edge.configDone()) {
        _logger->error("Loading edge detection parameters for Problem 3 failed!");
        configSuccess = false;
    }
    if (!_p3Hough.configDone()) {
        _logger->error("Loading Hough transform parameters for Problem 3 failed!");
        configSuccess = false;
    }
    if (!_p4Edge.configDone()) {
        _logger->error("Loading edge detection parameters for Problem 4 failed!");
        configSuccess = false;
    }
    if (!_p4Hough.configDone()) {
        _logger->error("Loading Hough transform parameters for Problem 4 failed!");
        configSuccess = false;
    }
    if (!_p5Edge.configDone()) {
        _logger->error("Loading edge detection parameters for Problem 5 failed!");
        configSuccess = false;
    }
    if (!_p5Hough.configDone()) {
        _logger->error("Loading Hough transform parameters for Problem 5 failed!");
        configSuccess = false;
    }
    if (!_p6Edge.configDone()) {
        _logger->error("Loading edge detection parameters for Problem 6 failed!");
        configSuccess = false;
    }
    if (!_p6Hough.configDone()) {
        _logger->error("Loading Hough transform parameters for Problem 6 failed!");
        configSuccess = false;
    }
    if (!_p7Edge.configDone()) {
        _logger->error("Loading edge detection parameters for Problem 7 failed!");
        configSuccess = false;
    }
    if (!_p7Hough.configDone()) {
        _logger->error("Loading Hough transform parameters for Problem 7 failed!");
        configSuccess = false;
    }
    if (!_p8Edge.configDone()) {
        _logger->error("Loading edge detection parameters for Problem 8 failed!");
        configSuccess = false;
    }
    if (!_p8HoughCircles.configDone()) {
        _logger->error("Loading Hough circle transform parameters for Problem 8 failed!");
        configSuccess = false;
    }
    if (!_p8HoughLines.configDone()) {
        _logger->error("Loading Hough line transform parameters for Problem 8 failed!");
        configSuccess = false;
    }

    _configDone = configSuccess;
    return _configDone;
}
