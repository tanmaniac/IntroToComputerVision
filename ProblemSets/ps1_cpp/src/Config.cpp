#include "Config.h"

#include <opencv2/imgcodecs.hpp>

#include <errno.h>
#include <sys/stat.h>

bool Config::BasicConfig::configDone() {
    return _configDone;
}

bool Config::BasicConfig::loadImgFromYAML(const YAML::Node& node,
                                          const std::string& key,
                                          std::string imgPath,
                                          cv::Mat& img) {
    bool loadSuccess = false;
    if (node[key]) {
        imgPath = node[key].as<std::string>();
        img = cv::imread(imgPath, cv::IMREAD_UNCHANGED);
        if (!img.empty()) {
            loadSuccess = true;
        } else {
            std::cerr << "Could not load image \"" << key << "\" at \"" << imgPath << "\""
                      << std::endl;
        }
    }

    return loadSuccess;
}

Config::Images::Images(const YAML::Node& imagesNode) {
    bool loadSuccess = true;
    loadSuccess = loadImgFromYAML(imagesNode, "input0", _input0Path, _input0);
    _input0.convertTo(_input0, CV_32FC1);
    loadSuccess = loadImgFromYAML(imagesNode, "input0_noise", _input0NoisePath, _input0Noise);
    _input0Noise.convertTo(_input0Noise, CV_32FC1);
    loadSuccess = loadImgFromYAML(imagesNode, "input1", _input1Path, _input1);
    loadSuccess = loadImgFromYAML(imagesNode, "input2", _input2Path, _input2);
    loadSuccess = loadImgFromYAML(imagesNode, "input3", _input3Path, _input3);

    _configDone = loadSuccess;
}

Config::EdgeDetect::EdgeDetect(const YAML::Node& edgeDetectNode) {
    bool loadSuccess = true;
    loadSuccess = loadParam(edgeDetectNode, "gaussian_size", _gaussianSize);
    loadSuccess = loadParam(edgeDetectNode, "gaussian_sigma", _gaussianSigma);
    loadSuccess = loadParam(edgeDetectNode, "lower_threshold", _lowerThreshold);
    loadSuccess = loadParam(edgeDetectNode, "upper_threshold", _upperThreshold);
    loadSuccess = loadParam(edgeDetectNode, "sobel_aperture_size", _sobelApertureSize);

    _configDone = loadSuccess;
}

Config::HoughLines::HoughLines(const YAML::Node& houghNode) {
    bool loadSuccess = true;
    loadSuccess = loadParam(houghNode, "rho_bin_size", _rhoBinSize);
    loadSuccess = loadParam(houghNode, "theta_bin_size", _thetaBinSize);
    loadSuccess = loadParam(houghNode, "threshold", _threshold);
    loadSuccess = loadParam(houghNode, "num_peaks", _numPeaks);

    _configDone = loadSuccess;
}

Config::HoughCircles::HoughCircles(const YAML::Node& houghNode) {
    bool loadSuccess = true;
    loadSuccess = loadParam(houghNode, "min_radius", _minRadius);
    loadSuccess = loadParam(houghNode, "max_radius", _maxRadius);
    loadSuccess = loadParam(houghNode, "threshold", _threshold);
    loadSuccess = loadParam(houghNode, "num_peaks", _numPeaks);

    _configDone = loadSuccess;
}

Config::Config(const std::string& configFilePath) {
    _logger = spdlog::get("logger");
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
        if (makeDir(_outputPathPrefix)) {
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

    _configDone = configSuccess;
    return _configDone;
}

bool Config::makeDir(const std::string& dirPath) {
    const int dirErr = mkdir(dirPath.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
    if (dirErr == -1) {
        // The directory already exists, so there's nothing to do anyway. Return true
        return errno == EEXIST;
    }
    return true;
}