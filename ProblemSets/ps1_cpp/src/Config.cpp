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
    loadSuccess = loadImgFromYAML(imagesNode, "input2", _input1Path, _input1);
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

Config::Hough::Hough(const YAML::Node& houghNode) {
    bool loadSuccess = true;
    loadSuccess = loadParam(houghNode, "rho_bin_size", _rhoBinSize);
    loadSuccess = loadParam(houghNode, "theta_bin_size", _thetaBinSize);
    loadSuccess = loadParam(houghNode, "threshold", _threshold);
    loadSuccess = loadParam(houghNode, "num_peaks", _numPeaks);

    _configDone = loadSuccess;
}

Config::Config(const std::string& configFilePath) {
    YAML::Node config = YAML::LoadFile(configFilePath);
    if (config.IsNull()) {
        std::cerr << "Could not load input file (was looking for " << configFilePath << ")"
                  << std::endl;
        return;
    }

    if (loadConfig(config)) {
        std::cout << "Loaded runtime configuration from " << configFilePath << std::endl;
    } else {
        std::cerr << "Configuration load failed!" << std::endl;
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
            std::cout << "Created output directory at \"" << _outputPathPrefix << "\"" << std::endl;
            madeOutputDir = true;
        }
    }
    if (!madeOutputDir) {
        _outputPathPrefix = "./";
        std::cout << "No output path specified or could not make new directory; using current "
                     "directory"
                  << std::endl;
    }

    if (YAML::Node edgeDetectNode = config["edge_detector_p2"]) {
        _p2Edge = EdgeDetect(edgeDetectNode);
    }

    if (YAML::Node houghNode = config["hough_transform_p2"]) {
        _p2Hough = Hough(houghNode);
    }

    if (YAML::Node edgeDetectNode = config["edge_detector_p3"]) {
        _p3Edge = EdgeDetect(edgeDetectNode);
    }

    if (YAML::Node houghNode = config["hough_transform_p3"]) {
        _p3Hough = Hough(houghNode);
    }

    bool configSuccess = true;
    // Verify that configurations were successful
    if (!_images.configDone()) {
        std::cerr << "Loading images failed!" << std::endl;
        configSuccess = false;
    }
    if (!_p2Edge.configDone()) {
        std::cerr << "Loading edge detection parameters for Problem 2 failed!" << std::endl;
        configSuccess = false;
    }
    if (!_p2Hough.configDone()) {
        std::cerr << "Loading Hough transform parameters for Problem 2 failed!" << std::endl;
        configSuccess = false;
    }
    if (!_p3Edge.configDone()) {
        std::cerr << "Loading edge detection parameters for Problem 3 failed!" << std::endl;
        configSuccess = false;
    }
    if (!_p3Hough.configDone()) {
        std::cerr << "Loading Hough transform parameters for Problem 3 failed!" << std::endl;
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