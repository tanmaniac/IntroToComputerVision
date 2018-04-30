#include "../include/Config.h"

#include <opencv2/imgcodecs.hpp>

#include <errno.h>
#include <sys/stat.h>

Config::Images::Images(const YAML::Node& imagesNode) {
    bool loadSuccess = true;
    loadSuccess = loadImgPair(imagesNode, "pair0-L", "pair0-R", _pair0);
    loadSuccess = loadImgPair(imagesNode, "pair1-L", "pair1-R", _pair1);
    loadSuccess = loadImgPair(imagesNode, "pair2-L", "pair2-R", _pair2);
    loadSuccess = loadImgPair(imagesNode, "pair1-GT-L", "pair1-GT-R", _pair1GT);
    loadSuccess = loadImgPair(imagesNode, "pair2-GT-L", "pair2-GT-R", _pair2GT);

    _configDone = loadSuccess;
}

Config::DisparitySSD::DisparitySSD(const YAML::Node& ssdNode) {
    bool loadSuccess = true;
    loadSuccess = loadParam(ssdNode, "window_radius", _windowRadius);
    loadSuccess = loadParam(ssdNode, "disparity_range", _disparityRange);

    _configDone = loadSuccess;
}

bool Config::Images::loadImgPair(const YAML::Node& node,
                                 const std::string& keyLeft,
                                 const std::string& keyRight,
                                 std::pair<cv::Mat, cv::Mat>& imgs) {
    auto tmpLogger = spdlog::get(config::STDOUT_LOGGER);
    return (loadImg(node, keyLeft, imgs.first, tmpLogger) &&
            loadImg(node, keyRight, imgs.second, tmpLogger));
}

Config::Config(const std::string& configFilePath) {
    _logger = spdlog::get(config::STDOUT_LOGGER);
    YAML::Node config = YAML::LoadFile(configFilePath);
    if (config.IsNull()) {
        _logger->error("Could not load input file (was looking for {})", configFilePath);
        exit(-1);
    }

    if (loadConfig(config)) {
        _logger->info("Loaded runtime configuration from \"{}\"", configFilePath);
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

    // Should we use GPU for disparity, or just use CPU
    if (config["use_gpu_disparity"]) {
        _useGpuDisparity = config["use_gpu_disparity"].as<bool>();
        _logger->info("Using {} for disparity computation", _useGpuDisparity ? "GPU" : "CPU");
    }

    if (YAML::Node ssdNode = config["problem_1_ssd"]) {
        _p1disp = DisparitySSD(ssdNode);
    }

    if (YAML::Node ssdNode = config["problem_2_ssd"]) {
        _p2disp = DisparitySSD(ssdNode);
    }

    if (YAML::Node ssdNode = config["problem_3_ssd"]) {
        _p3disp = DisparitySSD(ssdNode);
    }

    if (YAML::Node ssdNode = config["problem_4_ncorr"]) {
        _p4disp = DisparitySSD(ssdNode);
    }

    if (YAML::Node node = config["problem_5_ncorr"]) {
        _p5disp = DisparitySSD(node);
    }

    bool configSuccess = true;
    // Verify that configurations were successful
    if (!_images.configDone()) {
        _logger->error("Loading images failed!");
        configSuccess = false;
    }
    if (!_p1disp.configDone()) {
        _logger->error("Loading Problem 1 parameters failed!");
        configSuccess = false;
    }
    if (!_p2disp.configDone()) {
        _logger->error("Loading Problem 2 parameters failed!");
        configSuccess = false;
    }
    if (!_p3disp.configDone()) {
        _logger->error("Loading Problem 3 parameters failed!");
        configSuccess = false;
    }
    if (!_p4disp.configDone()) {
        _logger->error("Loading Problem 4 parameters failed!");
        configSuccess = false;
    }
    if (!_p5disp.configDone()) {
        _logger->error("Loading Problem 5 parameters failed!");
        configSuccess = false;
    }

    _configDone = configSuccess;
    return _configDone;
}
