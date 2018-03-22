#include "../include/Config.h"

#include <opencv2/imgcodecs.hpp>

#include <errno.h>
#include <sys/stat.h>

bool Config::BasicConfig::configDone() {
    return _configDone;
}

Config::Images::Images(const YAML::Node& imagesNode) {
    bool loadSuccess = true;
    loadSuccess = loadImgPair(imagesNode, "pair0-L", "pair0-R", _pair0Paths, _pair0);
    loadSuccess = loadImgPair(imagesNode, "pair1-L", "pair1-R", _pair1Paths, _pair1);
    loadSuccess = loadImgPair(imagesNode, "pair2-L", "pair2-R", _pair2Paths, _pair2);
    loadSuccess = loadImgPair(imagesNode, "pair1-GT-L", "pair1-GT-R", _pair1GTPaths, _pair1GT);
    loadSuccess = loadImgPair(imagesNode, "pair2-GT-L", "pair2-GT-R", _pair2GTPaths, _pair2GT);

    _configDone = loadSuccess;
}

bool Config::Images::loadImgPair(const YAML::Node& node,
                                 const std::string& keyLeft,
                                 const std::string& keyRight,
                                 std::pair<std::string, std::string>& imgPaths,
                                 std::pair<cv::Mat, cv::Mat>& imgs) {
    bool loadSuccess = false;
    auto tmpLogger = spdlog::get("logger");
    if (node[keyLeft] && node[keyRight]) {
        imgPaths =
            std::make_pair(node[keyLeft].as<std::string>(), node[keyRight].as<std::string>());
        imgs = std::make_pair(cv::imread(imgPaths.first, cv::IMREAD_UNCHANGED),
                              cv::imread(imgPaths.second, cv::IMREAD_UNCHANGED));
        if (!imgs.first.empty() && !imgs.second.empty()) {
            loadSuccess = true;
        } else {
            tmpLogger->error(
                "Could not load image(s) \"{}\", \"{}\"", imgPaths.first, imgPaths.second);
        }
    } else {
        tmpLogger->error("Could not find YAML key(s) \"{}\", \"{}\"", keyLeft, keyRight);
    }

    return loadSuccess;
}

Config::Config(const std::string& configFilePath) {
    _logger = spdlog::get("logger");
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

    // Should we use GPU for disparity, or just use CPU
    if (config["use_gpu_disparity"]) {
        _useGpuDisparity = config["use_gpu_disparity"].as<bool>();
        _logger->info("Using GPU for disparity computation");
    }

    bool configSuccess = true;
    // Verify that configurations were successful
    if (!_images.configDone()) {
        _logger->error("Loading images failed!");
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