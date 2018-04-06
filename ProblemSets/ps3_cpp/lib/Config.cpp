#include "../include/Config.h"

#include <opencv2/imgcodecs.hpp>

#include <errno.h>
#include <sys/stat.h>

bool Config::BasicConfig::configDone() {
    return _configDone;
}

Config::Images::Images(const YAML::Node& imagesNode) {
    bool loadSuccess = true;
    loadSuccess = loadImg(imagesNode, "pic_a", _picAPath, _picA);
    loadSuccess = loadImg(imagesNode, "pic_b", _picBPath, _picB);

    _configDone = loadSuccess;
}

bool Config::Images::loadImg(const YAML::Node& node,
                             const std::string& key,
                             std::string& imgPath,
                             cv::Mat& img) {
    bool loadSuccess = false;
    auto tmpLogger = spdlog::get(FILE_LOGGER);
    if (node[key]) {
        imgPath = node[key].as<std::string>();
        img = cv::imread(imgPath, cv::IMREAD_UNCHANGED);
        if (!img.empty()) {
            loadSuccess = true;
        } else {
            tmpLogger->error("Could not load image \"{}\"", imgPath);
        }
    } else {
        tmpLogger->error("Could not find YAML key \"{}\"", key);
    }
    tmpLogger->info("Loaded image from {}", imgPath);
    return loadSuccess;
}

Config::Points::Points(const YAML::Node& node) {
    bool loadSuccess = true;
    auto tmpLogger = spdlog::get(FILE_LOGGER);
    loadSuccess = loadPoints<float>(node, "pts2d_pic_a", _picA);
    loadSuccess = loadPoints<float>(node, "pts2d_pic_b", _picB);
    loadSuccess = loadPoints<float>(node, "pts2d_norm_pic_a", _picANorm);
    loadSuccess = loadPoints<float>(node, "pts3d", _pts3D);
    loadSuccess = loadPoints<float>(node, "pts3d_norm", _pts3DNorm);

    _configDone = loadSuccess;
}

Config::Config(const std::string& configFilePath) {
    _logger = spdlog::get(STDOUT_LOGGER);
    _fileLogger = spdlog::get(FILE_LOGGER);
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

    // Load points
    if (YAML::Node node = config["points"]) {
        _points = Points(node);
    }

    bool configSuccess = true;
    // Verify that configurations were successful
    if (!_images.configDone()) {
        _logger->error("Loading images failed!");
        configSuccess = false;
    }
    if (!_points.configDone()) {
        _logger->error("Loading points failed!");
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