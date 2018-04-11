#include "../include/Config.h"

#include <opencv2/imgcodecs.hpp>

#include <errno.h>
#include <sys/stat.h>
#include <iterator>
#include <sstream>

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
    auto tmpLogger = spdlog::get(config::FILE_LOGGER);
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
    auto tmpLogger = spdlog::get(config::FILE_LOGGER);
    loadSuccess = loadPoints<float>(node, "pts2d_pic_a", _picA);
    loadSuccess = loadPoints<float>(node, "pts2d_pic_b", _picB);
    loadSuccess = loadPoints<float>(node, "pts2d_norm_pic_a", _picANorm);
    loadSuccess = loadPoints<float>(node, "pts3d", _pts3D);
    loadSuccess = loadPoints<float>(node, "pts3d_norm", _pts3DNorm);

    _configDone = loadSuccess;
}

Config::Config(const std::string& configFilePath) {
    _logger = spdlog::get(config::STDOUT_LOGGER);
    _fileLogger = spdlog::get(config::FILE_LOGGER);
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

    if (config["mersenne_seed"]) {
        // Load seed as a string of hex values
        std::istringstream seedString(config["mersenne_seed"].as<std::string>());

        uint32_t i;
        std::vector<uint32_t> seedVals;
        while (seedString >> std::hex >> i) {
            seedVals.push_back(i);
        }

        _mersenneSeed =
            std::unique_ptr<std::seed_seq>(new std::seed_seq(seedVals.begin(), seedVals.end()));
        std::stringstream ss;
        ss << std::hex;
        _mersenneSeed->param(std::ostream_iterator<uint32_t>(ss, " "));
        _fileLogger->info("Using random seed {}", ss.str());
    } else {
        _mersenneSeed = std::unique_ptr<std::seed_seq>(new std::seed_seq({1}));
        std::stringstream ss;
        _mersenneSeed->param(std::ostream_iterator<uint32_t>(ss, " "));
        _logger->warn("No random seed specified; using {}", ss.str());
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