#include "../include/Config.h"
#include <common/Utils.h>

#include <opencv2/imgcodecs.hpp>

#include <errno.h>
#include <sys/stat.h>
#include <iterator>
#include <sstream>

Config::Images::Images(const YAML::Node& imagesNode) {
    bool loadSuccess = true;
    auto tmpLogger = spdlog::get(config::STDOUT_LOGGER);
    loadSuccess = loadImg(imagesNode, "pic_a", _picA, tmpLogger);
    loadSuccess = loadImg(imagesNode, "pic_b", _picB, tmpLogger);

    _configDone = loadSuccess;
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