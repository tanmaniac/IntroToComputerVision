#include "../include/Config.h"
#include <common/Utils.h>

#include <opencv2/imgcodecs.hpp>

#include <errno.h>
#include <sys/stat.h>
#include <iterator>
#include <sstream>

Config::Images::Images(const YAML::Node& imagesNode) {
    bool loadSuccess = true;
    auto tmpLogger = spdlog::get(config::FILE_LOGGER);
    loadSuccess = loadImg(imagesNode, "transA.jpg", _transA, tmpLogger);
    loadSuccess = loadImg(imagesNode, "transB.jpg", _transB, tmpLogger);
    loadSuccess = loadImg(imagesNode, "simA.jpg", _simA, tmpLogger);
    loadSuccess = loadImg(imagesNode, "simB.jpg", _simB, tmpLogger);
    loadSuccess = loadImg(imagesNode, "check.bmp", _check, tmpLogger);
    loadSuccess = loadImg(imagesNode, "check_rot.bmp", _checkRot, tmpLogger);

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

Config::Harris::Harris(const YAML::Node& harrisNode) {
    bool loadSuccess = true;
    auto tmpLogger = spdlog::get(config::FILE_LOGGER);
    loadSuccess = loadParam(harrisNode, "sobel_kernel_size", _sobelKernelSize, tmpLogger);
    loadSuccess = loadParam(harrisNode, "window_size", _windowSize, tmpLogger);
    loadSuccess = loadParam(harrisNode, "gaussian_sigma", _gaussianSigma, tmpLogger);
    loadSuccess = loadParam(harrisNode, "alpha", _alpha, tmpLogger);
    loadSuccess = loadParam(harrisNode, "response_threshold", _responseThresh, tmpLogger);
    loadSuccess = loadParam(harrisNode, "min_distance", _minDistance, tmpLogger);

    _configDone = loadSuccess;
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

    // Should we use GPU or CPU?
    if (config["use_gpu"]) {
        _useGpu = config["use_gpu"].as<bool>();
        _logger->info("Using {} for compute", _useGpu ? "GPU" : "CPU");
    }

    if (YAML::Node harrisNode = config["harris_trans"]) {
        _harrisTrans = Harris(harrisNode);
    }
    if (YAML::Node harrisNode = config["harris_sim"]) {
        _harrisSim = Harris(harrisNode);
    }

    bool configSuccess = true;
    // Verify that configurations were successful
    if (!_images.configDone()) {
        _logger->error("Loading images failed!");
        configSuccess = false;
    }
    if (!_harrisTrans.configDone()) {
        _logger->error("Could not load parameters for Harris operator of trans image set!");
        configSuccess = false;
    }
    if (!_harrisSim.configDone()) {
        _logger->error("Could not load parameters for Harris operator of sim image set!");
        configSuccess = false;
    }

    _configDone = configSuccess;
    return _configDone;
}
