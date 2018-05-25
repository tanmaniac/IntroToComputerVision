#include "../include/Config.h"
#include <common/Utils.h>

#include <spdlog/fmt/ostr.h>

#include <boost/filesystem.hpp>
#include <opencv2/imgcodecs.hpp>

#include <errno.h>
#include <sys/stat.h>
#include <fstream>
#include <iostream>
#include <iterator>
#include <sstream>

namespace fs = boost::filesystem;

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

Config::MHI::MHI(const YAML::Node& node) {
    bool loadSuccess = true;
    auto tmpLogger = spdlog::get(config::FILE_LOGGER);

    loadSuccess = loadParam(node, "diff_threshold", _threshold, tmpLogger);
    int blurSize;
    loadSuccess = loadParam(node, "pre_blur_size", blurSize, tmpLogger);
    _preBlurSize = cv::Size(blurSize, blurSize);
    loadSuccess = loadParam(node, "pre_blur_sigma", _preBlurSigma, tmpLogger);
    loadSuccess = loadParam(node, "tau", _tau, tmpLogger);

    _configDone = loadSuccess;
}

bool Config::loadActionLengths(const YAML::Node& actions) {
    // actions is structured as a map of maps of maps of sequences
    for (int action = 1; action <= actions.size(); action++) {
        auto curAction = actions["action" + std::to_string(action)];
        for (int person = 1; person <= curAction.size(); person++) {
            auto curPerson = curAction["person" + std::to_string(person)];
            int trial = 1;
            for (auto framesIter = curPerson.begin(); framesIter != curPerson.end(); framesIter++) {
                int frameLength = framesIter->as<int>();
                _lastFrames.insert({"PS7A" + std::to_string(action) + "P" + std::to_string(person) +
                                        "T" + std::to_string(trial),
                                    frameLength});
                trial++;
            }
        }
    }
    return true;
}

bool Config::getVidFilesFromDir(const std::string& dir) {
    // Enter directory
    fs::path path(dir);

    if (fs::exists(path)) {
        if (fs::is_directory(path)) {
            _fileLogger->info("Entered directory {}", dir);
            // Iterate over the files in the directoy
            for (const fs::directory_entry& entry : fs::directory_iterator(path)) {
                // Only get .avi files
                if (entry.path().extension().compare(".avi") == 0) {
                    _vidMap.insert({entry.path().stem().string(), entry.path().string()});
                    _fileLogger->info(
                        "Added {} with key {} to map", entry.path(), entry.path().stem());
                }
            }
            return true;
        }
    }
    _logger->error("Input directory does not exist or is a regular file");
    return false;
}

cv::VideoCapture Config::openVid(const std::string& name) {
    return cv::VideoCapture(_vidMap.at(name));
}

size_t Config::lastFrameOfAction(const std::string& name) { return _lastFrames.at(name); }

bool Config::loadConfig(const YAML::Node& config) {
    // Load input videos
    if (config["input_dir"]) {
        std::string inputDir = config["input_dir"].as<std::string>();
        if (getVidFilesFromDir(inputDir)) {
            _logger->info("Found input videos in dir \"{}\"", inputDir);
        } else {
            _logger->error("Failed to load input videos from \"{}\"", inputDir);
            return false;
        }
    }

    if (YAML::Node actionLastFrame = config["last_frame_of_action"]) {
        if (!loadActionLengths(actionLastFrame)) {
            _logger->error("Failed to load last frame of each action from configuration file");
            return false;
        }
    } else {
        _logger->error("No last frame specified for any actions");
        return false;
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

    // Load motion history settings
    if (YAML::Node node = config["mhi_action1"]) {
        _mhiAction1 = MHI(node);
    }
    if (YAML::Node node = config["mhi_action2"]) {
        _mhiAction2 = MHI(node);
    }
    if (YAML::Node node = config["mhi_action3"]) {
        _mhiAction3 = MHI(node);
    }

    bool configSuccess = true;
    // Verify that configurations were successful
    if (!_mhiAction1.configDone()) {
        _logger->error("Could not load MHI configuration for action 1!");
        configSuccess = false;
    }
    if (!_mhiAction2.configDone()) {
        _logger->error("Could not load MHI configuration for action 2!");
        configSuccess = false;
    }
    if (!_mhiAction3.configDone()) {
        _logger->error("Could not load MHI configuration for action 3!");
        configSuccess = false;
    }

    _configDone = configSuccess;
    return _configDone;
}
