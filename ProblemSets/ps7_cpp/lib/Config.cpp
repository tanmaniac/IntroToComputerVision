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

    // Should we use GPU or CPU?
    if (config["use_gpu"]) {
        _useGpu = config["use_gpu"].as<bool>();
        _logger->info("Using {} for compute", _useGpu ? "GPU" : "CPU");
    }

    bool configSuccess = true;
    // Verify that configurations were successful

    _configDone = configSuccess;
    return _configDone;
}
