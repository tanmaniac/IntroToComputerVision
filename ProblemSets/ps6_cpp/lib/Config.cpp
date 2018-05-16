#include "../include/Config.h"
#include <common/Utils.h>

#include <spdlog/fmt/ostr.h>

#include <opencv2/imgcodecs.hpp>

#include <boost/lexical_cast.hpp>
#include <boost/tokenizer.hpp>

#include <errno.h>
#include <sys/stat.h>
#include <fstream>
#include <iostream>
#include <iterator>
#include <sstream>

Config::Tracking::Tracking(const YAML::Node& node) {
    bool loadSuccess = true;
    auto tmpLogger = spdlog::get(config::FILE_LOGGER);
    std::string video, bbox;
    if (loadParam(node, "video", video, tmpLogger)) {
        _cap.open(video);
        if (!_cap.isOpened()) {
            tmpLogger->error("Failed to open video stream at \"{}\"", video);
            loadSuccess = false;
        }
    } else {
        tmpLogger->error("Could not find YAML parameter \"video\"");
        loadSuccess = false;
    }

    if (loadParam(node, "bounding_box", bbox, tmpLogger)) {
        if (loadBBox(bbox)) {
            tmpLogger->info("Loaded bounding box with coordinates ({}, {}); width={}; height={}",
                            _bbox.x,
                            _bbox.y,
                            _bboxSize.width,
                            _bboxSize.height);
        } else {
            tmpLogger->error("Failed to load bounding box from file {}", bbox);
            loadSuccess = false;
        }
    } else {
        tmpLogger->error("Could not find YAML parameter \"bounding_box\"");
        loadSuccess = false;
    }

    _configDone = loadSuccess;
}

bool Config::Tracking::loadBBox(const std::string& filename) {
    auto tmpLogger = spdlog::get(config::FILE_LOGGER);

    typedef boost::tokenizer<boost::char_separator<char>> tokenizer;
    boost::char_separator<char> separator(" ");

    // Open input file
    std::ifstream input(filename);
    if (input.is_open()) {
        std::string line;
        // The first line is the (x, y) coordinate of the top-left corner of the bounding box
        if (std::getline(input, line)) {
            tokenizer tokens(line, separator);
            std::vector<float> coords;
            for (auto tokenIter = tokens.begin(); tokenIter != tokens.end(); tokenIter++) {
                auto val = boost::lexical_cast<float>(*tokenIter);
                coords.push_back(val);
            }
            if (coords.size() != 2) {
                tmpLogger->error("Input file has invalid format: expected 2 values on each line");
                return false;
            }
            _bbox = cv::Point2f(coords[0], coords[1]);
        } else {
            tmpLogger->error("Could not read first line of input file {}", filename);
            return false;
        }

        // Second line is the width and height of the bounding box
        if (std::getline(input, line)) {
            tokenizer tokens(line, separator);
            std::vector<float> size;
            for (auto tokenIter = tokens.begin(); tokenIter != tokens.end(); tokenIter++) {
                auto val = boost::lexical_cast<float>(*tokenIter);
                size.push_back(val);
            }
            if (size.size() != 2) {
                tmpLogger->error("Input file has invalid format: expected 2 values on each line");
                return false;
            }
            _bboxSize = cv::Size2f(size[0], size[1]);
        } else {
            tmpLogger->error("Could not read second line of input file {}", filename);
            return false;
        }
    } else {
        tmpLogger->error("Could not open input file {}", filename);
        return false;
    }

    return true;
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
    // Load input videos
    if (YAML::Node videoNode = config["pres_debate"]) {
        _debate = Tracking(videoNode);
    }

    if (YAML::Node videoNode = config["noisy_debate"]) {
        _noisyDebate = Tracking(videoNode);
    }

    if (YAML::Node videoNode = config["pedestrians"]) {
        _pedestrians = Tracking(videoNode);
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
    if (!_debate.configDone()) {
        _logger->error("Loading input for debate data set failed!");
        configSuccess = false;
    }
    if (!_noisyDebate.configDone()) {
        _logger->error("Loading input for noisy debate data set failed!");
        configSuccess = false;
    }
    if (!_pedestrians.configDone()) {
        _logger->error("Loading input for pedestrians data set failed!");
        configSuccess = false;
    }

    _configDone = configSuccess;
    return _configDone;
}
