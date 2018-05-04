#include "../include/Config.h"
#include <common/Utils.h>

#include <spdlog/fmt/ostr.h>

#include <opencv2/imgcodecs.hpp>

#include <boost/filesystem.hpp>

#include <errno.h>
#include <sys/stat.h>
#include <iterator>
#include <sstream>

namespace fs = boost::filesystem;

bool Config::ImageSet::loadImFromDir(const std::string& dir) {
    auto tmpLogger = spdlog::get(config::FILE_LOGGER);

    fs::path path(dir);

    if (fs::exists(path)) {
        if (fs::is_directory(path)) {
            tmpLogger->info("Entering directory {}", dir);
            // Hold all the paths somewhere
            std::vector<fs::path> paths;
            // Find all the files in this directory
            for (const fs::directory_entry& entry : fs::directory_iterator(path)) {
                paths.push_back(entry.path());
            }
            // Sort paths to get them in alphabetical order
            std::sort(paths.begin(), paths.end());
            // For each entry in the directory, load it as an image and append it to the vector of
            // images
            for (const auto& entry : paths) {
                tmpLogger->info("   Reading {}", entry.native());
                _pics.push_back(cv::imread(entry.native(), cv::IMREAD_UNCHANGED));
            }
            return true;
        }
        tmpLogger->error("{} is not a directory", path.native());
        return false;
    }
    tmpLogger->error("{} is neither a regular file nor a directory", path.native());
    return false;
}

Config::ImageSet::ImageSet(const YAML::Node& node, const std::string& dirKey) {
    bool loadSuccess = true;
    auto tmpLogger = spdlog::get(config::FILE_LOGGER);
    std::string dir;
    if (loadParam(node, dirKey, dir, tmpLogger)) {
        loadSuccess = loadImFromDir(dir);
    }

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
    if (YAML::Node imagesNode = config["image_sets"]) {
        _yosemite = ImageSet(imagesNode, "yosemite");
        _pupper = ImageSet(imagesNode, "pupper");
        _juggle = ImageSet(imagesNode, "juggle");
        _shift = ImageSet(imagesNode, "shift");
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

        _mersenneSeed = std::make_shared<std::seed_seq>(seedVals.begin(), seedVals.end());
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

    bool configSuccess = true;
    // Verify that configurations were successful
    if (!_yosemite.configDone()) {
        _logger->error("Loading images from Yosemite data set failed!");
        configSuccess = false;
    }
    if (!_pupper.configDone()) {
        _logger->error("Could not load good boyes from Pupper data set!");
        configSuccess = false;
    }
    if (!_juggle.configDone()) {
        _logger->error("Loading images from Juggle data set failed!");
        configSuccess = false;
    }
    if (!_shift.configDone()) {
        _logger->error("Loading images from Shift data set failed!");
        configSuccess = false;
    }

    _configDone = configSuccess;
    return _configDone;
}
