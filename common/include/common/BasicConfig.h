#pragma once

#include <spdlog/spdlog.h>
#include <yaml-cpp/yaml.h>

#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs.hpp>

#include <memory>

// Basic configuration container to load runtime parameters from YAML files

struct BasicConfig {
protected:
    bool _configDone = false;

public:
    bool configDone() {
        return _configDone;
    }

    /**
     * \brief loadParam Loads a parameter from a YAML node and optionally logs if it was successful.
     *
     * \param node Input YAML-CPP node
     * \param key Key in YAML file from which the parameter should be loaded
     * \param val Var to which the value should be loaded
     * \param logger Shared pointer to a spdlog logger
     */
    template <typename T>
    bool loadParam(const YAML::Node& node,
                   const std::string& key,
                   T& val,
                   std::shared_ptr<spdlog::logger> logger = nullptr) {
        if (node[key]) {
            val = node[key].as<T>();
            return true;
        }
        if (logger) {
            logger->error("Could not load param \"{}\"", key);
        }
        return false;
    }

    /**
     * \brief loadImg Loads an image from a path defined in a YAML file. Images are loaded
     * unchanged.
     *
     * \param node Input YAML-CPP node
     * \param key Key in YAML file from which the image path is loaded
     * \param img OpenCV matrix to which the image is loaded
     * \param logger Optional shared pointer to a spdlog logger
     */
    bool loadImg(const YAML::Node& node,
                 const std::string& key,
                 cv::Mat& img,
                 std::shared_ptr<spdlog::logger> logger = nullptr) {
        bool loadSuccess = false;
        std::string imgPath;
        if (loadParam(node, key, imgPath, logger)) {
            img = cv::imread(imgPath, cv::IMREAD_UNCHANGED);
            if (!img.empty()) {
                loadSuccess = true;
            } else {
                if (logger) logger->error("Could not load image \"{}\"", imgPath);
            }
        } else {
            if (logger) logger->error("Could not find YAML key \"{}\"", key);
        }
        if (logger) logger->info("Loaded image from {}", imgPath);
        return loadSuccess;
    }
};