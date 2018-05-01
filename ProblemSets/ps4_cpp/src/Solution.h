#pragma once

#include "../include/Config.h"

#include <opencv2/core/core.hpp>
#include <opencv2/core/types.hpp>

class Solution {
public:
    Solution(const Config& config);

    void runProblem1();
    void runProblem2();

private:
    // Helpful struct to hold problem set configurations for Harris corners problems, so that we
    // don't have to copy/paste a bunch of code
    struct FeaturesContainer {
        // Reference to input image
        const cv::Mat& _input;
        // Reference to runtime configuration for a Harris corner detector
        const Config::Harris& _config;
        // Flags to use CUDA for computation
        const bool _useGpu;
        const std::string _outPrefix, _gradImgPath, _crImgPath, _cornersImgPath, _keypointsImgPath,
            _matchesImgPath;

        // Filled when Harris functions are run
        cv::Mat _gradientX, _gradientY, _cornerResponse, _corners, _angles;
        std::vector<std::pair<int, int>> _cornerLocs;
        std::vector<cv::KeyPoint> _keypoints;

        FeaturesContainer(const cv::Mat& input,
                          const Config::Harris& config,
                          const bool useGpu,
                          const std::string& outPrefix,
                          const std::string& gradImgPath,
                          const std::string& crImgPath,
                          const std::string& cornersImgPath,
                          const std::string& keypointsImgPath,
                          const std::string& matchesImgPath)
            : _input(input), _config(config), _useGpu(useGpu), _outPrefix(outPrefix),
              _gradImgPath(gradImgPath), _crImgPath(crImgPath), _cornersImgPath(cornersImgPath),
              _keypointsImgPath(keypointsImgPath), _matchesImgPath(matchesImgPath) {}
    };

    std::vector<FeaturesContainer> _featConts;

    // Draw dots given by an input mask onto an image
    void drawDots(const cv::Mat& mask, const cv::Mat& img, cv::Mat& dottedImg);

    // Compute Harris corners
    void harrisHelper(FeaturesContainer& conf);

    // Find SIFT descriptors
    void siftHelper(FeaturesContainer& img1, FeaturesContainer& img2);
};
