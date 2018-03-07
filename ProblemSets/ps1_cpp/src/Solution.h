#pragma once

// Definitions for solution class that runs all the solutions

#include <opencv2/core/core.hpp>

class Solution {
public:
    // TODO: Is there any way I can use an intialization list for this? Everything is loaded from
    // the YAML config file @tanmaniac
    Solution(const std::string& configFilePath);

    /**
     * Problem 1.a.
     * Load the input grayscale image (input/ps1-input0.png) as img and generate an edge image –
     * which is a binary image with white pixels (1) on the edges and black pixels (0) elsewhere.
     *
     * For reference, do “doc edge” in Matlab and read about edge operators. Use one operator of
     * your choosing – for this image it probably won’t matter much. If your edge operator uses
     * parameters (like ‘canny’) play with those until you get the edges you would expect to see.
     */
    void generateEdge(const cv::Mat& input, cv::Mat& output);
    // Using my separable convolution implementation
    void gpuGaussian(const cv::Mat& input, cv::Mat& output);

    // Matrices representing each image
    // TODO: I don't like that these are not const @tanmaniac
    cv::Mat _input0, _input0Noise, _input1, _input2, _input3;

private:
    // Create a directory
    bool makeDir(const std::string& dirPath);
    // Output path prefix
    std::string _outputPathPrefix;

    // Paths to each of the input images
    std::string _input0Path, _input0NoisePath, _input1Path, _input2Path, _input3Path;

    // Edge detector parameters
    size_t _gaussianSize;
    float _gaussianSigma;
    double _lowerThreshold, _upperThreshold, _sobelApertureSize;
};