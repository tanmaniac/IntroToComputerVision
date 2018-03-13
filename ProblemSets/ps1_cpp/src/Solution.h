#pragma once

// Definitions for solution class that runs all the solutions
#include <yaml-cpp/yaml.h>

#include <opencv2/core/core.hpp>

#include <memory>

class Solution {
public:
    struct EdgeDetectConfig {
        const size_t _gaussianSize;
        const float _gaussianSigma;
        const double _lowerThreshold, _upperThreshold, _sobelApertureSize;

        EdgeDetectConfig() = default;

        EdgeDetectConfig(const YAML::Node& edgeDetectNode)
            : _gaussianSize(edgeDetectNode["gaussian_size"].as<size_t>()),
              _gaussianSigma(edgeDetectNode["gaussian_sigma"].as<float>()),
              _lowerThreshold(edgeDetectNode["lower_threshold"].as<double>()),
              _upperThreshold(edgeDetectNode["upper_threshold"].as<double>()),
              _sobelApertureSize(edgeDetectNode["sobel_aperture_size"].as<double>()) {}
    };

    struct HoughConfig {
        const size_t _rhoBinSize, _thetaBinSize, _numPeaks;
        const int _threshold;

        HoughConfig(const YAML::Node& houghNode)
            : _rhoBinSize(houghNode["rho_bin_size"].as<size_t>()),
              _thetaBinSize(houghNode["theta_bin_size"].as<size_t>()),
              _threshold(houghNode["threshold"].as<int>()),
              _numPeaks(houghNode["num_peaks"].as<size_t>()) {}
    };

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
    std::shared_ptr<EdgeDetectConfig> _p2EdgeConfig;
    void generateEdge(const cv::Mat& input, cv::Mat& output);
    // Using my separable convolution implementation
    void gpuGaussian(const cv::Mat& input, cv::Mat& output);

    /**
     * Problem 2.a.
     *  Computes the Hough Transform for lines and produces an accumulator array.
     */
    std::shared_ptr<HoughConfig> _p2HoughConfig;
    void serialHoughLinesAccumulate(const cv::Mat& edgeMask, cv::Mat& accumulator);
    void houghLinesAccumulate(const cv::Mat& edgeMask, cv::Mat& accumulator);

    /**
     * Problem 2.b.
     *  Write a function hough_lines_draw to draw color lines that correspond to peaks found in the
     * accumulator array. This means you need to look up rho, theta values using the peak indices,
     * and then convert them (back) to line parameters in cartesian coordinates (you can then use
     * regular line-drawing functions).
     */
    void findLocalMaxima(const cv::Mat& accumulator,
                         std::vector<std::pair<unsigned int, unsigned int>>& localMaxima);

    /**
     * Problem 2.c.
     * Write a function hough_lines_draw to draw color lines that correspond to peaks found in the
     * accumulator array. This means you need to look up rho, theta values using the peak indices,
     * and then convert them (back) to line parameters in cartesian coordinates (you can then use
     * regular line-drawing functions).
     *
     * Use this to draw lines on the original grayscale (not edge) image. The lines should extend to
     * the edges of the image (aka infinite lines)
     */
    // Convert a row x column value computed by findLocalMaxima to a rho x theta value. This just
    // adjusts the row, column value to account for the bin sizes and the diagonal size of the
    // image.
    std::pair<int, int> rowColToRhoTheta(const std::pair<unsigned int, unsigned int>& coordinates,
                                         const cv::Mat& inputImage,
                                         const HoughConfig& config);
    void drawLineParametric(cv::Mat& image,
                            const float rho,
                            const float theta,
                            const cv::Scalar color);

    // Matrices representing each image
    // TODO: I don't like that these are not const @tanmaniac
    cv::Mat _input0, _input0Noise, _input1, _input2, _input3;

    // Output path prefix
    std::string _outputPathPrefix;

private:
    // Create a directory
    bool makeDir(const std::string& dirPath);

    // Paths to each of the input images
    std::string _input0Path, _input0NoisePath, _input1Path, _input2Path, _input3Path;

    // Edge detector parameters
    size_t _gaussianSize;
    float _gaussianSigma;
    double _lowerThreshold, _upperThreshold, _sobelApertureSize;
};