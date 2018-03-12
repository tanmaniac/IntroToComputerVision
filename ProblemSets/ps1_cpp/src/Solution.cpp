#include "Solution.h"
#include "Convolution.h"
#include "Hough.h"

#include <yaml-cpp/yaml.h>

#include <opencv2/gpu/gpu.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <errno.h>
#include <sys/stat.h>
#include <cmath>
#include <iostream>

static constexpr float PI = 3.14159265;

Solution::Solution(const std::string& configFilePath) {
    // Load images from config file
    YAML::Node config = YAML::LoadFile(configFilePath);
    if (config.IsNull()) {
        std::cerr << "Could not load input file (was looking for " << configFilePath << ")"
                  << std::endl;
        return;
    }
    YAML::Node imagesNode = config["images"];
    if (imagesNode.IsNull()) {
        std::cerr << "Malformed config file - could not find \"images\" node" << std::endl;
        return;
    }

    // TODO: wrap in checks to make sure these nodes actually exist. Maybe these paths should just
    // be stuck in a vector since yaml-cpp provides iterators over maps @tanmaniac
    _input0Path = imagesNode["input0"].as<std::string>();
    _input0NoisePath = imagesNode["input0_noise"].as<std::string>();
    _input1Path = imagesNode["input1"].as<std::string>();
    _input2Path = imagesNode["input2"].as<std::string>();
    _input3Path = imagesNode["input3"].as<std::string>();
    // Done loading configuration from YAML file

    // Load the images from the paths that were just read
    _input0 = cv::imread(_input0Path, CV_LOAD_IMAGE_GRAYSCALE);
    _input0.convertTo(_input0, CV_32FC1);
    _input0Noise = cv::imread(_input0NoisePath, CV_LOAD_IMAGE_COLOR);
    _input1 = cv::imread(_input1Path, CV_LOAD_IMAGE_COLOR);
    _input2 = cv::imread(_input2Path, CV_LOAD_IMAGE_COLOR);
    _input3 = cv::imread(_input3Path, CV_LOAD_IMAGE_COLOR);

    std::cout << "Loaded input images" << std::endl;

    // Create output directory
    if (config["output_dir"]) {
        _outputPathPrefix = config["output_dir"].as<std::string>();
        if (makeDir(_outputPathPrefix)) {
            std::cout << "Created output directory at \"" << _outputPathPrefix << "\"" << std::endl;
        }
    } else {
        _outputPathPrefix = "./";
        std::cout << "No output path specified; using current directory" << std::endl;
    }

    // Load edge detector parameters
    if (YAML::Node edgeDetectNode = config["edge_detector_p2"]) {
        _p2EdgeConfig = std::make_shared<EdgeDetectConfig>(edgeDetectNode);
    }

    if (YAML::Node houghConfigNode = config["hough_transform_p2"]) {
        _p2HoughConfig = std::make_shared<HoughConfig>(houghConfigNode);
    }
}

bool Solution::makeDir(const std::string& dirPath) {
    const int dirErr = mkdir(dirPath.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
    if (dirErr == -1) {
        // The directory already exists, so there's nothing to do anyway. Return true
        return errno == EEXIST;
    }
    return true;
}

// Run on GPU
void Solution::generateEdge(const cv::Mat& input, cv::Mat& output) {
    cv::gpu::GpuMat d_input(input);
    cv::gpu::GpuMat d_blurred, d_output;

    // Run Gaussian blur
    cv::gpu::GaussianBlur(d_input,
                          d_blurred,
                          cv::Size(_p2EdgeConfig->_gaussianSize, _p2EdgeConfig->_gaussianSize),
                          _p2EdgeConfig->_gaussianSigma);
    // Compute edges
    // First convert to uchar8 matrix
    d_blurred.convertTo(d_blurred, CV_8UC1);
    cv::gpu::Canny(d_blurred,
                   d_output,
                   _p2EdgeConfig->_lowerThreshold,
                   _p2EdgeConfig->_upperThreshold,
                   _p2EdgeConfig->_sobelApertureSize);
    // Copy result back to output
    d_output.download(output);
}

void Solution::gpuGaussian(const cv::Mat& input, cv::Mat& output) {
    cv::Mat colGaussian =
        cv::getGaussianKernel(_p2EdgeConfig->_gaussianSize, _p2EdgeConfig->_gaussianSigma, CV_32F);
    cv::Mat rowGaussian;
    cv::transpose(colGaussian, rowGaussian);
    // Running CUDA convolution
    separableConvolution(input, rowGaussian, colGaussian, output);
}

// Serial implementation of Hough transform accumulation
void Solution::houghLinesAccumulate(const cv::Mat& edgeMask, cv::Mat& accumulator) {
    static constexpr int MIN_THETA = -90;
    static constexpr int MAX_THETA = 90;
    static constexpr int THETA_WIDTH = MAX_THETA - MIN_THETA;
    size_t maxDist = ceil(cv::sqrt(edgeMask.rows * edgeMask.rows + edgeMask.cols * edgeMask.cols));
    std::cout << "MaxDist = " << maxDist << std::endl;

    accumulator = cv::Mat(2 * maxDist, THETA_WIDTH, CV_32SC1);
    accumulator = cv::Scalar(0);

    // Iterate over the mask
    for (unsigned int y = 0; y < edgeMask.rows; y++) {
        for (unsigned int x = 0; x < edgeMask.cols; x++) {
            if (edgeMask.at<unsigned char>(y, x, 0) != 0) {
                // Vote in Hough accumulator
                for (int theta = MIN_THETA; theta < MAX_THETA; theta++) {
                    double thetaRad = theta * PI / 180.0;
                    unsigned int rho = round(x * cos(thetaRad) + y * sin(thetaRad)) + maxDist;
                    accumulator.at<int>(rho, theta - MIN_THETA, 0) += 1;
                }
            }
        }
    }
}

void Solution::houghCudaAccumulate(const cv::Mat& edgeMask, cv::Mat& accumulator) {
    houghAccumulate(
        edgeMask, _p2HoughConfig->_rhoBinSize, _p2HoughConfig->_thetaBinSize, accumulator);
}