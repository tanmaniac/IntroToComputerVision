#include "Solution.h"
#include "Convolution.h"

#include <yaml-cpp/yaml.h>

#include <opencv2/gpu/gpu.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <errno.h>
#include <sys/stat.h>
#include <iostream>

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
    if (YAML::Node edgeDetectNode = config["edge_detector"]) {
        _gaussianSize = edgeDetectNode["gaussian_size"].as<size_t>();
        _gaussianSigma = edgeDetectNode["gaussian_sigma"].as<float>();
        _lowerThreshold = edgeDetectNode["lower_threshold"].as<double>();
        _upperThreshold = edgeDetectNode["upper_threshold"].as<double>();
        _sobelApertureSize = edgeDetectNode["sobel_aperture_size"].as<double>();
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
    cv::gpu::GaussianBlur(
        d_input, d_blurred, cv::Size(_gaussianSize, _gaussianSize), _gaussianSigma);
    // Compute edges
    // First convert to uchar8 matrix
    d_blurred.convertTo(d_blurred, CV_8UC1);
    cv::gpu::Canny(d_blurred, d_output, _lowerThreshold, _upperThreshold, _sobelApertureSize);
    // Copy result back to output
    d_output.download(output);
}

void Solution::gpuGaussian(const cv::Mat& input, cv::Mat& output) {
    cv::Mat colGaussian = cv::getGaussianKernel(_gaussianSize, _gaussianSigma, CV_32F);
    cv::Mat rowGaussian;
    cv::transpose(colGaussian, rowGaussian);
    // Running CUDA convolution
    separableConvolution(input, rowGaussian, colGaussian, output);
}
