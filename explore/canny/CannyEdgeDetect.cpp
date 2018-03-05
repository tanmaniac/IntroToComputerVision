#include <yaml-cpp/yaml.h>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <iostream>

static constexpr char CONFIG_FILE_PATH[] = "../../config/cannyEdgeDetect.yaml";

// CUDA kernel adapters
extern void naiveGlobalConvolution(const cv::Mat& source, const cv::Mat& gaussian, cv::Mat& dest);

extern void buildGaussian(const size_t kernelRadius, const float sigma, cv::Mat& gaussianMat);

extern void convolutionRowsGPU(std::vector<float>& dest,
                               std::vector<float>& source,
                               std::vector<float>& kernel,
                               int imageW,
                               int imageH);

extern void convolutionColumnsGPU(std::vector<float>& dest,
                                  std::vector<float>& source,
                                  std::vector<float>& kernel,
                                  int imageW,
                                  int imageH);
// C++ functions

// Launch convolution using NVIDIA's convolution algorithm
void runSeparableConv(const cv::Mat& kernel,
                      const cv::Mat& source,
                      const float scale,
                      cv::Mat& dest) {
    // Copy kernel to a vector
    const size_t kernelWidth = kernel.cols;
    std::vector<float> gaussianVec(kernelWidth);
    for (unsigned int y = 0; y < kernelWidth; y++) {
        gaussianVec[y] = kernel.at<float>(y, 0, 0);
    }

    // Copy input image to a vector
    const size_t sourceRows = source.rows;
    const size_t sourceCols = source.cols;
    std::vector<float> sourceVec(sourceRows * sourceCols);
    for (unsigned int y = 0; y < sourceRows; y++) {
        for (unsigned int x = 0; x < sourceCols; x++) {
            sourceVec[y * sourceCols + x] = source.at<float>(y, x, 0);
        }
    }

    // Allocate space for the destination image
    std::vector<float> buffer(sourceRows * sourceCols);
    std::vector<float> destVec(sourceRows * sourceCols);

    // Do the convolution
    convolutionRowsGPU(buffer, sourceVec, gaussianVec, sourceCols, sourceRows);
    convolutionColumnsGPU(destVec, buffer, gaussianVec, sourceCols, sourceRows);

    // Copy back to the destination matrix
    for (unsigned int y = 0; y < sourceRows; y++) {
        for (unsigned int x = 0; x < sourceCols; x++) {
            dest.at<float>(y, x, 0) = destVec[y * sourceCols + x];
        }
    }

    dest *= scale * 2048;
}

int main() {
    // Load runtime configuration from YAML file
    YAML::Node config = YAML::LoadFile(CONFIG_FILE_PATH);
    if (config.IsNull()) {
        std::cerr << "Could not load input file (was looking for " << CONFIG_FILE_PATH << ")"
                  << std::endl;
        return -1;
    }
    YAML::Node imagesNode = config["images"];
    if (imagesNode.IsNull()) {
        std::cerr << "Malformed config file - could not find \"images\" node" << std::endl;
        return -1;
    }

    size_t gaussRad = config["gaussianRadius"].as<size_t>();
    float sigma = config["sigma"].as<float>();
    std::string sourceImgPath = imagesNode["source"].as<std::string>();
    // Done loading configuration from YAML file

    // Load input image
    cv::Mat sourceImg = cv::imread(sourceImgPath, CV_LOAD_IMAGE_GRAYSCALE);
    sourceImg.convertTo(sourceImg, CV_32FC1);
    std::cout << "Loaded " << sourceImgPath << ": rows = " << sourceImg.rows
              << ", cols = " << sourceImg.cols << ", channels = " << sourceImg.channels()
              << std::endl;
    std::cout << "Blurring with radius " << gaussRad << std::endl;

    // Build the gaussian kernel and store it in a cv::Mat
    cv::Mat gaussianKernel;
    buildGaussian(gaussRad, sigma, gaussianKernel);

    // Compute a scaling factor so we can get an image back with the same brightness as the original
    float scale = 1.f / cv::sum(gaussianKernel)[0];
    std::cout << "Scaling factor = " << scale << std::endl;

    // Blur the input image and store it in a cv::Mat
    cv::Mat blurredImg;
    naiveGlobalConvolution(sourceImg, gaussianKernel, blurredImg);

    // Multiply by scaling factor
    blurredImg = blurredImg * scale;

    cv::imwrite("blurred.png", blurredImg);

    // Nvidia
    std::cout << "Running NVIDIA's separable convolution" << std::endl;
    runSeparableConv(gaussianKernel, sourceImg, scale, blurredImg);
    cv::imwrite("blurred_nvidia.png", blurredImg);
}