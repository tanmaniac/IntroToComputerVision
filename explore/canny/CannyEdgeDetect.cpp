#include <yaml-cpp/yaml.h>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <iostream>

static constexpr char CONFIG_FILE_PATH[] = "../../config/cannyEdgeDetect.yaml";

// CUDA kernel adapters
extern void testAdapter();

extern void makeGaussianAdapter(const size_t kernelRadius, const float sigma, cv::Mat& gaussianMat);

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
    cv::Mat sourceImg = cv::imread(sourceImgPath, CV_LOAD_IMAGE_COLOR);
    std::cout << "Loaded " << sourceImgPath << ": rows = " << sourceImg.rows
              << ", cols = " << sourceImg.cols << std::endl;
    std::cout << "Blurring with radius " << gaussRad << std::endl;

    // Build the gaussian kernel and store it in a cv::Mat
    cv::Mat gaussianKernel;
    makeGaussianAdapter(gaussRad, sigma, gaussianKernel);

    std::cout << "gaussianKernel = " << std::endl << " " << gaussianKernel << std::endl;

    std::cout << "Hello world" << std::endl;
    // testAdapter();
    std::cout << "ran kernel" << std::endl;
}