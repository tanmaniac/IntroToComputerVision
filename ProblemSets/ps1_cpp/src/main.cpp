#include "Config.h"
#include "Solution.h"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <chrono>
#include <iostream>

// YAML file containing input parameters
static constexpr char CONFIG_FILE_PATH[] = "../config/ps1.yaml";

/**
 * Executes problems 1 and 2.
 */
void runProb1Prob2(Config& config) {
    // Detect edges in the input image, which is just a checkerboard pattern
    cv::Mat detectedEdges;
    sol::generateEdge(config._images._input0, config._p2Edge, detectedEdges);
    cv::imwrite(config._outputPathPrefix + "/ps1-1-a-1.png", detectedEdges);

    // Compute the Hough transform of the image using the edges detected previously
    cv::Mat accumulator;
    sol::houghLinesAccumulate(detectedEdges, config._p2Hough, accumulator);

    cv::imwrite(config._outputPathPrefix + "/ps1-2-a-1.png", accumulator);

    // Find local maxima in the Hough transform
    std::vector<std::pair<unsigned int, unsigned int>> localMaxima;
    sol::findLocalMaxima(accumulator, config._p2Hough, localMaxima);

    // Convert those local maxima to rho and theta values
    std::vector<std::pair<int, int>> rhoThetaVals;
    std::cout << "Maxima rho, theta values:" << std::endl;
    for (const auto& val : localMaxima) {
        auto rt = sol::rowColToRhoTheta(val, config._images._input0, config._p2Hough);
        rhoThetaVals.push_back(rt);
        std::cout << "  (rho = " << rt.first << ", theta = " << rt.second << ")" << std::endl;
    }

    // Draw lines computed from rho and theta values onto image
    cv::Mat drawnLines;
    // Need to convert to RGB so that we can draw a colored line
    cv::cvtColor(config._images._input0, drawnLines, CV_GRAY2RGB);
    for (const auto& val : rhoThetaVals) {
        sol::drawLineParametric(drawnLines, val.first, val.second, CV_RGB(0x00, 0xFF, 0x00));
    }
    cv::imwrite(config._outputPathPrefix + "/ps1-2-c-1.png", drawnLines);
}

void runProblem3(Config& config) {
    // Compute a blurred version of the noisy input image
    cv::Mat gaussFromNoisy;
    sol::gpuGaussian(config._images._input0Noise, config._p3Edge, gaussFromNoisy);
    cv::imwrite(config._outputPathPrefix + "/ps1-3-a-1.png", gaussFromNoisy);

    // Find edges in the noisy input image using a Canny edge detector
    cv::Mat edgeFromNoisy;
    sol::generateEdge(config._images._input0Noise, config._p3Edge, edgeFromNoisy);
    cv::imwrite(config._outputPathPrefix + "/ps1-3-b-2.png", edgeFromNoisy);

    // Compute the Hough transform of the edge image
    cv::Mat accumulator;
    sol::houghLinesAccumulate(edgeFromNoisy, config._p3Hough, accumulator);
    cv::imwrite(config._outputPathPrefix + "/ps1-3-c-1.png", accumulator);

    // Find local maxima in the Hough transform accumulation matrix
    std::vector<std::pair<unsigned int, unsigned int>> localMaxima;
    sol::findLocalMaxima(accumulator, config._p3Hough, localMaxima);

    // Convert local maxima from (row, col) values to (rho, theta) values
    std::vector<std::pair<int, int>> rhoThetaVals;
    std::cout << "Maxima rho, theta values:" << std::endl;
    for (const auto& val : localMaxima) {
        auto rt = sol::rowColToRhoTheta(val, config._images._input0Noise, config._p3Hough);
        rhoThetaVals.push_back(rt);
        std::cout << "  (rho = " << rt.first << ", theta = " << rt.second << ")" << std::endl;
    }

    // Draw computed lines onto image, converting the image to RGB first to allow for colored lines
    cv::Mat drawnLines;
    cv::cvtColor(config._images._input0Noise, drawnLines, CV_GRAY2RGB);
    for (const auto& val : rhoThetaVals) {
        sol::drawLineParametric(drawnLines, val.first, val.second, CV_RGB(0x00, 0xFF, 0x00));
    }
    cv::imwrite(config._outputPathPrefix + "/ps1-3-c-2.png", drawnLines);
}

int main() {
    Config config(CONFIG_FILE_PATH);
    // Run Problems 1 and 2
    runProb1Prob2(config);

    // Run Problem 3
    runProblem3(config);
}