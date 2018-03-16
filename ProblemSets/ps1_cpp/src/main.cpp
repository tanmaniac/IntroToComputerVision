#include "Config.h"
#include "Solution.h"

#include "spdlog/spdlog.h"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <chrono>
#include <iostream>

// YAML file containing input parameters
static constexpr char CONFIG_FILE_PATH[] = "../config/ps1.yaml";

std::shared_ptr<spdlog::logger> _logger;

/**
 * Executes problems 1 and 2.
 */
void runProb1Prob2(Config& config) {
    _logger->info("////////// Problem 1 & 2 output //////////");
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
    _logger->info("Maxima rho, theta values:");
    for (const auto& val : localMaxima) {
        auto rt = sol::rowColToRhoTheta(val, config._images._input0, config._p2Hough);
        rhoThetaVals.push_back(rt);
        _logger->info("    (rho = {}, theta = {})", rt.first, rt.second);
    }

    // Draw lines computed from rho and theta values onto image
    cv::Mat drawnLines;
    // Need to convert to RGB so that we can draw a colored line
    cv::cvtColor(config._images._input0, drawnLines, CV_GRAY2RGB);
    sol::drawLinesParametric(drawnLines, rhoThetaVals, CV_RGB(0, 0xFF, 0));
    cv::imwrite(config._outputPathPrefix + "/ps1-2-c-1.png", drawnLines);
}

// Executes problem 3
void runProblem3(Config& config) {
    _logger->info("////////// Problem 3 output //////////");
    // Compute a blurred version of the noisy input image
    cv::Mat gaussFromNoisy;
    sol::gaussianBlur(config._images._input0Noise, config._p3Edge, gaussFromNoisy);
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
    _logger->info("Maxima rho, theta values:");
    for (const auto& val : localMaxima) {
        auto rt = sol::rowColToRhoTheta(val, config._images._input0Noise, config._p3Hough);
        rhoThetaVals.push_back(rt);
        _logger->info("    (rho = {}, theta = {})", rt.first, rt.second);
    }

    // Draw computed lines onto image, converting the image to RGB first to allow for colored lines
    cv::Mat drawnLines;
    cv::cvtColor(config._images._input0Noise, drawnLines, CV_GRAY2RGB);
    sol::drawLinesParametric(drawnLines, rhoThetaVals, CV_RGB(0, 0xFF, 0));
    cv::imwrite(config._outputPathPrefix + "/ps1-3-c-2.png", drawnLines);
}

void runProblem4(const Config& config) {
    _logger->info("////////// Problem 4 output //////////");
    // Convert input image to monochrome
    cv::Mat input1Mono;
    cv::cvtColor(config._images._input1, input1Mono, cv::COLOR_RGB2GRAY);
    input1Mono.convertTo(input1Mono, CV_32FC1); // Convert to floating point for gaussian

    // Blur the input image
    cv::Mat blurred;
    sol::gaussianBlur(input1Mono, config._p4Edge, blurred);
    cv::imwrite(config._outputPathPrefix + "/ps1-4-a-1.png", blurred);

    // Find edges in the input image
    cv::Mat edges;
    sol::generateEdge(input1Mono, config._p4Edge, edges);
    cv::imwrite(config._outputPathPrefix + "/ps1-4-b-1.png", edges);

    // Build accumulator matrix for Hough line transform
    cv::Mat accumulator;
    sol::houghLinesAccumulate(edges, config._p4Hough, accumulator);

    // Find local maxima in accumulation matrix
    std::vector<std::pair<unsigned int, unsigned int>> localMaxima;
    sol::findLocalMaxima(accumulator, config._p4Hough, localMaxima);
    cv::imwrite(config._outputPathPrefix + "/ps1-4-c-1.png", accumulator);

    // Convert local maxima from (row, col) values to (rho, theta) values
    std::vector<std::pair<int, int>> rhoThetaVals;
    _logger->info("Maxima rho, theta values:");
    for (const auto& val : localMaxima) {
        auto rt = sol::rowColToRhoTheta(val, input1Mono, config._p4Hough);
        rhoThetaVals.push_back(rt);
        _logger->info("    (rho = {}, theta = {})", rt.first, rt.second);
    }

    // Draw lines
    cv::Mat drawnLines;
    cv::cvtColor(input1Mono, drawnLines, CV_GRAY2RGB);
    sol::drawLinesParametric(drawnLines, rhoThetaVals, CV_RGB(0, 0xFF, 0));
    cv::imwrite(config._outputPathPrefix + "/ps1-4-c-2.png", drawnLines);
}

void runProblem5(const Config& config) {
    _logger->info("////////// Problem 5 output //////////");
    // Convert input image to monochrome
    cv::Mat input1Mono;
    cv::cvtColor(config._images._input1, input1Mono, cv::COLOR_RGB2GRAY);
    input1Mono.convertTo(input1Mono, CV_32FC1); // Convert to floating point for gaussian

    // Blur the input image
    cv::Mat blurred;
    sol::gaussianBlur(input1Mono, config._p5Edge, blurred);
    cv::imwrite(config._outputPathPrefix + "/ps1-5-a-1.png", blurred);

    // Find edges in the input image
    cv::Mat edges;
    sol::generateEdge(input1Mono, config._p5Edge, edges);
    cv::imwrite(config._outputPathPrefix + "/ps1-5-a-2.png", edges);

    // Build accumulator matrix for Hough line transform
    cv::Mat accumulator;
    sol::houghCirclesAccumulate(edges, config._p5Hough._minRadius, accumulator);
    cv::imwrite(config._outputPathPrefix + "/ps1-5-a-3.png", accumulator);

    // Find local maxima in accumulation matrix
    std::vector<std::pair<unsigned int, unsigned int>> localMaxima;
    sol::findLocalMaxima(accumulator, config._p5Hough, localMaxima);
    _logger->info("Local maxima:");
    for (const auto& val : localMaxima) {
        _logger->info("    y = {}, x = {}", val.first, val.second);
    }

    // Draw circles
    cv::Mat circles;
    cv::cvtColor(input1Mono, circles, CV_GRAY2RGB);
    sol::drawCircles(circles, localMaxima, config._p5Hough._minRadius, CV_RGB(0, 0xFF, 0));
    cv::imwrite(config._outputPathPrefix + "/ps1-5-a-4.png", circles);

    // Do the same thing, but iteratively across the entire radius range
    cv::cvtColor(input1Mono, circles, CV_GRAY2RGB); // Effectively reset the previous image
    for (size_t radius = config._p5Hough._minRadius; radius <= config._p5Hough._maxRadius;
         radius++) {
        accumulator = 0;
        localMaxima.clear();
        sol::houghCirclesAccumulate(edges, radius, accumulator);
        sol::findLocalMaxima(accumulator, config._p5Hough, localMaxima);
        sol::drawCircles(circles, localMaxima, radius, CV_RGB(0, 0xFF, 0));
    }
    cv::imwrite(config._outputPathPrefix + "/ps1-5-b-1.png", circles);
}

void runProblem6(const Config& config) {
    _logger->info("////////// Problem 6 output //////////");
    // Convert input image to monochrome
    cv::Mat input2Mono;
    cv::cvtColor(config._images._input2, input2Mono, cv::COLOR_RGB2GRAY);
    input2Mono.convertTo(input2Mono, CV_32FC1); // Convert to floating point for gaussian

    // Find edges in the input image
    cv::Mat edges;
    sol::generateEdge(input2Mono, config._p6Edge, edges);
    cv::imwrite(config._outputPathPrefix + "/ps1-6-a-0.1.png", edges);

    // Build accumulator matrix for Hough line transform
    cv::Mat accumulator;
    sol::houghLinesAccumulate(edges, config._p6Hough, accumulator);

    // Find local maxima in accumulation matrix
    std::vector<std::pair<unsigned int, unsigned int>> localMaxima;
    sol::findLocalMaxima(accumulator, config._p6Hough, localMaxima);
    cv::imwrite(config._outputPathPrefix + "/ps1-6-a-0.2.png", accumulator);

    // Convert local maxima from (row, col) values to (rho, theta) values
    std::vector<std::pair<int, int>> rhoThetaVals;
    _logger->info("Maxima rho, theta values:");
    for (const auto& val : localMaxima) {
        auto rt = sol::rowColToRhoTheta(val, input2Mono, config._p6Hough);
        rhoThetaVals.push_back(rt);
        _logger->info("    (rho = {}, theta = {})", rt.first, rt.second);
    }

    // Draw lines
    cv::Mat drawnLines;
    cv::cvtColor(input2Mono, drawnLines, CV_GRAY2RGB);
    sol::drawLinesParametric(drawnLines, rhoThetaVals, CV_RGB(0, 0xFF, 0));
    cv::imwrite(config._outputPathPrefix + "/ps1-6-a-1.png", drawnLines);

    // Part 2: Find parallel lines
    std::vector<std::pair<unsigned int, unsigned int>> parallels;
    sol::findParallelLines(localMaxima, 4, 150, parallels);

    std::vector<std::pair<int, int>> parallelRhoThetaVals;
    _logger->info("Parallel rho, theta values:");
    for (const auto& val : parallels) {
        auto rt = sol::rowColToRhoTheta(val, input2Mono, config._p6Hough);
        parallelRhoThetaVals.push_back(rt);
        _logger->info("    (rho = {}, theta = {})", rt.first, rt.second);
    }

    // Draw parallel lines onto image
    cv::cvtColor(input2Mono, drawnLines, CV_GRAY2RGB);
    sol::drawLinesParametric(drawnLines, parallelRhoThetaVals, CV_RGB(0, 0xFF, 0));
    cv::imwrite(config._outputPathPrefix + "/ps1-6-c-1.png", drawnLines);
}

int main() {
    // Setup
    _logger = spdlog::basic_logger_mt("logger", "ps1_log.txt");
    Config config(CONFIG_FILE_PATH);
    auto start = std::chrono::high_resolution_clock::now();

    // Run Problems 1 and 2
    runProb1Prob2(config);
    // Run Problem 3
    runProblem3(config);
    runProblem4(config);
    runProblem5(config);
    runProblem6(config);

    // Record runtime
    auto finish = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> runtime = finish - start;
    _logger->info("Total runtime = {} ms", runtime.count());
    _logger->info("------------------------------------------------------------");
    return 0;
}