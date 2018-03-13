#include "Solution.h"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <chrono>
#include <iostream>

// YAML file containing input parameters
static constexpr char _CONFIG_FILE_PATH[] = "../config/ps1.yaml";

int main() {
    Solution mySol(_CONFIG_FILE_PATH);
    // Find edges in the first input image
    cv::Mat detectedEdges;
    mySol.generateEdge(mySol._input0, detectedEdges);
    cv::imwrite(mySol._outputPathPrefix + "/ps1-1-a-1.png", detectedEdges);

    // Find lines in image
    cv::Mat accumulator;
    mySol.houghLinesAccumulate(detectedEdges, accumulator);

    cv::imwrite(mySol._outputPathPrefix + "/ps1-2-a-1.png", accumulator);

    // Find local maxima
    std::vector<std::pair<unsigned int, unsigned int>> localMaxima;
    mySol.findLocalMaxima(accumulator, localMaxima);

    // Print local maxima
    std::cout << "Found maxima:" << std::endl;
    for (const auto& val : localMaxima) {
        std::cout << "  (row = " << val.first << ", col = " << val.second << ")" << std::endl;
    }

    // Convert those local maxima to rho and theta values
    std::vector<std::pair<int, int>> rhoThetaVals;
    std::cout << "Maxima rho, theta values:" << std::endl;
    for (const auto& val : localMaxima) {
        auto rt = mySol.rowColToRhoTheta(val, mySol._input0, *(mySol._p2HoughConfig));
        rhoThetaVals.push_back(rt);
        std::cout << "  (rho = " << rt.first << ", theta = " << rt.second << ")" << std::endl;
    }

    // Draw onto image
    cv::Mat drawnLines;
    cv::cvtColor(mySol._input0, drawnLines, CV_GRAY2RGB);
    for (const auto& val : rhoThetaVals) {
        mySol.drawLineParametric(drawnLines, val.first, val.second, CV_RGB(0x00, 0xFF, 0x00));
    }
    cv::imwrite(mySol._outputPathPrefix + "/ps1-2-c-1.png", drawnLines);
}