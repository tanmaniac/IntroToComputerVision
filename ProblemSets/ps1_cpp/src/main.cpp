#include "Solution.h"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <chrono>
#include <iostream>

// YAML file containing input parameters
static constexpr char _CONFIG_FILE_PATH[] = "../config/ps1.yaml";

int main() {
    Solution mySol(_CONFIG_FILE_PATH);
    // Find edges in the first input image
    cv::Mat detectedEdges;
    mySol.generateEdge(mySol._input0, detectedEdges);
    cv::imwrite("ps1_output/ps1-1-a-1.png", detectedEdges);
}