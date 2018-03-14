#include "Solution.h"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <chrono>
#include <iostream>

// YAML file containing input parameters
static constexpr char _CONFIG_FILE_PATH[] = "../config/ps1.yaml";

int main() {
    Solution sol(_CONFIG_FILE_PATH);
    // Find edges in the first input image
    {
        cv::Mat detectedEdges;
        sol.generateEdge(sol._input0, *(sol._p2EdgeConfig), detectedEdges);
        cv::imwrite(sol._outputPathPrefix + "/ps1-1-a-1.png", detectedEdges);

        // Find lines in image
        cv::Mat accumulator;
        sol.houghLinesAccumulate(detectedEdges, *(sol._p2HoughConfig), accumulator);

        cv::imwrite(sol._outputPathPrefix + "/ps1-2-a-1.png", accumulator);

        // Find local maxima
        std::vector<std::pair<unsigned int, unsigned int>> localMaxima;
        sol.findLocalMaxima(accumulator, *(sol._p2HoughConfig), localMaxima);

        // Convert those local maxima to rho and theta values
        std::vector<std::pair<int, int>> rhoThetaVals;
        std::cout << "Maxima rho, theta values:" << std::endl;
        for (const auto& val : localMaxima) {
            auto rt = sol.rowColToRhoTheta(val, sol._input0, *(sol._p2HoughConfig));
            rhoThetaVals.push_back(rt);
            std::cout << "  (rho = " << rt.first << ", theta = " << rt.second << ")" << std::endl;
        }

        // Draw onto image
        cv::Mat drawnLines;
        cv::cvtColor(sol._input0, drawnLines, CV_GRAY2RGB);
        for (const auto& val : rhoThetaVals) {
            sol.drawLineParametric(drawnLines, val.first, val.second, CV_RGB(0x00, 0xFF, 0x00));
        }
        cv::imwrite(sol._outputPathPrefix + "/ps1-2-c-1.png", drawnLines);
    }

    //-------------
    // Problem 3
    {
        cv::Mat gaussFromNoisy;
        sol.gpuGaussian(sol._input0Noise, *(sol._p3EdgeConfig), gaussFromNoisy);
        cv::imwrite(sol._outputPathPrefix + "/ps1-3-a-1.png", gaussFromNoisy);

        cv::Mat edgeFromNoisy;
        sol.generateEdge(sol._input0Noise, *(sol._p3EdgeConfig), edgeFromNoisy);
        cv::imwrite(sol._outputPathPrefix + "/ps1-3-b-2.png", edgeFromNoisy);

        cv::Mat accumulator;
        sol.houghLinesAccumulate(edgeFromNoisy, *(sol._p3HoughConfig), accumulator);
        cv::imwrite(sol._outputPathPrefix + "/ps1-3-c-1.png", accumulator);

        std::vector<std::pair<unsigned int, unsigned int>> localMaxima;
        sol.findLocalMaxima(accumulator, *(sol._p3HoughConfig), localMaxima);

        std::vector<std::pair<int, int>> rhoThetaVals;
        std::cout << "Maxima rho, theta values:" << std::endl;
        for (const auto& val : localMaxima) {
            auto rt = sol.rowColToRhoTheta(val, sol._input0Noise, *(sol._p3HoughConfig));
            rhoThetaVals.push_back(rt);
            std::cout << "  (rho = " << rt.first << ", theta = " << rt.second << ")" << std::endl;
        }

        // Draw onto image
        cv::Mat drawnLines;
        cv::cvtColor(sol._input0Noise, drawnLines, CV_GRAY2RGB);
        for (const auto& val : rhoThetaVals) {
            sol.drawLineParametric(drawnLines, val.first, val.second, CV_RGB(0x00, 0xFF, 0x00));
        }
        cv::imwrite(sol._outputPathPrefix + "/ps1-3-c-2.png", drawnLines);
    }
}