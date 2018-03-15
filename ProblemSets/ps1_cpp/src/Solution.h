#pragma once

// Definitions for solution class that runs all the solutions
#include <yaml-cpp/yaml.h>
#include "Config.h"

#include <opencv2/core/core.hpp>

namespace sol {
/**
 * Problem 1.a.
 * Load the input grayscale image (input/ps1-input0.png) as img and generate an edge image –
 * which is a binary image with white pixels (1) on the edges and black pixels (0) elsewhere.
 *
 * For reference, do “doc edge” in Matlab and read about edge operators. Use one operator of
 * your choosing – for this image it probably won’t matter much. If your edge operator uses
 * parameters (like ‘canny’) play with those until you get the edges you would expect to see.
 */
void generateEdge(const cv::Mat& input, const Config::EdgeDetect& config, cv::Mat& output);
// Using my separable convolution implementation
void gaussianBlur(const cv::Mat& input, const Config::EdgeDetect& config, cv::Mat& output);

/**
 * Problem 2.a.
 *  Computes the Hough Transform for lines and produces an accumulator array.
 */
void houghLinesAccumulate(const cv::Mat& edgeMask,
                          const Config::Hough& config,
                          cv::Mat& accumulator);

/**
 * Problem 2.b.
 *  Write a function hough_lines_draw to draw color lines that correspond to peaks found in the
 * accumulator array. This means you need to look up rho, theta values using the peak indices,
 * and then convert them (back) to line parameters in cartesian coordinates (you can then use
 * regular line-drawing functions).
 */
void findLocalMaxima(const cv::Mat& accumulator,
                     const Config::Hough& config,
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
                                     const Config::Hough& config);
void drawLineParametric(cv::Mat& image, const float rho, const float theta, const cv::Scalar color);
}; // namespace sol