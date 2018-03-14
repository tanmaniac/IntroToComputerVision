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
void gpuGaussian(const cv::Mat& input, const Config::EdgeDetect& config, cv::Mat& output);

/**
 * Problem 2.a.
 *  Computes the Hough Transform for lines and produces an accumulator array.
 */
void serialHoughLinesAccumulate(const cv::Mat& edgeMask, cv::Mat& accumulator);
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

/**
 * Problem 3
 * Use ps1-input0-noise.png - same image as before, but with noise. Compute a modestly smoothed
 * version of this image by using a Gaussian filter. Make σ at least a few pixels big.
 * Output:
 * Smoothed image: ps1-3-a-1.png
 *
 * Using an edge operator of your choosing, create a binary edge image for both the original
 * image (ps1-input0-noise.png) and the smoothed version above. Output: Two edge images:
 *  ps1-3-b-1.png (from original), ps1-3-b-2.png (from smoothed)
 *
 * Now apply your Hough method to the smoothed version of the edge image. Your goal is to adjust
 * the filtering, edge finding, and Hough algorithms to find the lines as best you can in this
 * test case.
 * Output:
 *  - Hough accumulator array image with peaks highlighted: ps1-3-c-1.png
 *  - Intensity image (original one with the noise) with lines drawn on them: ps1-3-c-2.png
 */
}; // namespace sol