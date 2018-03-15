#pragma once

#include <opencv2/core/core.hpp>
#include <opencv2/core/cuda.hpp>

#define MIN_THETA 0.f
#define MAX_THETA 180.f
#define THETA_WIDTH (MAX_THETA - MIN_THETA)

// C++ wrappers around CUDA kernels

namespace cuda {
void houghAccumulate(const cv::cuda::GpuMat& edgeMask,
                     const size_t rhoBinSize,
                     const size_t thetaBinSize,
                     cv::cuda::GpuMat& accumulator);

void houghAccumulate(const cv::Mat& edgeMask,
                     const size_t rhoBinSize,
                     const size_t thetaBinSize,
                     cv::Mat& accumulator);

/**
 * \brief Locates peaks in the Hough transform matrix.
 * \param accumulator Hough transform accumulator
 * \param numPeaks maximum number of peaks to detect
 * \param threshold minimum number of votes required to be considered a peak
 * \param localMaxima output vector of pairs, where the first value is the row and second value is
 * the column in which the peak was found
 */
void findLocalMaxima(const cv::cuda::GpuMat& accumulator,
                     const size_t numPeaks,
                     const int threshold,
                     std::vector<std::pair<unsigned int, unsigned int>>& localMaxima);

/**
 * \brief Locates peaks in the Hough transform matrix.
 * \param accumulator Hough transform accumulator
 * \param numPeaks maximum number of peaks to detect
 * \param threshold minimum number of votes required to be considered a peak
 * \param localMaxima output vector of pairs, where the first value is the row and second value is
 * the column in which the peak was found
 */
void findLocalMaxima(const cv::Mat& accumulator,
                     const size_t numPeaks,
                     const int threshold,
                     std::vector<std::pair<unsigned int, unsigned int>>& localMaxima);
}; // namespace cuda
