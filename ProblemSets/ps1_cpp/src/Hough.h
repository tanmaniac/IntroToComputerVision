#pragma once

#include <opencv2/core/core.hpp>
#include <opencv2/core/cuda.hpp>

#define MIN_THETA -90.f
#define MAX_THETA 90.f
#define THETA_WIDTH (MAX_THETA - MIN_THETA)

// C++ wrappers around CUDA kernels

namespace cuda {

/**
 * \brief Create Hough lines accumulation matrix.
 *
 * \param edgemask input matrix of edges
 * \param rhoBinSize rho resolution
 * \param thetaBinSize theta resolution
 * \pararm accumulator output matrix of accumulated votes
 */
void houghLinesAccumulate(const cv::cuda::GpuMat& edgeMask,
                          const unsigned int rhoBinSize,
                          const unsigned int thetaBinSize,
                          cv::cuda::GpuMat& accumulator);

/**
 * \brief Create Hough lines accumulation matrix.
 *
 * \param edgemask input matrix of edges
 * \param rhoBinSize rho resolution
 * \param thetaBinSize theta resolution
 * \pararm accumulator output matrix of accumulated votes
 */
void houghLinesAccumulate(const cv::Mat& edgeMask,
                          const unsigned int rhoBinSize,
                          const unsigned int thetaBinSize,
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
                     const unsigned int numPeaks,
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
                     const unsigned int numPeaks,
                     const int threshold,
                     std::vector<std::pair<unsigned int, unsigned int>>& localMaxima);

/**
 * \brief Create Hough circles accumulation matrix
 *
 * \param edgeMask input matrix of edges
 * \param radius radius to which the Hough transform is computed
 * \param accumulator output matrix of accumulated votes
 */
void houghCirclesAccumulate(const cv::cuda::GpuMat& edgeMask,
                            const size_t radius,
                            cv::cuda::GpuMat& accumulator);

/**
 * \brief Create Hough circles accumulation matrix
 *
 * \param edgeMask input matrix of edges
 * \param radius radius to which the Hough transform is computed
 * \param accumulator output matrix of accumulated votes
 */
void houghCirclesAccumulate(const cv::Mat& edgeMask, const size_t radius, cv::Mat& accumulator);
}; // namespace cuda
