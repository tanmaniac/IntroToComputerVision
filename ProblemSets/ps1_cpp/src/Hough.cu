#include <common/CudaCommon.cuh>
#include "Hough.h"

#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/remove.h>
#include <thrust/sort.h>
#include <opencv2/core/cuda_types.hpp>

#include <iostream>
#include <thread>
#include <utility>
#include <vector>

// Compute Hough transform accumulator matrix and local maxima in Hough space

#define PI 3.14159265

__host__ __device__ inline float degToRad(float theta) {
    return theta * PI / 180.f;
}

/**
 * \brief Compute Hough transform accumulator
 * \param edgeMask pointer to matrix containing masked edge points
 * \param diagDist diagonal size of the input matrix
 * \param rhoBinSize size of rho bins
 * \param thetaBinSize size of theta bins
 * \param histo output matrix of histogram values
 */
__global__ void houghAccumulateKernel(const cv::cuda::PtrStepSz<unsigned char> edgeMask,
                                      const size_t diagDist,
                                      const size_t rhoBinSize,
                                      const size_t thetaBinSize,
                                      cv::cuda::PtrStepSz<int> histo) {
    const uint2 threadPos = getPosition();

    // Return if we're outside the bounds of the image, or if this is not a masked point
    if (threadPos.x >= edgeMask.cols || threadPos.y >= edgeMask.rows ||
        edgeMask(threadPos.x, threadPos.y) == 0) {
        return;
    }

    // Iterate over all values of theta and sum up in histogram
    for (float theta = MIN_THETA; theta < MAX_THETA; theta += thetaBinSize) {
        float thetaRad = degToRad(theta);
        int rho = roundf(threadPos.x * cosf(thetaRad) + threadPos.y * sinf(thetaRad)) + diagDist;
        int rhoBin = roundf(rho / rhoBinSize);
        int thetaBin = roundf((theta - MIN_THETA) / thetaBinSize);
        atomicAdd(&histo(rhoBin, thetaBin), 1);
    }
}

struct HoughPoint2D {
    int _votes;
    unsigned int _rho, _theta;
    bool _isLocalMaxima = false;

    __host__ __device__ HoughPoint2D() : _votes(0), _rho(0), _theta(0) {}

    __host__ __device__ HoughPoint2D(int votes, unsigned int rho, unsigned int theta)
        : _votes(votes), _rho(rho), _theta(theta) {}

    __host__ __device__
        HoughPoint2D(int votes, unsigned int rho, unsigned int theta, bool isLocalMaxima)
        : _votes(votes), _rho(rho), _theta(theta), _isLocalMaxima(isLocalMaxima) {}

    __host__ __device__ friend bool operator<(const HoughPoint2D& lhs, const HoughPoint2D& rhs) {
        return lhs._votes < rhs._votes;
    }

    __host__ __device__ friend bool operator>(const HoughPoint2D& lhs, const HoughPoint2D& rhs) {
        return rhs < lhs;
    }

    __host__ __device__ friend bool operator<=(const HoughPoint2D& lhs, const HoughPoint2D& rhs) {
        return !(lhs > rhs);
    }

    __host__ __device__ friend bool operator>=(const HoughPoint2D& lhs, const HoughPoint2D& rhs) {
        return !(rhs < lhs);
    }
};

/**
 * \brief Find the local maxima of the Hough transform accumulator. If it's a maxima, set the index
 * in the mask output to 1; if not, set to 0.
 * \param accumulator Hough transform accumulator matrix, where y axis is rho and x axis is theta
 * \param localMaximaPoints Binary mask of local maxima, where 1 is marked as being a local maxima
 * and 0 is not
 */
__global__ void findLocalMaximaKernel(const cv::cuda::PtrStepSz<int> accumulator,
                                      HoughPoint2D* localMaximaPoints) {
    const uint2 threadPos = getPosition();

    // Return if outside the bounds of the image
    if (threadPos.x >= accumulator.cols || threadPos.y >= accumulator.rows) {
        return;
    }

    const unsigned int threadLoc = convert2dTo1d(threadPos, accumulator.cols);

    // TODO: use shared memory
    bool isLocalMaxima = true;
    for (int y = max(0, int(threadPos.y) - 1); y < min(accumulator.rows - 1, threadPos.y + 1);
         y++) {
        for (int x = max(0, int(threadPos.x) - 1); x < min(accumulator.cols - 1, threadPos.x + 1);
             x++) {
            if (accumulator(y, x) > accumulator(threadPos.y, threadPos.x)) {
                isLocalMaxima = false;
            }
        }
    }

    localMaximaPoints[threadLoc] = HoughPoint2D(
        accumulator(threadPos.y, threadPos.x), threadPos.y, threadPos.x, isLocalMaxima);
}

//-----------------------------------------------------------------------------
// C++ wrappers

// Model of Predicate that returns true if the input point both is a local maxima and has a vote
// count greater than the set threshold.
struct MaskAndThreshold {
    int _threshold;

    __host__ __device__ MaskAndThreshold() : _threshold(0) {}

    __host__ __device__ MaskAndThreshold(int threshold) : _threshold(threshold) {}

    __host__ __device__ bool operator()(const HoughPoint2D& val) {
        return !(val._isLocalMaxima && val._votes >= _threshold);
    }
};

void cuda::houghAccumulate(const cv::cuda::GpuMat& edgeMask,
                           const size_t rhoBinSize,
                           const size_t thetaBinSize,
                           cv::cuda::GpuMat& accumulator) {
    assert(edgeMask.type() == CV_8UC1);
    static constexpr size_t TILE_SIZE = 16;

    const size_t maxDist =
        ceil(sqrt(edgeMask.rows * edgeMask.rows + edgeMask.cols * edgeMask.cols));
    const size_t rhoBins = (max(size_t(1), size_t(ceil(float(2 * maxDist) / float(rhoBinSize)))));
    const size_t thetaBins =
        (max(size_t(1), size_t(ceil(float(THETA_WIDTH) / float(thetaBinSize)))));
    accumulator.create(rhoBins, thetaBins, CV_32SC1);
    std::cout << "accumulator size = " << accumulator.rows << " x " << accumulator.cols
              << std::endl;

    // Determine block and grid size
    dim3 blocks(max(1, (unsigned int)ceil(float(edgeMask.cols) / float(TILE_SIZE))),
                max(1, (unsigned int)ceil(float(edgeMask.rows) / float(TILE_SIZE))));
    dim3 threads(TILE_SIZE, TILE_SIZE);

    // Launch kernel. cv::cuda::GpuMat types are convertable to cv::cuda::PtrStepSz wrapper types
    houghAccumulateKernel<<<blocks, threads>>>(
        edgeMask, maxDist, rhoBinSize, thetaBinSize, accumulator);
    cudaDeviceSynchronize();
    checkCudaErrors(cudaGetLastError());
}

void cuda::houghAccumulate(const cv::Mat& edgeMask,
                           const size_t rhoBinSize,
                           const size_t thetaBinSize,
                           cv::Mat& accumulator) {
    assert(edgeMask.type() == CV_8UC1);
    cv::cuda::GpuMat d_edgeMask, d_accumulator;

    // Copy input to GPU
    d_edgeMask.upload(edgeMask);
    std::cout << "d_edgemask size = " << d_edgeMask.rows << " x " << d_edgeMask.cols << std::endl;

    houghAccumulate(d_edgeMask, rhoBinSize, thetaBinSize, d_accumulator);

    // Copy result back to CPU
    d_accumulator.download(accumulator);
}

void cuda::findLocalMaxima(const cv::cuda::GpuMat& accumulator,
                           const size_t numPeaks,
                           const int threshold,
                           std::vector<std::pair<unsigned int, unsigned int>>& localMaxima) {
    static constexpr size_t TILE_SIZE = 16;

    thrust::device_vector<HoughPoint2D> localMaximaPoints(accumulator.rows * accumulator.cols);

    // Determine block and grid size
    dim3 blocks(max(1, (unsigned int)ceil(float(accumulator.cols) / float(TILE_SIZE))),
                max(1, (unsigned int)ceil(float(accumulator.rows) / float(TILE_SIZE))));
    dim3 threads(TILE_SIZE, TILE_SIZE);

    findLocalMaximaKernel<<<blocks, threads>>>(accumulator, localMaximaPoints.data().get());
    cudaDeviceSynchronize();
    checkCudaErrors(cudaGetLastError());

    // Filter by threshold, where threshold is the minimum number of votes required to count as a
    // peak
    // thrust::device_vector<HoughPoint2D>::iterator threshEnd = thrust::remove_if(
    auto threshEnd = thrust::remove_if(
        localMaximaPoints.begin(), localMaximaPoints.end(), MaskAndThreshold(threshold));

    // Sort points by threshold values in descending order
    thrust::stable_sort(localMaximaPoints.begin(), threshEnd, thrust::greater<HoughPoint2D>());

    // Figure out how many values to copy
    thrust::device_vector<HoughPoint2D>::iterator copyEnd =
        (numPeaks < threshEnd - localMaximaPoints.begin()) ? localMaximaPoints.begin() + numPeaks
                                                           : threshEnd;

    // Transfrom the masked and thresholded points into pairs of (rho, theta) values
    for (auto begin = localMaximaPoints.begin(); begin < copyEnd; begin++) {
        unsigned int rho = static_cast<HoughPoint2D>(*begin)._rho;
        unsigned int theta = static_cast<HoughPoint2D>(*begin)._theta;
        localMaxima.emplace_back(std::make_pair(rho, theta));
    }
}

void cuda::findLocalMaxima(const cv::Mat& accumulator,
                           const size_t numPeaks,
                           const int threshold,
                           std::vector<std::pair<unsigned int, unsigned int>>& localMaxima) {
    cv::cuda::GpuMat d_accumulator, d_localMaximaMask;

    // Copy to GPU
    d_accumulator.upload(accumulator);
    findLocalMaxima(d_accumulator, numPeaks, threshold, localMaxima);
}