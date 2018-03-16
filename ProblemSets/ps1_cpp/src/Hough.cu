#include <common/GpuTimer.h>
#include <common/CudaCommon.cuh>
#include "Hough.h"

#include "spdlog/spdlog.h"

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
 * \param maskedPoints array of (x,y) coordinates of each edge pixel
 * \param numPoints length of maskedPoints array
 * \param diagDist diagonal size of the input matrix
 * \param rhoBinSize size of rho bins
 * \param thetaBinSize size of theta bins
 * \param histo output matrix of histogram values
 */
__global__ void houghLinesAccumulateKernel(const uint2* const maskedPoints,
                                           const size_t numPoints,
                                           const size_t diagDist,
                                           const unsigned int rhoBinSize,
                                           const unsigned int thetaBinSize,
                                           cv::cuda::PtrStepSz<int> histo) {
    const unsigned int threadPos = blockIdx.x * blockDim.x + threadIdx.x;

    // Return if we're outside the bounds of the image, or if this is not a masked point
    if (threadPos >= numPoints) {
        return;
    }

    uint2 thisPoint = maskedPoints[threadPos];
    float cosTheta, sinTheta;
    // Iterate over all values of theta and sum up in histogram
    for (int theta = MIN_THETA; theta < MAX_THETA; theta += thetaBinSize) {
        float thetaRad = degToRad(theta);
        __sincosf(thetaRad, &sinTheta, &cosTheta);
        float rho = roundf(thisPoint.x * cosTheta + thisPoint.y * sinTheta) + diagDist;
        int rhoBin = roundf(rho / rhoBinSize);
        int thetaBin = roundf((theta - MIN_THETA) / thetaBinSize);
        atomicAdd(&histo(rhoBin, thetaBin), 1);
    }
}

// Horrible global memory implementation
__global__ void houghCirclesAccumulateKernel(const uint2* const maskedPoints,
                                             const size_t numPoints,
                                             const size_t radius,
                                             cv::cuda::PtrStepSz<int> histo) {
    const unsigned int threadPos = blockIdx.x * blockDim.x + threadIdx.x;

    // Return if we're outside the bounds of the image, or if this is not a masked point
    if (threadPos >= numPoints) {
        return;
    }

    uint2 thisPoint = maskedPoints[threadPos];
    unsigned int a, b;
    float sinTheta, cosTheta;
    // Generate circle
    for (int theta = 0; theta < 360; theta++) {
        __sincosf(degToRad(theta), &sinTheta, &cosTheta);
        a = thisPoint.x - radius * cosTheta;
        b = thisPoint.y - radius * sinTheta;

        // Make sure we're within the bounds of the image
        if (a < histo.cols && b < histo.rows) {
            atomicAdd(&histo(b, a), 1);
        }
    }
}

struct HoughPoint2D {
    int _votes;
    float _rho, _theta;
    bool _isLocalMaxima = false;

    __host__ __device__ HoughPoint2D() : _votes(0), _rho(0), _theta(0) {}

    __host__ __device__ HoughPoint2D(int votes, float rho, float theta)
        : _votes(votes), _rho(rho), _theta(theta) {}

    __host__ __device__ HoughPoint2D(int votes, float rho, float theta, bool isLocalMaxima)
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
 *
 * TODO: Use Trove for higher-performance array-of-structures access
 * https://github.com/bryancatanzaro/trove
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

// Edge images are massive sparse vectors, so I want to compact them down into more manageable
// arrays rather than wasting lots of warps processing empty pixels

// Convert matrix into a row-vector of (x, y) points
__global__ void maskToPointKernel(const cv::cuda::PtrStepSz<unsigned char> image, uint2* points) {
    const uint2 threadPos = getPosition();

    if (threadPos.x < image.cols && threadPos.y < image.rows) {
        points[convert2dTo1d(threadPos, image.cols)] = make_uint2(threadPos.x, threadPos.y);
    }
}

struct IsNonzero {
    __host__ __device__ bool operator()(const unsigned char val) {
        return val > 0;
    }
};

void streamCompactMask(const cv::cuda::GpuMat& edgeMask, thrust::device_vector<uint2>& points) {
    assert(edgeMask.isContinuous());
    auto logger = spdlog::get("logger");
    logger->info("Running stream compact on matrix with rows={}, cols={}, step={}",
                 edgeMask.rows,
                 edgeMask.cols,
                 edgeMask.step);

    static constexpr size_t TILE_SIZE = 16;

    // Resize output vector to fit the image
    thrust::device_vector<uint2> rawPoints(edgeMask.rows * edgeMask.cols);
    points.resize(edgeMask.rows * edgeMask.cols);

    // Determine block and grid size
    dim3 blocks(max(1, (unsigned int)ceil(float(edgeMask.cols) / float(TILE_SIZE))),
                max(1, (unsigned int)ceil(float(edgeMask.rows) / float(TILE_SIZE))));
    dim3 threads(TILE_SIZE, TILE_SIZE);

    // Timing
    GpuTimer timer;
    timer.start();
    maskToPointKernel<<<blocks, threads>>>(edgeMask, rawPoints.data().get());
    timer.stop();
    logger->info("Converting matrix to (x,y) points took {} ms", timer.getTime());

    thrust::device_ptr<const unsigned char> edgeMaskPtr =
        thrust::device_pointer_cast<const unsigned char>(edgeMask.ptr<unsigned char>());

    // Copy usable 2D points to the output vector
    auto endIter = thrust::copy_if(
        rawPoints.begin(), rawPoints.end(), edgeMaskPtr, points.begin(), IsNonzero());
    int numPointsCopied = endIter - points.begin();
    // Resize output to fit
    points.resize(numPointsCopied);
    logger->info("Masked {} points; points is now {} elems long", numPointsCopied, points.size());
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

void cuda::houghLinesAccumulate(const cv::cuda::GpuMat& edgeMask,
                                const unsigned int rhoBinSize,
                                const unsigned int thetaBinSize,
                                cv::cuda::GpuMat& accumulator) {
    assert(edgeMask.type() == CV_8UC1);
    static constexpr size_t TILE_SIZE = 128;

    const size_t maxDist =
        ceil(sqrt(edgeMask.rows * edgeMask.rows + edgeMask.cols * edgeMask.cols));
    const size_t rhoBins = (max(size_t(1), size_t(ceil(float(2 * maxDist) / float(rhoBinSize)))));
    const size_t thetaBins =
        (max(size_t(1), size_t(ceil(float(THETA_WIDTH) / float(thetaBinSize)))));
    cv::cuda::createContinuous(rhoBins, thetaBins, CV_32SC1, accumulator);
    // Logging
    auto logger = spdlog::get("logger");
    logger->info("Hough lines accumulator size = {}x{}", accumulator.rows, accumulator.cols);

    // Compact sparse masked points matrix to a more manageable dataset
    thrust::device_vector<uint2> points;
    streamCompactMask(edgeMask, points);

    // Determine block and grid size
    dim3 blocks(max(1, (unsigned int)ceil(float(points.size()) / float(TILE_SIZE))), 1, 1);
    dim3 threads(TILE_SIZE);

    // Timing stuff
    GpuTimer timer;
    timer.start();

    // Launch kernel with the previously detected mask points
    houghLinesAccumulateKernel<<<blocks, threads>>>(
        points.data().get(), points.size(), maxDist, rhoBinSize, thetaBinSize, accumulator);

    timer.stop();
    cudaDeviceSynchronize();
    checkCudaErrors(cudaGetLastError());

    // Log timing data
    logger->info("houghLinesAccumulateKernel execution took {} ms", timer.getTime());
}

void cuda::houghLinesAccumulate(const cv::Mat& edgeMask,
                                const unsigned int rhoBinSize,
                                const unsigned int thetaBinSize,
                                cv::Mat& accumulator) {
    assert(edgeMask.type() == CV_8UC1);
    cv::cuda::GpuMat d_edgeMask, d_accumulator;

    // Create continuous GPU matrices so that we can pass them to Thrust
    cv::cuda::createContinuous(edgeMask.rows, edgeMask.cols, edgeMask.type(), d_edgeMask);

    // Copy input to GPU
    d_edgeMask.upload(edgeMask);

    houghLinesAccumulate(d_edgeMask, rhoBinSize, thetaBinSize, d_accumulator);

    // Copy result back to CPU
    d_accumulator.download(accumulator);
}

void cuda::houghCirclesAccumulate(const cv::cuda::GpuMat& edgeMask,
                                  const size_t radius,
                                  cv::cuda::GpuMat& accumulator) {
    assert(edgeMask.type() == CV_8UC1);
    static constexpr size_t TILE_SIZE = 16;

    // Allocate space for output
    accumulator.create(edgeMask.rows, edgeMask.cols, CV_32SC1);

    // Logging
    auto logger = spdlog::get("logger");
    logger->info("Hough circles accumulator size = {}x{}", accumulator.rows, accumulator.cols);

    // Compact sparse masked points matrix to a more manageable dataset
    thrust::device_vector<uint2> points;
    streamCompactMask(edgeMask, points);

    // Determine block and grid size
    dim3 blocks(max(1, (unsigned int)ceil(float(points.size()) / float(TILE_SIZE))), 1, 1);
    dim3 threads(TILE_SIZE);

    // Timing stuff
    GpuTimer timer;
    timer.start();

    // Launch circle accumulation kernel
    houghCirclesAccumulateKernel<<<blocks, threads>>>(
        points.data().get(), points.size(), radius, accumulator);

    timer.stop();
    cudaDeviceSynchronize();
    checkCudaErrors(cudaGetLastError());

    // Log timing data
    logger->info("houghCirclesAccumulateKernel execution took {} ms", timer.getTime());
}

void cuda::houghCirclesAccumulate(const cv::Mat& edgeMask,
                                  const size_t radius,
                                  cv::Mat& accumulator) {
    cv::cuda::GpuMat d_edgeMask, d_accumulator;

    // Create continuous GPU matrices so that we can pass them to Thrust
    cv::cuda::createContinuous(edgeMask.rows, edgeMask.cols, edgeMask.type(), d_edgeMask);

    // Copy input matrix to GPU
    d_edgeMask.upload(edgeMask);

    // Run GpuMat implementation
    houghCirclesAccumulate(d_edgeMask, radius, d_accumulator);

    // Copy results back to CPU
    d_accumulator.download(accumulator);
}

void cuda::findLocalMaxima(const cv::cuda::GpuMat& accumulator,
                           const unsigned int numPeaks,
                           const int threshold,
                           std::vector<std::pair<unsigned int, unsigned int>>& localMaxima) {
    static constexpr size_t TILE_SIZE = 16;

    thrust::device_vector<HoughPoint2D> localMaximaPoints(accumulator.rows * accumulator.cols);

    // Determine block and grid size
    dim3 blocks(max(1, (unsigned int)ceil(float(accumulator.cols) / float(TILE_SIZE))),
                max(1, (unsigned int)ceil(float(accumulator.rows) / float(TILE_SIZE))));
    dim3 threads(TILE_SIZE, TILE_SIZE);

    // Timing stuff
    auto logger = spdlog::get("logger");
    GpuTimer timer;
    timer.start();

    findLocalMaximaKernel<<<blocks, threads>>>(accumulator, localMaximaPoints.data().get());

    timer.stop();
    cudaDeviceSynchronize();
    checkCudaErrors(cudaGetLastError());

    // Log timing data
    logger->info("findLocalMaximaKernel execution took {} ms", timer.getTime());

    // Filter by threshold, where threshold is the minimum number of votes required to count as a
    // peak
    timer.start();
    auto threshEnd = thrust::remove_if(
        localMaximaPoints.begin(), localMaximaPoints.end(), MaskAndThreshold(threshold));
    timer.stop();
    logger->info("Filtering local maxima points took {} ms", timer.getTime());

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
                           const unsigned int numPeaks,
                           const int threshold,
                           std::vector<std::pair<unsigned int, unsigned int>>& localMaxima) {
    cv::cuda::GpuMat d_accumulator, d_localMaximaMask;

    // Copy to GPU
    d_accumulator.upload(accumulator);
    findLocalMaxima(d_accumulator, numPeaks, threshold, localMaxima);
}