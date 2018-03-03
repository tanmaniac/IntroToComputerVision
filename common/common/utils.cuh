#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>

#include <iostream>

// This will output the proper CUDA error strings
// in the event that a CUDA host call returns an error
#define checkCudaErrors(err) __checkCudaErrors(err, __FILE__, __LINE__)

inline void __checkCudaErrors(cudaError_t err, const char* file, const int line) {
    if (cudaSuccess != err) {
        fprintf(stderr,
                "CUDA Driver API error : %s at file <%s>, line %i.\n",
                cudaGetErrorString(err),
                file,
                line);
        exit(-1);
    }
}

__device__ inline uint2 getPosition() {
    return make_uint2(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);
}

__device__ inline unsigned int convert2dTo1d(const uint2 loc, const size_t numCols) {
    return loc.y * numCols + loc.x;
}

void copyRGBAToThrust(const cv::Mat& inputImage, thrust::device_vector<uchar4>& thrustImage) {
    const size_t numRows = inputImage.rows;
    const size_t numCols = inputImage.cols;
    printf("numRows = %lu, numCols = %lu\n", numRows, numCols);

    thrustImage.resize(numRows * numCols);

    // Convert the input image to an RGBA image
    cv::Mat imageRGBA;
    cv::cvtColor(inputImage, imageRGBA, CV_BGR2RGBA);

    const unsigned char* cvPtr = imageRGBA.ptr<unsigned char>(0);
    for (unsigned int i = 0; i < numRows * numCols; i++) {
        thrustImage[i] =
            make_uchar4(cvPtr[4 * i + 0], cvPtr[4 * i + 1], cvPtr[4 * i + 2], cvPtr[4 * i + 3]);
    }
}

void copyThrustToRGBA(const thrust::device_vector<uchar4>& thrustImage,
                      const size_t numRows,
                      const size_t numCols,
                      cv::Mat& image) {
    image.create(numRows, numCols, CV_8UC4);

    unsigned char* cvPtr = image.ptr<unsigned char>(0);
    for (unsigned int i = 0; i < numRows * numCols; i++) {
        uchar4 pixel = thrustImage[i];
        cvPtr[4 * i + 0] = pixel.x;
        cvPtr[4 * i + 1] = pixel.y;
        cvPtr[4 * i + 2] = pixel.z;
        cvPtr[4 * i + 3] = pixel.w;
    }
}

void copyThrustToGrey(const thrust::device_vector<unsigned char>& thrustImage,
                      const size_t numRows,
                      const size_t numCols,
                      cv::Mat& image) {
    image.create(numRows, numCols, CV_8UC1);

    unsigned char* cvPtr = image.ptr<unsigned char>(0);
    for (unsigned int i = 0; i < numRows * numCols; i++) {
        cvPtr[i + 0] = thrustImage[i];
    }
}