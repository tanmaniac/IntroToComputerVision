#include "../include/Harris.h"

#include <opencv2/core/cuda.hpp>
#include <opencv2/cudafilters.hpp>

void harris::getGradients(const cv::Mat& in, int kernelSize, cv::Mat& outX, cv::Mat& outY) {
    // Kernel size must be 1, 3, 5, or 7
    assert(kernelSize == 1 || kernelSize == 3 || kernelSize == 5 || kernelSize == 7);
    cv::Mat inFlt;
    in.convertTo(inFlt, CV_32F);

    // Set up async stream
    cv::cuda::Stream stream;
    auto sobelX = cv::cuda::createSobelFilter(inFlt.type(), inFlt.type(), 1, 0, kernelSize);
    auto sobelY = cv::cuda::createSobelFilter(inFlt.type(), inFlt.type(), 0, 1, kernelSize);

    outX.create(inFlt.rows, inFlt.cols, inFlt.type());
    outY.create(inFlt.rows, inFlt.cols, inFlt.type());

    // Copy to GPU memory
    cv::cuda::GpuMat d_in, d_outX, d_outY;
    d_in.upload(inFlt, stream);
    d_outX.upload(outX, stream);
    d_outY.upload(outY, stream);

    // Apply sobel operator in X and Y direction
    sobelX->apply(d_in, d_outX, stream);
    sobelY->apply(d_in, d_outY, stream);
    d_outX.download(outX, stream);
    d_outY.download(outY, stream);
}