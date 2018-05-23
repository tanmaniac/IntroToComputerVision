#include "../include/MotionHistory.h"
#include "../include/Config.h"

#include <spdlog/spdlog.h>

#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudafilters.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <cmath>

// Defined in MotionHistory.cu
extern void thresholdDifference(const cv::cuda::GpuMat& src,
                                const double thresh,
                                cv::cuda::GpuMat& dst,
                                cv::cuda::Stream& stream);

extern void motionHistoryKernelCall(cv::cuda::GpuMat& history,
                                    const cv::cuda::GpuMat& binaryMask,
                                    const int tau);

// Functions in mhi namespace

void mhi::frameDifference(const cv::Mat& f1,
                          const cv::Mat& f2,
                          const double thresh,
                          cv::Mat& diff,
                          const cv::Size& blurSize,
                          const double blurSigma) {
    assert(f1.rows == f2.rows && f1.cols == f2.cols && f1.type() == f2.type());
    auto logger = spdlog::get(config::STDOUT_LOGGER);

    const cv::Size imSize = f1.size();
    const int dstType = CV_8UC1;

    diff.create(imSize, dstType);

    // Copy to GPU memory
    cv::cuda::Stream stream;

    cv::cuda::GpuMat d_f1, d_f2;
    d_f1.upload(f1, stream);
    d_f2.upload(f2, stream);
    cv::cuda::GpuMat d_f1Blur(imSize, f1.type()), d_f2Blur(imSize, f2.type()),
        d_diff(imSize, f1.type());

    // Blur input images
    auto gauss = cv::cuda::createGaussianFilter(f1.type(), -1, blurSize, blurSigma);
    gauss->apply(d_f1, d_f1Blur, stream);
    gauss->apply(d_f2, d_f2Blur, stream);
    auto morph = cv::cuda::createMorphologyFilter(
        cv::MORPH_OPEN, dstType, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(7, 7)));

    cv::cuda::subtract(d_f2Blur, d_f1Blur, d_diff, cv::Mat(), -1, stream);
    // logger->info("Found difference");
    // Convert to float and single channel
    cv::cuda::GpuMat d_diffGrey(imSize, dstType);
    if (d_diff.channels() > 1) {
        cv::cuda::cvtColor(d_diff, d_diffGrey, cv::COLOR_RGB2GRAY, 1, stream);
    } else {
        d_diff.convertTo(d_diffGrey, dstType, stream);
    }
    // logger->info("Converted to float");

    // Threshold
    cv::cuda::GpuMat d_thresholded(imSize, dstType);
    thresholdDifference(d_diffGrey, thresh, d_thresholded, stream);

    // Clean up noise with a morphological operator
    cv::cuda::GpuMat d_morphed(imSize, dstType);
    morph->apply(d_thresholded, d_morphed, stream);

    // Copy back to CPU memory
    d_morphed.download(diff, stream);
}

void mhi::calcMotionHistory(cv::Mat& history, const cv::Mat& binaryMask, const int tau) {
    assert(tau > 0); // Tau must be a time in frames greater than 0
    assert(history.rows == binaryMask.rows && history.cols == binaryMask.cols);
    assert(history.type() == binaryMask.type() &&
           history.type() == CV_8UC1); // Only support unsigned 8-bit int

    // Upload to GPU
    cv::cuda::Stream stream;
    cv::cuda::GpuMat d_history, d_binaryMask;
    d_history.upload(history, stream);
    d_binaryMask.upload(binaryMask, stream);

    // Call kernel
    motionHistoryKernelCall(d_history, d_binaryMask, tau);

    // Copy back to CPU
    d_history.download(history, stream);
}