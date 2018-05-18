#pragma once

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <tuple>

class ParticleFilter {
public:
    enum class SimilarityMode { MEAN_SQ_ERR, MEAN_SHIFT_LT };
    // Constructor
    ParticleFilter(const cv::Mat& model,
                   const cv::Size& imSize,
                   const size_t numParticles,
                   const SimilarityMode simMode,
                   const double mseSigma,
                   const double sampleSigma,
                   const cv::Point2f& initModelPos = cv::Point2f(-1, -1),
                   const double alpha = 0.1);

    // Update the particle filter with a new frame
    std::tuple<cv::Point2f, float, float> tick(const cv::Mat& frame);

    // Get reference to particles
    const std::vector<cv::Point2f>& getParticles() const;

    // Draw particles on an image
    void drawParticles(cv::Mat& img, const cv::Scalar& color);

private:
    // Displaces particles by a Gaussian random amount in the x and y directions
    void displaceParticles();

    // Compute similarity between the patch around each particle and the model that we are tracking
    void updateParticles(const cv::Mat& frame);

    // Naive multinomial resampling
    void resampleMultinomial();

    // Update tracking model as a weighted sum of the last model and the current patch in the
    // estimated area
    void updateModel(const cv::Mat& frame, const cv::Point2f& newModelPos);

    // Estimate the new state of the tracked object based on current particle states
    // Returns a tuple: {(x,y) mean of the particles, x variance, y variance}
    std::tuple<cv::Point2f, float, float> estimateState();

    // Computes the similarity of a patch to the model using mean-squared error
    double computeSimilarityToModel(const cv::Mat& candidate);

    // Compute mean-squared error of a patch compared to the model
    double calcMeanSqErr(const cv::Mat& candidate);

    // Computes the histogram of an image patch
    std::vector<cv::Mat> calcHist(const cv::Mat& patch);

    enum class GenParticleMode { UNIFORM, GAUSSIAN };
    // Generate uniform random 2D points
    template <typename T>
    void genParticles(const T xMin,
                      const T xMax,
                      const T yMin,
                      const T yMax,
                      const size_t numParticles,
                      const GenParticleMode mode,
                      std::vector<cv::Point_<T>>& particles,
                      const double sigma = 10,
                      const T xInit = 0,
                      const T yInit = 0);

    // Collection of our particles
    std::vector<cv::Point2f> _particles;
    // Weights for each particle
    std::vector<float> _weights;
    // Model when using mean-shift lite
    std::vector<cv::Mat> _modelHist;
    std::vector<cv::Mat> _lastModelHist;
    // Model when using mean squared error
    cv::Mat _modelPatch;
    cv::Mat _lastModelPatch;
    const cv::Size _modelPatchSize;
    // Weight for updating the model
    const double _alpha;
    // Initial position of the mode
    const cv::Point2f _initModelPos;
    // Search space size
    const cv::Size _imSize;
    // Total number of particles N
    const size_t _numParticles;
    // Which method we should use to compute similarity
    const SimilarityMode _simMode;
    // Sigma for mean squared error
    const double _mseSigma;
    // Sigma for displacement of particles at each step
    const double _sampleSigma;
    // Histogram comparison method
    const cv::HistCompMethods _histComp = cv::HISTCMP_CHISQR;
    // Number of bins per channel for histogram
    const int _histBins = 32;

    // Padded copy of the current frame so we can search all the edges
    cv::Mat _paddedFrame;
};