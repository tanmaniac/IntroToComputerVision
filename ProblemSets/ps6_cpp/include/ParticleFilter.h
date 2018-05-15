#pragma once

#include <opencv2/core/core.hpp>

#include <tuple>

class ParticleFilter {
public:
    // Constructor
    ParticleFilter(const cv::Mat& model,
                   const cv::Size& imSize,
                   const size_t numParticles,
                   const double mseSigma,
                   const double sampleSigma);

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

    // Estimate the new state of the tracked object based on current particle states
    // Returns a tuple: {(x,y) mean of the particles, x variance, y variance}
    std::tuple<cv::Point2f, float, float> estimateState();

    // Generate uniform random 2D points
    template <typename T>
    void genUniformParticles(const T xMin,
                             const T xMax,
                             const T yMin,
                             const T yMax,
                             const size_t numParticles,
                             std::vector<cv::Point_<T>>& particles);

    double computeSimilarityToModel(const cv::Mat& candidate);

    // Collection of our particles
    std::vector<cv::Point2f> _particles;
    // Weights for each particle
    std::vector<float> _weights;
    // Image patch model that particles should try to match to
    const cv::Mat _model;
    // Search space size
    const cv::Size _imSize;
    // Total number of particles N
    const size_t _numParticles;
    // Sigma for mean squared error
    const double _mseSigma;
    // Sigma for displacement of particles at each step
    const double _sampleSigma;
};