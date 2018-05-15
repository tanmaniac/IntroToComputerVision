#include "../include/ParticleFilter.h"

#include <boost/functional/hash.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <unordered_set>

ParticleFilter::ParticleFilter(const cv::Mat& model,
                               const cv::Size& imSize,
                               const size_t numParticles,
                               const double mseSigma,
                               const double sampleSigma)
    : _model(model), _imSize(imSize), _numParticles(numParticles), _mseSigma(mseSigma),
      _sampleSigma(sampleSigma) {
    // Initialize particles as uniformly distributed random points in the image
    genUniformParticles(
        0.f, float(imSize.width), 0.f, float(imSize.height), numParticles, _particles);

    // Initialize N weights to 1/N
    _weights.assign(_particles.size(), 1.f / float(_particles.size()));
}

std::tuple<cv::Point2f, float, float> ParticleFilter::tick(const cv::Mat& frame) {
    // Predict the movement of all the particles
    displaceParticles();
    // std::cout << "Displaced particles" << std::endl;
    // Update weights with the movement of those particles
    updateParticles(frame);
    // std::cout << "Updated weights" << std::endl;
    // Resample based on newly computed weights
    resampleMultinomial();
    // Predict the new state
    auto estimate = estimateState();
    return estimate;
}

const std::vector<cv::Point2f>& ParticleFilter::getParticles() const { return _particles; }

void ParticleFilter::drawParticles(cv::Mat& frame, const cv::Scalar& color) {
    for (const auto& p : _particles) {
        cv::circle(frame, p, 1, color, -1);
    }
}

void ParticleFilter::displaceParticles() {
    // Iterate over particles and displace
    cv::RNG rng;
    for (auto& p : _particles) {
        p.x += rng.gaussian(_sampleSigma);
        p.y += rng.gaussian(_sampleSigma);

        // If the particle has exceeded the bounds of the image, then it "dies" and is replaced by a
        // new one
        if (p.x < 0 || p.x >= _imSize.width || p.y < 0 || p.y >= _imSize.height) {
            p.x = rng.uniform(0.f, float(_imSize.width));
            p.y = rng.uniform(0.f, float(_imSize.height));
        }
    }
}

void ParticleFilter::updateParticles(const cv::Mat& frame) {
    const int rows = _model.rows;
    const int cols = _model.cols;

    const int xOffset = std::ceil(float(cols) / 2.f);
    const int yOffset = std::ceil(float(rows) / 2.f);

    cv::Mat framePadded;
    cv::copyMakeBorder(
        frame, framePadded, yOffset, yOffset, xOffset, xOffset, cv::BORDER_REPLICATE);

    // std::cout << "xOffset = " << xOffset << "; yOffset = " << yOffset << "; model rows=" << rows
    //           << ", cols=" << cols << std::endl;

    // Iterate over particles and compute similarity
    double simSum = 0;
    for (int i = 0; i < _numParticles; i++) {
        const auto& p = _particles[i];
        // std::cout << "p.x=" << p.x << " p.y=" << p.y << "; xMin=" << xMin << " yMin=" << yMin
        //           << "; framePadded.cols=" << framePadded.cols << " .rows=" << framePadded.rows
        //           << std::endl;

        // If the particle is outside the bounds of the image, then similarity can no longer be
        // computed. In that case just set the weight to 0.
        if (p.x < 0 || p.x >= _imSize.width || p.y < 0 || p.y >= _imSize.height) {
            _weights[i] = 0;
        } else {
            // cv::Mat candidate = framePadded(cv::Rect(xMin + xOffset, yMin + yOffset, cols,
            // rows));
            cv::Mat candidate = framePadded(cv::Rect(p.x, p.y, cols, rows));

            double similarity = computeSimilarityToModel(candidate);
            simSum += similarity;
            _weights[i] = similarity;
        }
    }

    // Normalize weights
    std::transform(_weights.begin(), _weights.end(), _weights.begin(), [&simSum](float f) -> float {
        return f / simSum;
    });
}

void ParticleFilter::resampleMultinomial() {
    // Compute the cumulative sum of the weights to get a sorted array from 0.f to 1.f
    std::vector<float> cumulativeSum(_numParticles);
    if (_numParticles > 0) {
        cumulativeSum[0] = _weights[0];
    }

    for (int i = 1; i < _numParticles; i++) {
        cumulativeSum[i] = _weights[i] + cumulativeSum[i - 1];
    }

    // Generate N random points and see where they would fit in the array
    std::vector<cv::Point2f> partSampled(_numParticles);
    cv::RNG rng;
    for (int i = 0; i < _numParticles; i++) {
        float val = rng.uniform(0.f, 1.f);
        // Get an iterator to the first index that is greater than the random number
        auto iter = std::upper_bound(cumulativeSum.begin(), cumulativeSum.end(), val);
        int idx = iter - cumulativeSum.begin();
        // Copy that particle to a new vector of particles
        partSampled[i] = _particles[idx];
    }

    // Replace each particle with the particle corresponding with the index found previously
    _particles = partSampled;
}

double ParticleFilter::computeSimilarityToModel(const cv::Mat& candidate) {
    assert(candidate.rows == _model.rows && candidate.cols == _model.cols &&
           candidate.type() == _model.type());

    // Compute mean-squared error
    cv::Mat diff = _model - candidate;
    diff = diff.mul(diff);
    cv::Scalar sums = cv::sum(diff);
    double mse = 0;
    for (int ch = 0; ch < 4; ch++) {
        mse += sums[ch];
    }
    mse /= double(candidate.rows * candidate.cols);
    // Compute the similarity with squared exponential
    return std::exp(-mse / (2 * _mseSigma * _mseSigma));
}

std::tuple<cv::Point2f, float, float> ParticleFilter::estimateState() {
    float xMean = 0;
    float yMean = 0;
    // X and Y variances
    float xVar = 0;
    float yVar = 0;

    // Iterate over particles and sum
    for (const auto& p : _particles) {
        xMean += p.x;
        yMean += p.y;
    }

    xMean /= float(_numParticles);
    yMean /= float(_numParticles);

    // Compute variance
    for (const auto& p : _particles) {
        xVar += (p.x - xMean) * (p.x - xMean);
        yVar += (p.y - yMean) * (p.y - yMean);
    }

    xVar /= float(_numParticles);
    yVar /= float(_numParticles);

    return std::make_tuple(cv::Point2f(xMean, yMean), xVar, yVar);
}

struct PairEqual {
    template <typename T1, typename T2>
    bool operator()(const std::pair<T1, T2>& lhs, const std::pair<T1, T2>& rhs) const {
        return lhs.first == rhs.first && lhs.second == rhs.second;
    }
};

template <typename T>
void ParticleFilter::genUniformParticles(const T xMin,
                                         const T xMax,
                                         const T yMin,
                                         const T yMax,
                                         const size_t numParticles,
                                         std::vector<cv::Point_<T>>& particles) {
    cv::RNG rng;

    // Hold points in a set with constant lookup so we can make sure the generated particles are
    // unique
    std::unordered_set<std::pair<T, T>, boost::hash<std::pair<T, T>>, PairEqual> particleMap;

    size_t count = 0;
    while (count < numParticles) {
        T xVal = rng.uniform(xMin, xMax);
        T yVal = rng.uniform(yMin, yMax);

        auto result = particleMap.emplace(xVal, yVal);
        // Make sure that emplacing was successful
        if (result.second) {
            // The point did not already exist in the map, so add it to the list of particles
            particles.emplace_back(xVal, yVal);
            count++;
        }
    }
}