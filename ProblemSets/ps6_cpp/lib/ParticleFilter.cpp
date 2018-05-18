#include "../include/ParticleFilter.h"

#include <boost/functional/hash.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <numeric>
#include <unordered_set>

ParticleFilter::ParticleFilter(const cv::Mat& model,
                               const cv::Size& imSize,
                               const size_t numParticles,
                               const SimilarityMode simMode,
                               const double mseSigma,
                               const double sampleSigma,
                               const cv::Point2f& initModelPos,
                               const double alpha)
    : _modelPatch(model), _lastModelPatch(model), _modelPatchSize(model.size()), _imSize(imSize),
      _numParticles(numParticles), _simMode(simMode), _mseSigma(mseSigma),
      _sampleSigma(sampleSigma), _initModelPos(initModelPos), _alpha(alpha) {
    // If the similarity mode uses mean-squared error, then the model is just the input image patch
    if (simMode == SimilarityMode::MEAN_SHIFT_LT) {
        // Model is the histogram of the image patch
        _modelHist = calcHist(model);
    }
    std::cout << "Initialized" << std::endl;

    if (initModelPos.x == -1 && initModelPos.y == -1) {
        // No initial position, so generate uniform particles
        genParticles(0.f,
                     float(imSize.width),
                     0.f,
                     float(imSize.height),
                     numParticles,
                     GenParticleMode::UNIFORM,
                     _particles);
    } else {
        // Use initial position to generate particles
        float modelCenterX = initModelPos.x + float(model.cols) / 2.f;
        float modelCenterY = initModelPos.y + float(model.rows) / 2.f;
        genParticles(0.f,
                     float(imSize.width),
                     0.f,
                     float(imSize.height),
                     numParticles,
                     GenParticleMode::GAUSSIAN,
                     _particles,
                     _sampleSigma,
                     modelCenterX,
                     modelCenterY);
    }

    // Initialize N weights to 1/N
    _weights.assign(_particles.size(), 1.f / float(_particles.size()));
}

std::tuple<cv::Point2f, float, float> ParticleFilter::tick(const cv::Mat& frame) {
    // Predict the movement of all the particles
    displaceParticles();
    // Update weights with the movement of those particles
    updateParticles(frame);
    // Resample based on newly computed weights
    resampleMultinomial();
    // Predict the new state
    cv::Point2f bestEstPos;
    auto estimate = estimateState();
    // Update model
    std::tie(bestEstPos, std::ignore, std::ignore) = estimate;
    updateModel(frame, bestEstPos);

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
    }
}

void ParticleFilter::updateParticles(const cv::Mat& frame) {
    const int rows = _modelPatchSize.height;
    const int cols = _modelPatchSize.width;

    const int xOffset = std::ceil(float(cols) / 2.f);
    const int yOffset = std::ceil(float(rows) / 2.f);

    cv::copyMakeBorder(
        frame, _paddedFrame, yOffset, yOffset, xOffset, xOffset, cv::BORDER_REPLICATE);

    // Iterate over particles and compute similarity
    double simSum = 0;
    for (int i = 0; i < _numParticles; i++) {
        const auto& p = _particles[i];

        // If the particle is outside the bounds of the image, then similarity can no longer be
        // computed. In that case just set the weight to 0.
        if (p.x < 0 || p.x >= _imSize.width || p.y < 0 || p.y >= _imSize.height) {
            _weights[i] = 0;
        } else {
            // Get a sample patch from the image
            cv::Mat candidate = _paddedFrame(cv::Rect(p, _modelPatchSize));

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
    if (_simMode == SimilarityMode::MEAN_SQ_ERR) {
        double mse = calcMeanSqErr(candidate);
        // Compute the similarity with squared exponential
        return std::exp(-mse / (2 * _mseSigma * _mseSigma));
    } else /* if (_simMode == SimilarityMode::MEAN_SHIFT_LT) */ {
        std::vector<cv::Mat> candidateHist = calcHist(candidate);
        double comp = 0;
        for (int ch = 0; ch < candidateHist.size(); ch++) {
            comp += cv::compareHist(_modelHist[ch], candidateHist[ch], _histComp);
        }
        comp /= candidateHist.size();
        return std::exp(-comp);
    }
}

double ParticleFilter::calcMeanSqErr(const cv::Mat& candidate) {
    assert(candidate.rows == _modelPatch.rows && candidate.cols == _modelPatch.cols &&
           candidate.type() == _modelPatch.type());

    // Compute mean-squared error
    cv::Mat diff = _modelPatch - candidate;
    diff = diff.mul(diff);
    cv::Scalar sums = cv::sum(diff);
    double mse = 0;
    for (int ch = 0; ch < 4; ch++) {
        mse += sums[ch];
    }
    mse /= double(candidate.rows * candidate.cols);
}

std::vector<cv::Mat> ParticleFilter::calcHist(const cv::Mat& patch) {
    // Calculate histogram for each channel, then push that channel's hist back onto a vector
    std::vector<cv::Mat> hists;
    for (int i = 0; i < patch.channels(); i++) {
        cv::Mat hist;
        cv::calcHist(std::vector<cv::Mat>{patch},
                     std::vector<int>{i},
                     cv::Mat(),
                     hist,
                     std::vector<int>{_histBins},
                     std::vector<float>{0, 256});

        cv::normalize(hist, hist, 1, 0, cv::NORM_L2, -1);
        hists.push_back(hist);
    }

    return hists;
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

void ParticleFilter::updateModel(const cv::Mat& frame, const cv::Point2f& newModelPos) {
    // Padded frame was already computed by updateParticles()
    cv::Mat newPatch = _paddedFrame(cv::Rect(newModelPos, _modelPatchSize));
    if (_simMode == SimilarityMode::MEAN_SQ_ERR) {
        // Save the old model
        _lastModelPatch = _modelPatch;
        // Update the model
        _modelPatch = _alpha * newPatch + (1.0 - _alpha) * _lastModelPatch;
    } else {
        _lastModelPatch = _modelPatch;
        _lastModelHist = _modelHist;
        cv::Mat blended = _alpha * newPatch + (1.0 - _alpha) * _lastModelPatch;
        _modelHist = calcHist(blended);
    }
}

struct PairEqual {
    template <typename T1, typename T2>
    bool operator()(const std::pair<T1, T2>& lhs, const std::pair<T1, T2>& rhs) const {
        return lhs.first == rhs.first && lhs.second == rhs.second;
    }
};

template <typename T>
void ParticleFilter::genParticles(const T xMin,
                                  const T xMax,
                                  const T yMin,
                                  const T yMax,
                                  const size_t numParticles,
                                  const GenParticleMode mode,
                                  std::vector<cv::Point_<T>>& particles,
                                  const double sigma,
                                  const T xInit,
                                  const T yInit) {
    cv::RNG rng;

    // Hold points in a set with constant lookup so we can make sure the generated particles are
    // unique
    std::unordered_set<std::pair<T, T>, boost::hash<std::pair<T, T>>, PairEqual> particleMap;

    size_t count = 0;
    T xVal, yVal;
    while (count < numParticles) {
        if (mode == GenParticleMode::UNIFORM) {
            xVal = rng.uniform(xMin, xMax);
            yVal = rng.uniform(yMin, yMax);
        } else /* if (mode == GenParticleMode::GAUSSIAN) */ {
            xVal = rng.gaussian(sigma) + xInit;
            yVal = rng.gaussian(sigma) + yInit;
        }

        auto result = particleMap.emplace(xVal, yVal);
        // Make sure that emplacing was successful
        if (result.second) {
            // The point did not already exist in the map, so add it to the list of particles
            particles.emplace_back(xVal, yVal);
            count++;
        }
    }
}