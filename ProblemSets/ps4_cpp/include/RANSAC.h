#pragma once

#include <opencv2/core/core.hpp>

#include <memory>
#include <random>
#include <tuple>

namespace ransac {
enum class TransformType { TRANSLATION = 1, SIMILARITY = 2, AFFINE = 3 };
/**
 * \brief Gets translation between a set of source points and destination points. Returns a pair
 * where the first element is the similarity transformation matrix and the second element is a
 * vector of indices indicating the largest consensus set between the two point sets.
 *
 */
std::tuple<cv::Mat, std::vector<int>, double> solve(const std::vector<cv::Point2f>& srcPts,
                                                    const std::vector<cv::Point2f>& destPts,
                                                    const TransformType whichTransform,
                                                    const int ransacReprojThresh = 3,
                                                    const int maxIters = 2000,
                                                    const double minConsensusRatio = 0.75);

// Seeds the random number generator used by RANSAC
void seed(std::shared_ptr<std::seed_seq> seq);
} // namespace ransac