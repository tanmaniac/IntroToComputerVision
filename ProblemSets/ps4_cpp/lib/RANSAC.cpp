#include "../include/RANSAC.h"
#include "../include/Config.h"

#include <spdlog/spdlog.h>

#include <iostream>
#include <iterator>
#include <numeric>
#include <vector>

// Mersenne twister engine
static std::mt19937 rng;
static bool seeded = false;

inline float euclidianDist(const cv::Point& p, const cv::Point& q) {
    cv::Point diff = p - q;
    return cv::sqrt(diff.x * diff.x + diff.y * diff.y);
}

void ransac::seed(std::shared_ptr<std::seed_seq> seq) {
    if (!seeded) {
        rng.seed(*seq);
        seeded = true;
    }
}

std::tuple<cv::Mat, std::vector<int>, double> ransac::solve(const std::vector<cv::Point2f>& srcPts,
                                                            const std::vector<cv::Point2f>& destPts,
                                                            const TransformType whichTransform,
                                                            const int ransacReprojThresh,
                                                            const int maxIters,
                                                            const double minConsensusRatio) {
    // Source and training dataset sizes must be the same
    assert(srcPts.size() == destPts.size());
    const size_t MINIMUM_SET = static_cast<size_t>(whichTransform);

    // Create vector of indices to select sample set and test set
    const size_t numPts = srcPts.size();
    std::vector<int> indices(numPts);
    std::iota(indices.begin(), indices.end(), 0);

    // Mersenne twister engine
    // std::random_device rd;
    // rng.seed(rd());

    std::vector<int> consensusSet;
    double consensusRatio = 0;
    int iterations = 0;
    cv::Mat transform;

    while (consensusRatio < minConsensusRatio && iterations < maxIters) {
        // Random shuffle the indices
        std::shuffle(indices.begin(), indices.end(), rng);

        // Compute transform matrix based on which transform type we're using
        switch (whichTransform) {
        case TransformType::TRANSLATION: {
            cv::Point2f pt1(srcPts[indices[0]]), pt1Prime(destPts[indices[0]]);

            // clang-format off
            // Translation matrix is the same as similarity, but with no rotation
            transform = (cv::Mat_<float>(2, 3) << 1, 0, pt1Prime.x - pt1.x,
                                                  0, 1, pt1Prime.y - pt1.y);
            // clang-format on
            break;
        }
        case TransformType::SIMILARITY: {
            // The first two make up the sample set; the rest are used for testing
            cv::Point2f pt1(srcPts[indices[0]]), pt2(srcPts[indices[1]]),
                pt1Prime(destPts[indices[0]]), pt2Prime(destPts[indices[1]]);
            // clang-format off
            // We have a system of equations
            //  [u'      [a  -b   c   [u
            //   v']  =   b   a   d] * v]
            cv::Mat A = (cv::Mat_<float>(4, 4) <<   pt1.x, -pt1.y, 1, 0,
                                                    pt1.y,  pt1.x, 0, 1,
                                                    pt2.x, -pt2.y, 1, 0,
                                                    pt2.y,  pt2.x, 0, 1);
            cv::Mat b = (cv::Mat_<float>(4, 1) <<   pt1Prime.x, 
                                                    pt1Prime.y, 
                                                    pt2Prime.x, 
                                                    pt2Prime.y);
            // clang-format on
            // Solve system Ax = b
            cv::Mat x;
            cv::solve(A, b, x);

            // std::cout << "Matrix x = \n" << x << std::endl;
            // clang-format off
            transform = (cv::Mat_<float>(2, 3) << x.at<float>(0, 0), -x.at<float>(1, 0), x.at<float>(2, 0),
                                                  x.at<float>(1, 0), x.at<float>(0, 0), x.at<float>(3, 0));
            // clang-format on
            break;
        }
        case TransformType::AFFINE: {
            // Take the first three points as the sample set
            cv::Point2f pt1(srcPts[indices[0]]), pt2(srcPts[indices[1]]), pt3(srcPts[indices[2]]),
                pt1Prime(destPts[indices[0]]), pt2Prime(destPts[indices[1]]),
                pt3Prime(destPts[indices[2]]);

            // clang-format off
            // Treat the affine transform matrix as a 3x3 with the last row [0, 0, 1]
            // -> Pad the columns of the three input points with 1s to make 3x3 matrices
            //      P' = T*P
            // Where P is a 3x3 of source points, P' is a 3x3 of dest points, and T is the 3x3 
            // transformation matrix. T can then be computed with
            //      T = P'*P^(-1)
            cv::Mat Pprime = (cv::Mat_<float>(3, 3) << pt1Prime.x, pt2Prime.x, pt3Prime.x,
                                                       pt1Prime.y, pt2Prime.y, pt3Prime.y,
                                                           1     ,     1     ,     1      );
            cv::Mat P = (cv::Mat_<float>(3, 3) << pt1.x, pt2.x, pt3.x,
                                                  pt1.y, pt2.y, pt3.y,
                                                    1  ,   1  ,   1   );
            //clang-format on
            transform = Pprime * P.inv();
            transform = transform.rowRange(0, 2);                                     
        }
        }

        // Iterate over the rest of the points and find the consensus set
        std::vector<int> curConsensusSet;
        for (int idx = MINIMUM_SET; idx < numPts; idx++) {
            cv::Point2f pt(srcPts[indices[idx]]), ptPrime(destPts[indices[idx]]);
            cv::Mat testX = (cv::Mat_<float>(3, 1) << pt.x, pt.y, 1);
            cv::Mat testB = transform * testX;
            // std::cout << "result = \n" << testB << std::endl;
            // Compute euclidian distance between the transformed point and the actual point
            float dist =
                euclidianDist(cv::Point2f(testB.at<float>(0, 0), testB.at<float>(1, 0)), ptPrime);
            // std::cout << "Distance = " << dist << std::endl;

            // If the distance is small enough, add it to the consensus set
            if (dist <= ransacReprojThresh) {
                curConsensusSet.push_back(idx);
            }
        }

        // If the consensus ratio for this set is greater than the current largest consensus ratio,
        // set the largest consensus ratio and set to this iteration's
        double curConsensusRatio = double(curConsensusSet.size()) / double(numPts);
        if (curConsensusRatio > consensusRatio) {
            consensusRatio = curConsensusRatio;
            consensusSet = curConsensusSet;
        }
        iterations++;
    }

    auto flogger = spdlog::get(config::FILE_LOGGER);
    flogger->info("RANSAC took {} iterations", iterations);

    return std::make_tuple(transform, consensusSet, consensusRatio);
}
