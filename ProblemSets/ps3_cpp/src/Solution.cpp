#include "Solution.h"

#include "../include/Calibration.h"
#include "../include/Config.h"
#include "../include/Fundamental.h"

#include <spdlog/fmt/ostr.h>
#include <spdlog/spdlog.h>

#include <opencv2/core/mat.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <algorithm>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <vector>

static std::mt19937 rng; // Mersenne twister engine

// Struct to allow prettier printing of cv::Mats
struct FormattedMat {
    cv::Mat _mat;

    FormattedMat(const cv::Mat& mat) : _mat(mat) {}

    friend std::ostream& operator<<(std::ostream& stream, const FormattedMat& fm) {
        std::ios init(NULL);
        init.copyfmt(stream);
        // Align right
        stream << "[" << std::setprecision(5);
        for (size_t y = 0; y < fm._mat.rows; y++) {
            stream << (y == 0 ? " " : "  ");
            for (size_t x = 0; x < fm._mat.cols; x++) {
                stream << std::left << std::setw(13) << fm._mat.at<float>(y, x);
            }
            stream << (y < fm._mat.rows - 1 ? "\n" : " ");
        }
        stream << "]";
        // Restore original formatting
        stream.copyfmt(init);
        return stream;
    }
};

// Project a set of points from 3D space to the 2D image plane
cv::Mat project3D(const cv::Mat& projMat, const cv::Mat& pt3d) {
    // No assert here because matrix multiplication in OpenCV already has one
    cv::Mat projected = projMat * pt3d;
    // Last value in projected is the homogeneous value - divide by this to scale correctly to an
    // inhomogeneous point
    for (size_t col = 0; col < projected.cols; col++) {
        float s = projected.at<float>(2, col);
        projected.col(col) = projected.col(col) / s;
    }

    return projected;
}

/**
 * \brief Generate k unique random values for calibration constraints in range [a, b] and return a
 * vector of those points. Also generate j more unique random values to use as test points.
 *
 * \param min Minimum random value
 * \param max Maximum random value
 * \param k Number of unique randoms to generate
 * \param constraintPts Vector of constraint point output values
 * \param testPts Vector of test point output values
 */
template <typename T>
void genUniqueRands(const T min,
                    const T max,
                    const size_t k,
                    const size_t j,
                    std::vector<T>& constraintPts,
                    std::vector<T>& testPts) {
    assert(max > min);
    std::vector<T> nums(max - min);
    std::iota(nums.begin(), nums.end(), min);

    // Random shuffle
    std::shuffle(nums.begin(), nums.end(), rng);

    int i = 0;
    for (; i < k; i++) {
        constraintPts.push_back(nums[i]);
    }
    for (; i < k + j; i++) {
        testPts.push_back(nums[i]);
    }
}

/**
 * \brief Build a matrix of points from a given matrix of possible points and a vector of the
 * columns to use to build the new matrix.
 *
 * \param pts Possible points to be selected
 * \param cols Columns to use in new matrix
 * \param out Output matrix
 */
template <typename T>
void buildMatFromCols(const cv::Mat& pts, const std::vector<T>& cols, cv::Mat& out) {
    // Iterate over the vector of columns and append that column to the output
    out.create(pts.rows, 0, pts.type());

    for (const auto& col : cols) {
        // Horizontal concatenation
        cv::hconcat(out, pts.col(col), out);
    }
}

/**
 * \brief Draw epipolar lines on a given image
 *
 * \param img Input/output image on which the lines will be drawn
 * \param epiLines Set of epipolar lines to draw.
 * \param color Color in which the lines should be drawn.
 */
void drawEpipolarLines(cv::Mat& img, const cv::Mat& epiLines, const cv::Scalar color) {
    auto fileLogger = spdlog::get(config::FILE_LOGGER);
    size_t rows = img.rows;
    size_t cols = img.cols;

    // Compute left and right edge intersections for the image
    cv::Mat P_UL = (cv::Mat_<float>(3, 1) << 0, 0, 1);               // Upper left
    cv::Mat P_BL = (cv::Mat_<float>(3, 1) << 0, rows - 1, 1);        // Bottom left
    cv::Mat P_UR = (cv::Mat_<float>(3, 1) << cols - 1, 0, 1);        // Upper right
    cv::Mat P_BR = (cv::Mat_<float>(3, 1) << cols - 1, rows - 1, 1); // Bottom right

    // Compute the lines corresponding to the left and right edges of the image
    cv::Mat I_L = P_UL.cross(P_BL);
    cv::Mat I_R = P_UR.cross(P_BR);

    // Iterate over columns of lines and compute the endpoints, and then draw them
    for (int col = 0; col < epiLines.cols; col++) {
        cv::Mat curLine = epiLines.col(col);
        cv::Mat P_iL = curLine.cross(I_L);
        cv::Mat P_iR = curLine.cross(I_R);

        // Scale correctly
        P_iL /= P_iL.at<float>(2, 0);
        P_iR /= P_iR.at<float>(2, 0);

        // Transpose just for logging (not actually used in calculation)
        cv::Mat P_iL_T, P_iR_T;
        cv::transpose(P_iL, P_iL_T);
        cv::transpose(P_iR, P_iR_T);
        fileLogger->info("@pt{}: P_iL={}, P_iR={}", col, P_iL_T, P_iR_T);

        cv::line(img,
                 cv::Point2f(P_iL.at<float>(0, 0), P_iL.at<float>(1, 0)),
                 cv::Point2f(P_iR.at<float>(0, 0), P_iR.at<float>(1, 0)),
                 color);
    }
}

void solution::runProblem1a(const Config& config) {
    // Time runtime
    auto logger = spdlog::get(config::STDOUT_LOGGER);
    logger->info("Problem 1a begins");
    auto start = std::chrono::high_resolution_clock::now();

    // Get the last 3D normalized point so we can check our m matrix later
    cv::Mat lastPt3D = config._points._pts3DNorm.col(config._points._pts3DNorm.cols - 1);
    lastPt3D.push_back(1.f);
    cv::Mat lastPt2D = config._points._picANorm.col(config._points._picANorm.cols - 1);

    // Row-vector version of the last point so that we can print it more easily
    cv::Mat lastPt3D_T;
    cv::transpose(lastPt3D, lastPt3D_T);

    // Compute projection matrix
    {
        auto sol = calib::solveLeastSquares(config._points._picANorm, config._points._pts3DNorm);
        cv::Mat params = sol.reshape(0, 3);

        cv::Mat projection = project3D(params, lastPt3D);

        // Compute residual
        double residual = cv::norm(projection.rowRange(0, 2), lastPt2D);

        // Transpose of projected point (just for logging)
        cv::Mat projection_T;
        cv::transpose(projection, projection_T);
        logger->info("Calibration parameters (using normal least squares):\n{}\nProjected "
                     "3D point\n{}\nto 2D point\n{}\nResidual = {}",
                     FormattedMat(params),
                     lastPt3D_T,
                     projection_T,
                     residual);
    }

    {
        auto sol = calib::solveSVD(config._points._picANorm, config._points._pts3DNorm);
        cv::Mat params = sol.reshape(0, 3);

        cv::Mat projection = project3D(params, lastPt3D);

        // Compute residual
        double residual = cv::norm(projection.rowRange(0, 2), lastPt2D);

        // Transpose of projected point (just for logging)
        cv::Mat projection_T;
        cv::transpose(projection, projection_T);
        logger->info("Calibration parameters (using singular value decomposition):\n{}\nProjected "
                     "3D point\n{}\nto 2D point\n{}\nResidual = {}",
                     FormattedMat(params),
                     lastPt3D_T,
                     projection_T,
                     residual);
    }

    auto finish = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> runtime = finish - start;
    logger->info("Problem 1a runtime = {} ms", runtime.count());
}

void solution::runProblem1bc(const Config& config) {
    // Time runtime
    auto logger = spdlog::get(config::STDOUT_LOGGER);
    logger->info("Problems 1b and 1c begin");
    auto start = std::chrono::high_resolution_clock::now();

    //---------------- Part b ----------------
    // Use three point set sizes - 8, 12, and 16
    size_t numTestPts = 4;
    size_t numIters = 10; // 10 iterations per constraint set
    std::vector<int> constraintPts, testPts;
    std::vector<size_t> numConstraints{8, 12, 16};
    // Data structure to hold all the residuals
    cv::Mat residuals(numIters, numConstraints.size(), CV_64F);

    // Overall minimum residual, corresponding parameters, and at what constraints this was found
    double ovMinResidual = std::numeric_limits<double>::max();
    cv::Mat ovMinParams;
    size_t ovConstraintSize;

    // Seed random number generator
    rng.seed(*(config._mersenneSeed));

    for (int numConstraintsIdx = 0; numConstraintsIdx < numConstraints.size();
         numConstraintsIdx++) {
        // Minimum residual for this number of constraints
        double minResidual = std::numeric_limits<double>::max();
        // Parameter matrix corresponding to the minimum residual
        cv::Mat minParams;

        for (size_t iter = 0; iter < numIters; iter++) {
            genUniqueRands(0,
                           config._points._pts3D.cols,
                           numConstraints[numConstraintsIdx],
                           numTestPts,
                           constraintPts,
                           testPts);

            // Build matrices for constraints and test points
            cv::Mat constraints3D, constraints2D, tests3D, tests2D;
            buildMatFromCols(config._points._pts3D, constraintPts, constraints3D);
            buildMatFromCols(config._points._picB, constraintPts, constraints2D);
            buildMatFromCols(config._points._pts3D, testPts, tests3D);
            buildMatFromCols(config._points._picB, testPts, tests2D);

            // Solve using least squares
            auto sol = calib::solveLeastSquares(constraints2D, constraints3D);
            cv::Mat params = sol.reshape(0, 3);

            // Transform points
            tests3D.push_back(cv::Mat::ones(1, tests3D.cols, tests3D.type()));
            tests2D.push_back(cv::Mat::ones(1, tests2D.cols, tests2D.type()));
            cv::Mat projected = project3D(params, tests3D);

            double residual = 0;
            for (size_t col = 0; col < projected.cols; col++) {
                residual +=
                    cv::norm(projected.col(col).rowRange(0, 2), tests2D.col(col).rowRange(0, 2));
            }
            residual /= double(tests2D.cols);

            // Add this residual to the list of residuals
            residuals.at<double>(iter, numConstraintsIdx) = residual;

            if (residual < minResidual) {
                minResidual = residual;
                minParams = params;
            }

            // Clear vectors of test/constraint points
            constraintPts.clear();
            testPts.clear();
        }

        if (minResidual < ovMinResidual) {
            ovMinResidual = minResidual;
            ovMinParams = minParams;
            ovConstraintSize = numConstraints[numConstraintsIdx];
        }
    }

    // Print all residuals
    logger->info("All computed residuals:\n{}", residuals);

    logger->info("\nMinimum residual: {}\nFound with constraint size: {}\nComputed parameters:\n{}",
                 ovMinResidual,
                 ovConstraintSize,
                 FormattedMat(ovMinParams));

    //---------------- Part c ----------------
    // Find center of camera
    cv::Mat Q = ovMinParams.colRange(0, 3);
    cv::Mat m4 = ovMinParams.col(3);
    cv::Mat centerOfCam = -1.f * Q.inv() * m4;

    logger->info("Center of camera:\n{}", centerOfCam);

    auto finish = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> runtime = finish - start;
    logger->info("Problems 1b and 1c runtime = {} ms", runtime.count());
}

void solution::runProblem2(const Config& config) {
    // Time runtime
    auto logger = spdlog::get(config::STDOUT_LOGGER);
    logger->info("Problem 2 begins");
    auto start = std::chrono::high_resolution_clock::now();

    //----------------- Part a -------------------
    cv::Mat fMatEst = fundamental::solveLeastSquares(config._points._picA, config._points._picB);
    fMatEst = fMatEst.reshape(0, 3);

    logger->info("Fundamental matrix estimate:\n{}", FormattedMat(fMatEst));

    //----------------- Part b ------------------
    cv::Mat fMat = fundamental::rankReduce(fMatEst);

    logger->info("Fundamental matrix with rank = 2\n{}", FormattedMat(fMat));

    //------------------ Part c -------------------
    cv::Mat ptsA = config._points._picA.clone();
    cv::Mat ptsB = config._points._picB.clone();

    // Append a row of ones to the input points to make 3xn matrices
    ptsA.push_back(cv::Mat::ones(1, ptsA.cols, ptsA.type()));
    ptsB.push_back(cv::Mat::ones(1, ptsB.cols, ptsB.type()));

    // Compute epipolar lines
    cv::Mat ptsB_T;
    cv::transpose(ptsB, ptsB_T);
    cv::Mat linesA = ptsB_T * fMat;
    cv::transpose(linesA, linesA);
    cv::Mat linesB = fMat * ptsA;

    // Make copies of the input images to draw on
    cv::Mat picA = config._images._picA.clone();
    cv::Mat picB = config._images._picB.clone();

    // Draw epipolar lines on A and B images
    drawEpipolarLines(picA, linesA, CV_RGB(0, 0xFF, 0));
    drawEpipolarLines(picB, linesB, CV_RGB(0, 0xFF, 0));
    cv::imwrite(config._outputPathPrefix + "/ps3-2-c-1.png", picA);
    cv::imwrite(config._outputPathPrefix + "/ps3-2-c-2.png", picB);

    auto finish = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> runtime = finish - start;
    logger->info("Problem 2 runtime = {} ms", runtime.count());
}

void solution::runExtraCredit(const Config& config) {
    // Time runtime
    auto logger = spdlog::get(config::STDOUT_LOGGER);
    auto fileLogger = spdlog::get(config::FILE_LOGGER);
    logger->info("Extra credit problem begins");
    auto start = std::chrono::high_resolution_clock::now();

    //--------------- Part d ---------------
    // Normalize input points
    // Make copies of the input point sets first
    cv::Mat ptsA = config._points._picA.clone();
    cv::Mat ptsB = config._points._picB.clone();

    // Append row of ones to input points to make 3xn matrices
    ptsA.push_back(cv::Mat::ones(1, ptsA.cols, ptsA.type()));
    ptsB.push_back(cv::Mat::ones(1, ptsB.cols, ptsB.type()));

    // Compute the mean of the x and y coordinates of points from image A and image B
    cv::Mat meanA = (cv::Mat_<float>(2, 1) << cv::mean(ptsA.row(0))[0], cv::mean(ptsA.row(1))[0]);
    cv::Mat meanB = (cv::Mat_<float>(2, 1) << cv::mean(ptsB.row(0))[0], cv::mean(ptsB.row(1))[0]);

    fileLogger->info("meanA = {}\nmeanB = {}", meanA, meanB);

    // Find maximum values in point sets
    cv::Mat absPtsA = cv::abs(ptsA.rowRange(0, 2));
    cv::Mat absPtsB = cv::abs(ptsB.rowRange(0, 2));
    double maxA, maxB;
    cv::minMaxLoc(cv::abs(ptsA), nullptr, &maxA);
    cv::minMaxLoc(cv::abs(ptsB), nullptr, &maxB);

    fileLogger->info("Max of A = {}; Max of B = {}", maxA, maxB);

    // Build normalization matrices
    cv::Mat transformA, transformB;
    // First we'll do it for image A
    {
        // Such ugly formatting smh
        cv::Mat scale = (cv::Mat_<float>(3, 3) << 1.f / maxA, 0, 0, 0, 1.f / maxA, 0, 0, 0, 1);
        cv::Mat offset = (cv::Mat_<float>(3, 3) << 1,
                          0,
                          -meanA.at<float>(0, 0),
                          0,
                          1,
                          -meanA.at<float>(1, 0),
                          0,
                          0,
                          1);
        transformA = scale * offset;
    }
    // Image B
    {
        cv::Mat scale = (cv::Mat_<float>(3, 3) << 1.f / maxB, 0, 0, 0, 1.f / maxB, 0, 0, 0, 1);
        cv::Mat offset = (cv::Mat_<float>(3, 3) << 1,
                          0,
                          -meanB.at<float>(0, 0),
                          0,
                          1,
                          -meanB.at<float>(1, 0),
                          0,
                          0,
                          1);
        transformB = scale * offset;
    }

    logger->info("\nTransform matrix T_a:\n{}\nTransform matrix T_b:\n{}",
                 FormattedMat(transformA),
                 FormattedMat(transformB));

    // Normalize the input points with the transform matrices that were just computed
    cv::Mat ptsAPrime = transformA * ptsA;
    cv::Mat ptsBPrime = transformB * ptsB;

    // Compute the normalized fundamental matrix. We can throw away row 3 of the point matrices
    // since they are all ones.
    cv::Mat FHat =
        fundamental::solveLeastSquares(ptsAPrime.rowRange(0, 2), ptsBPrime.rowRange(0, 2));
    FHat = FHat.reshape(0, 3);
    // Reduce rank from 3 to 2
    FHat = fundamental::rankReduce(FHat);
    logger->info("\nFundamental matrix F_Hat:\n{}", FormattedMat(FHat));

    //----------------- Part e -----------------
    // Compute the "better" fundamental matrix, F = transformB^T * FHat * transformA
    cv::Mat transformB_T;
    cv::transpose(transformB, transformB_T);
    cv::Mat F = transformB_T * FHat * transformA;

    logger->info("\n\"Better\" fundamental matrix F:\n{}", FormattedMat(F));

    // Now we can draw the epipolar lines like in the previous problem
    // Compute epipolar lines
    cv::Mat ptsB_T;
    cv::transpose(ptsB, ptsB_T);
    cv::Mat linesA = ptsB_T * F;
    cv::transpose(linesA, linesA);
    cv::Mat linesB = F * ptsA;

    // Make copies of the input images to draw on
    cv::Mat picA = config._images._picA.clone();
    cv::Mat picB = config._images._picB.clone();

    // Draw epipolar lines on A and B images
    drawEpipolarLines(picA, linesA, CV_RGB(0, 0xFF, 0));
    drawEpipolarLines(picB, linesB, CV_RGB(0, 0xFF, 0));
    cv::imwrite(config._outputPathPrefix + "/ps3-2-e-1.png", picA);
    cv::imwrite(config._outputPathPrefix + "/ps3-2-e-2.png", picB);

    // Finish timing
    auto finish = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> runtime = finish - start;
    logger->info("Extra credit problem runtime = {} ms", runtime.count());
}
