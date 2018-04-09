#include "../include/Calibration.h"
#include "../include/Config.h"
#include "../include/FParse.h"

#include <spdlog/spdlog.h>

#include <algorithm>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <sstream>
#include <vector>

// YAML file containing input parameters
static constexpr char CONFIG_FILE_PATH[] = "../config/ps3.yaml";

std::shared_ptr<spdlog::logger> _logger, _fileLogger;

// Project a set of points from 3D space to the 2D image plane
cv::Mat project3D(const cv::Mat& projMat, const cv::Mat& pt3d) {
    // No assert here because matrix multiplication in OpenCV already has one
    cv::Mat projected = projMat * pt3d;
    // Last value in projected is the homogeneous value - divide by this to scale correctly to an
    // inhomogeneous point
    for (size_t col = 0; col < projected.cols; col++) {
        float s = projected.at<float>(2, col);
        projected.col(col) = projected.col(col) / s;
        // projected = projected / projected.at<float>(2, 0);
    }

    return projected;
}

// Struct to allow prettier printing of cv::Mats
struct FormattedMat {
    cv::Mat _mat;

    FormattedMat(const cv::Mat& mat) : _mat(mat) {}

    friend std::ostream& operator<<(std::ostream& stream, const FormattedMat& fm) {
        // Align right
        stream << "[" << std::setprecision(5);
        for (size_t y = 0; y < fm._mat.rows; y++) {
            stream << (y == 0 ? " " : "  ");
            for (size_t x = 0; x < fm._mat.cols; x++) {
                stream << std::right << std::setw(13) << fm._mat.at<float>(y, x);
            }
            stream << (y < fm._mat.rows - 1 ? "\n" : " ");
        }
        stream << "]" << std::setprecision(9);
        return stream;
    }
};

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
    std::shuffle(nums.begin(), nums.end(), std::mt19937{std::random_device{}()});

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

void runProblem1(const Config& config) {
    // Time runtime
    _logger->info("Problem 1 begins");
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
        // Sol is missing the last value (m_2,3, which was set to 1), so append 1 to it
        sol.push_back(1.f);
        cv::Mat params = sol.reshape(0, 3);

        std::stringstream ss;
        ss << "Calibration parameters (using normal least squares): \n" << FormattedMat(params);

        cv::Mat projection = project3D(params, lastPt3D);
        cv::Mat projection_T; // Transpose of projected point
        cv::transpose(projection, projection_T);
        ss << "\nProjected 3D point \n"
           << FormattedMat(lastPt3D_T) << "\nto 2D point \n"
           << FormattedMat(projection_T) << std::endl;
        // Compute residual
        double residual = cv::norm(projection.rowRange(0, 2), lastPt2D);
        ss << "Residual = " << residual;
        _logger->info("{}", ss.str());
    }

    {
        auto sol = calib::solveSVD(config._points._picANorm, config._points._pts3DNorm);
        cv::Mat params = sol.reshape(0, 3);

        std::stringstream ss;
        ss << "Calibration parameters (using singular value decomposition): \n"
           << FormattedMat(params);

        cv::Mat projection = project3D(params, lastPt3D);
        cv::Mat projection_T; // Transpose of projected point
        cv::transpose(projection, projection_T);
        ss << "\nProjected 3D point \n"
           << FormattedMat(lastPt3D_T) << "\nto 2D point \n"
           << FormattedMat(projection_T) << std::endl;
        // Compute residual
        double residual = cv::norm(projection.rowRange(0, 2), lastPt2D);
        ss << "Residual = " << residual;
        _logger->info("{}", ss.str());
    }

    //----------------- Part b --------------------
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

            // Solve using singular value decomposition
            // auto sol = calib::solveSVD(constraints2D, constraints3D);
            // cv::Mat params = sol.reshape(0, 3);
            auto sol = calib::solveLeastSquares(constraints2D, constraints3D);
            // Sol is missing the last value (m_2,3, which was set to 1), so append 1 to it
            sol.push_back(1.f);
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

    // Print
    {
        std::stringstream ss;
        ss << "Minimum residual = " << ovMinResidual
           << "\nFound with constraints = " << ovConstraintSize << "\nComputed parameters:\n"
           << FormattedMat(ovMinParams) << std::endl;
        _logger->info("{}", ss.str());
    }

    //------------------ Part c --------------------
    // Find center of camera
    cv::Mat Q = ovMinParams.colRange(0, 3);
    cv::Mat m4 = ovMinParams.col(3);
    cv::Mat centerOfCam = -1.f * Q.inv() * m4;

    {
        std::stringstream ss;
        ss << "Center of camera:\n" << FormattedMat(centerOfCam) << std::endl;
        _logger->info("{}", ss.str());
    }

    auto finish = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> runtime = finish - start;
    _logger->info("Problem 1 runtime = {} ms", runtime.count());
}

int main() {
    // Set up loggers
    std::vector<spdlog::sink_ptr> sinks;
    auto colorStdoutSink = std::make_shared<spdlog::sinks::ansicolor_stdout_sink_mt>();
    auto fileSink = std::make_shared<spdlog::sinks::simple_file_sink_mt>("ps3.log");
    sinks.push_back(colorStdoutSink);
    sinks.push_back(fileSink);
    _logger = std::make_shared<spdlog::logger>("logger", begin(sinks), end(sinks));
    // File logger is just for CUDA kernel outputs
    _fileLogger = std::make_shared<spdlog::logger>("file_logger", fileSink);
    spdlog::register_logger(_logger);
    spdlog::register_logger(_fileLogger);

    Config config(CONFIG_FILE_PATH);
    runProblem1(config);
}