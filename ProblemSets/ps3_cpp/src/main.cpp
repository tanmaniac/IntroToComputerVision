#include "../include/Calibration.h"
#include "../include/Config.h"
#include "../include/FParse.h"

#include <spdlog/spdlog.h>

#include <chrono>
#include <iomanip>
#include <iostream>
#include <sstream>

// YAML file containing input parameters
static constexpr char CONFIG_FILE_PATH[] = "../config/ps3.yaml";

std::shared_ptr<spdlog::logger> _logger, _fileLogger;

// Project a point from 3D space to the 2D image plane
cv::Mat project3D(const cv::Mat& projMat, const cv::Mat& pt3d) {
    // No assert here because matrix multiplication in OpenCV already has one
    cv::Mat projected = projMat * pt3d;
    // Last value in projected is the homogeneous value - divide by this to scale correctly to an
    // inhomogeneous point
    projected = projected / projected.at<float>(2, 0);

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
                stream << std::right << std::setw(11) << fm._mat.at<float>(y, x);
            }
            stream << (y < fm._mat.rows - 1 ? "\n" : " ");
        }
        stream << "]" << std::setprecision(9);
        return stream;
    }
};

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

        cv::Mat projection;
        cv::transpose(project3D(params, lastPt3D), projection);
        ss << "\nProjected 3D point \n"
           << FormattedMat(lastPt3D_T) << "\nto 2D point \n"
           << FormattedMat(projection) << std::endl;
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