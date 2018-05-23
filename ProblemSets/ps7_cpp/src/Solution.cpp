#include "Solution.h"

#include "../include/MotionHistory.h"

#include <opencv2/highgui/highgui.hpp>

#include <unordered_set>

void mhiHelper(Config& config,
               const Config::MHI& mhiConf,
               const std::string& filename,
               const std::string& diffPrefix,
               const std::unordered_set<int>& saveDiffFrames,
               const std::string& mhiPrefix,
               const std::unordered_set<int>& saveMHIFrames,
               const bool vis = true) {
    using clock = std::chrono::high_resolution_clock;
    auto logger = spdlog::get(config::STDOUT_LOGGER);
    cv::VideoCapture vid = config.openVid(filename);

    cv::Mat lastFrame;
    vid >> lastFrame;
    if (lastFrame.empty()) {
        logger->error("Could not retrieve frame");
    } else {
        int frameNum = 1;
        cv::Mat history = cv::Mat::zeros(lastFrame.size(), CV_8UC1);
        if (vis) {
            cv::imshow("history", history);
            cv::moveWindow("history", lastFrame.cols + 10, 0);
        }
        while (1) {
            cv::Mat frame;
            vid >> frame;
            if (frame.empty()) break;

            cv::Mat diff;
            mhi::frameDifference(lastFrame,
                                 frame,
                                 mhiConf._threshold,
                                 diff,
                                 mhiConf._preBlurSize,
                                 mhiConf._preBlurSigma);

            // Compute motion history image
            mhi::calcMotionHistory(history, diff, mhiConf._tau);

            // Update last frame
            lastFrame = frame;

            // Visualization and frame saving
            cv::Mat diffNorm, historyNorm;
            cv::normalize(diff, diffNorm, 0, 255, cv::NORM_MINMAX, CV_8UC1);
            cv::normalize(history, historyNorm, 0, 255, cv::NORM_MINMAX, CV_8UC1);
            if (saveDiffFrames.count(frameNum))
                cv::imwrite(diffPrefix + "-diff" + std::to_string(frameNum) + ".png", diffNorm);
            if (saveMHIFrames.count(frameNum))
                cv::imwrite(mhiPrefix + "-mhi" + std::to_string(frameNum) + ".png", historyNorm);
            if (vis) {
                cv::imshow("difference", diffNorm);
                cv::imshow("history", historyNorm);
            }
            char c = char(cv::waitKey(1));
            if (c == 27) {
                break;
            }

            // Keep track of frame number
            frameNum++;
        }
    }
}

void sol::runProblem1(Config& config) {
    // Time runtime
    using clock = std::chrono::high_resolution_clock;
    auto logger = spdlog::get(config::STDOUT_LOGGER);
    logger->info("Problem 1 begins");
    auto start = clock::now();

    mhiHelper(config,
              config._mhi1,
              "PS7A1P1T1",
              config._outputPathPrefix + "/ps7-1-a",
              {{10, 20, 30}},
              config._outputPathPrefix + "/ps7-1-b",
              {{60}});

    auto finish = clock::now();
    std::chrono::duration<double, std::milli> runtime = finish - start;
    logger->info("Problem 1 runtime = {} ms", runtime.count());
}

void sol::runProblem2(Config& config) { ; }