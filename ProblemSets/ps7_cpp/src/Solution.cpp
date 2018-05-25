#include "Solution.h"

#include "../include/Matching.h"
#include "../include/Moments.h"
#include "../include/MotionHistory.h"

#include <spdlog/fmt/ostr.h>

#include <opencv2/highgui/highgui.hpp>

#include <tuple>
#include <unordered_set>

// First value in output is a vector of difference images; second output in pair is a vector of MHI
// images
std::pair<std::vector<cv::Mat>, std::vector<cv::Mat>>
    mhiHelper(Config& config,
              const Config::MHI& mhiConf,
              const std::string& filename,
              const std::unordered_set<size_t>& saveDiffFrames,
              const std::unordered_set<size_t>& saveMHIFrames,
              const bool vis = true) {
    using clock = std::chrono::high_resolution_clock;
    auto logger = spdlog::get(config::STDOUT_LOGGER);
    cv::VideoCapture vid = config.openVid(filename);

    std::vector<cv::Mat> diffFrames, mhiFrames;

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
        std::chrono::duration<double, std::milli> frameTimes;
        while (1) {
            cv::Mat frame;
            vid >> frame;
            if (frame.empty()) break;

            // Measure frame times
            auto tickStart = clock::now();
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

            auto tickEnd = clock::now();
            std::chrono::duration<double, std::milli> frameTime = tickEnd - tickStart;
            frameTimes += frameTime;

            // Visualization and frame saving
            if (saveDiffFrames.count(frameNum)) {
                diffFrames.push_back(diff.clone());
            }
            if (saveMHIFrames.count(frameNum)) {
                // std::cout << "Saving frame " << frameNum << std::endl;
                mhiFrames.push_back(history.clone());
            }
            if (vis) {
                cv::Mat diffNorm, historyNorm;
                cv::normalize(diff, diffNorm, 0, 255, cv::NORM_MINMAX, CV_8UC1);
                cv::normalize(history, historyNorm, 0, 255, cv::NORM_MINMAX, CV_8UC1);
                cv::imshow("difference", diffNorm);
                cv::imshow("history", historyNorm);
            }
            // Write the frame time out to the console
            std::cout << "\rMHI update time = " << frameTime.count() << " ms ("
                      << 1000.0 / frameTime.count() << " fps)";
            char c = char(cv::waitKey(1));
            if (c == 27) {
                break;
            }

            // Keep track of frame number
            frameNum++;
        }
        cv::destroyAllWindows();
        std::cout << std::endl;
        double avgTime = frameTimes.count() / double(frameNum - 1);
        logger->info("Average MHI update time was {} ms ({} fps) over {} frames",
                     avgTime,
                     1000.0 / avgTime,
                     frameNum);
    }
    return std::make_pair(diffFrames, mhiFrames);
}

void normAndSave(const cv::Mat& img, const std::string& filename) {
    cv::Mat norm;
    cv::normalize(img, norm, 0, 255, cv::NORM_MINMAX);
    cv::imwrite(filename, norm);
}

// Computes all the MHIs for each person and each action and places them in the output vector
// mhiFrames. Corresponding actions for each computed MHI is placed in the actions vector, which
// is used for the KNN classifier. Each person corresponding to each action is placed in the person
// vector.
void getAllMHIs(Config& config,
                std::vector<cv::Mat>& mhiFrames,
                std::vector<int>& actions,
                std::vector<int>& people) {
    auto logger = spdlog::get(config::STDOUT_LOGGER);
    // Iterate over each action
    std::vector<Config::MHI> actionConfs{
        {config._mhiAction1, config._mhiAction2, config._mhiAction3}};
    for (int action = 1; action <= 3; action++) {
        Config::MHI curConf = actionConfs[action - 1];
        // Iterate over each person
        for (int person = 1; person <= 3; person++) {
            // Iterate over each trial
            for (int trial = 1; trial <= 3; trial++) {
                std::vector<cv::Mat> frame;
                std::string vidFile = "PS7A" + std::to_string(action) + "P" +
                                      std::to_string(person) + "T" + std::to_string(trial);
                logger->info("Computing MHI for {}.avi", vidFile);
                std::tie(std::ignore, frame) = mhiHelper(
                    config, curConf, vidFile, {{}}, {{config.lastFrameOfAction(vidFile)}}, false);
                if (frame.size() == 1) {
                    mhiFrames.push_back(frame[0]);
                    actions.push_back(action);
                    people.push_back(person);
                    normAndSave(frame[0], config._outputPathPrefix + "/" + vidFile + ".png");
                } else {
                    logger->error("Could not get MHI for {} at given frame time ({})",
                                  vidFile,
                                  config.lastFrameOfAction(vidFile));
                }
            }
        }
    }
}

// Which moment type to use for confusion matrix
// MU = central moment
// ETA = scale-invariant moment
// BOTH = both mu and eta
enum class MomentType { MU, ETA, BOTH };

// Arranges training data in a more useful format.
// featuresMat is an Nx2M matrix, where N is the number of data points and M is the number of
// computed central moments. There are 2M columns since we use mu and eta as features.
// labelsMat is an Nx1 matrix of classes corresponding to the data points in the set.
void arrangeTrainingData(const std::vector<std::vector<std::pair<float, float>>>& features,
                         const std::vector<int>& labels,
                         const MomentType whichMoment,
                         cv::Mat& featuresMat,
                         cv::Mat& labelsMat) {
    assert(features.size() == labels.size());
    size_t rows = labels.size();
    assert(rows > 0);
    size_t cols = (whichMoment == MomentType::BOTH) ? features[0].size() * 2 : features[0].size();
    assert(cols > 0);

    // Allocate space
    featuresMat.create(rows, cols, CV_32F);
    labelsMat = cv::Mat(labels, true); // We can just copy this
    labelsMat.convertTo(labelsMat, CV_32F);

    for (size_t r = 0; r < rows; r++) {
        for (size_t c = 0; c < cols; c += (whichMoment == MomentType::BOTH) ? 2 : 1) {
            if (whichMoment == MomentType::MU) {
                featuresMat.at<float>(r, c) = features[r][c].first;
            } else if (whichMoment == MomentType::ETA) {
                featuresMat.at<float>(r, c) = features[r][c].second;
            } else {
                featuresMat.at<float>(r, c) = features[r][c].first;
                featuresMat.at<float>(r, c + 1) = features[r][c].second;
            }
        }
    }
}

void sol::runProblem1(Config& config) {
    // Time runtime
    using clock = std::chrono::high_resolution_clock;
    auto logger = spdlog::get(config::STDOUT_LOGGER);
    logger->info("Problem 1 begins");
    auto start = clock::now();

    // Part a
    std::unordered_set<size_t> diffFramesToSave{{10, 20, 30}};
    std::vector<cv::Mat> diffFrames;
    std::tie(diffFrames, std::ignore) =
        mhiHelper(config, config._mhiAction1, "PS7A1P1T1", diffFramesToSave, {{}});
    // Save difference frames
    for (int i = 0; i < diffFrames.size(); i++) {
        normAndSave(diffFrames[i],
                    config._outputPathPrefix + "/ps7-1-a-" + std::to_string(i + 1) + ".png");
    }

    // Part b
    std::vector<cv::Mat> mhiFrames;
    std::tie(std::ignore, mhiFrames) = mhiHelper(
        config, config._mhiAction1, "PS7A1P2T1", {{}}, {{config.lastFrameOfAction("PS7A1P2T1")}});
    normAndSave(mhiFrames[0], config._outputPathPrefix + "/ps7-1-b-1.png");

    std::tie(std::ignore, mhiFrames) = mhiHelper(
        config, config._mhiAction2, "PS7A2P3T2", {{}}, {{config.lastFrameOfAction("PS7A2P3T2")}});
    normAndSave(mhiFrames[0], config._outputPathPrefix + "/ps7-1-b-2.png");

    std::tie(std::ignore, mhiFrames) = mhiHelper(
        config, config._mhiAction3, "PS7A3P2T2", {{}}, {{config.lastFrameOfAction("PS7A3P2T2")}});
    normAndSave(mhiFrames[0], config._outputPathPrefix + "/ps7-1-b-3.png");

    auto finish = clock::now();
    std::chrono::duration<double, std::milli> runtime = finish - start;
    logger->info("Problem 1 runtime = {} ms", runtime.count());
}

void sol::runProblem2(Config& config) {
    // Time runtime
    using clock = std::chrono::high_resolution_clock;
    auto logger = spdlog::get(config::STDOUT_LOGGER);
    logger->info("Problem 2 begins");
    auto start = clock::now();

    std::vector<cv::Mat> MHIs;
    std::vector<int> actions, people;
    getAllMHIs(config, MHIs, actions, people);
    logger->info("Computed {} MHIs", MHIs.size());

    // Compute motion energy from motion history
    std::vector<cv::Mat> MEIs;
    mhi::energyFromHistory(MHIs, MEIs);
    logger->info("Converted {} MHIs to MEIs", MEIs.size());

    // Normalize motion history images to floating point with max at 1.0
    for (auto& mhi : MHIs) {
        cv::normalize(mhi, mhi, 1.0, 0.0, cv::NORM_INF, CV_32FC1);
    }

    // Central moment orders to compute
    std::vector<std::pair<int, int>> momentOrders = {
        {2, 0}, {0, 2}, {1, 2}, {2, 1}, {2, 2}, {3, 0}, {0, 3}};
    // Compute central moments (mu and eta) for motion history images
    std::vector<std::vector<std::pair<float, float>>> mhiMoments, meiMoments;
    {
        auto mhiIter = MHIs.begin();
        auto meiIter = MEIs.begin();
        for (; mhiIter != MHIs.end() && meiIter != MEIs.end(); mhiIter++, meiIter++) {
            mhiMoments.push_back(moments::centralMoment(*mhiIter, momentOrders));
            meiMoments.push_back(moments::centralMoment(*meiIter, momentOrders));
        }
    }

    {
        cv::Mat features, labels;
        arrangeTrainingData(mhiMoments, actions, MomentType::MU, features, labels);

        cv::Mat confusion;
        matching::naiveConfusionMatrix(features, labels, confusion);
        logger->info("\nConfusion matrix with central moments=\n{}", confusion);
    }

    {
        cv::Mat features, labels;
        arrangeTrainingData(mhiMoments, actions, MomentType::ETA, features, labels);

        cv::Mat confusion;
        matching::naiveConfusionMatrix(features, labels, confusion);
        logger->info("\nConfusion matrix with scale-invariant moments=\n{}", confusion);
    }

    // Part b : more intelligent confusion matrices
    // Convert "people" vector to a matrix
    cv::Mat peopleMat(people, true);
    peopleMat.convertTo(peopleMat, CV_32S);
    {
        cv::Mat features, labels;
        arrangeTrainingData(mhiMoments, actions, MomentType::BOTH, features, labels);

        std::vector<cv::Mat> confusions;
        matching::confusionMatrix(features, labels, peopleMat, 3, confusions);

        // Iterate over confusions matrices and print
        for (int p = 0; p < confusions.size(); p++){
            std::string whichPerson = (p == confusions.size() - 1) ? "average of all" : "person " + std::to_string(p + 1);
            logger->info("\nConfusion matrix for {}=\n{}", whichPerson, confusions[p]);
        }
    }

    auto finish = clock::now();
    std::chrono::duration<double, std::milli> runtime = finish - start;
    logger->info("Problem 2 runtime = {} ms", runtime.count());
}