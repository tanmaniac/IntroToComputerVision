#include "Solution.h"
#include <common/make_unique.h>
#include "../include/ParticleFilter.h"

#include <spdlog/spdlog.h>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/videoio/videoio.hpp>

#include <memory>
#include <tuple>
#include <unordered_set>

void pfDriver(Config::Tracking& tracking,
              const Config::PFConf& pfConf,
              const ParticleFilter::SimilarityMode simMode,
              const std::string& outputPrefix,
              const std::unordered_set<int>& saveFrames,
              const std::string& vidOutputPath = "") {
    using clock = std::chrono::high_resolution_clock;
    auto logger = spdlog::get(config::STDOUT_LOGGER);

    std::unique_ptr<ParticleFilter> pf;
    cv::Point2f bboxCenter; // Center of bounding box around tracked point
    float xVar, yVar;       // x and y variances

    // Figure out if we should save the video
    bool saveVideo = (vidOutputPath.compare("") == 0 ? false : true);
    // Set up a video writer
    cv::VideoWriter videoOut;
    static constexpr int frameRate = 30; // 30 fps

    // Play back the first data set
    bool firstFrame = true;
    std::chrono::duration<double, std::milli> frameTimes;
    int numFrames = 0;
    while (1) {
        cv::Mat frame;
        tracking._cap >> frame;
        if (frame.empty()) break;

        if (firstFrame) {
            cv::Mat model = frame(cv::Rect(tracking._bbox, tracking._bboxSize));
            // Initialize the particle filter
            pf = cpp_upstream::make_unique<ParticleFilter>(model,
                                                           frame.size(),
                                                           pfConf._numParticles,
                                                           simMode,
                                                           pfConf._mseSigma,
                                                           pfConf._dynamicsSigma,
                                                           tracking._bbox);
            // Initialize video writer
            if (saveVideo) {
                videoOut = cv::VideoWriter(vidOutputPath,
                                           cv::VideoWriter::fourcc('M', 'P', 'E', 'G'),
                                           frameRate,
                                           frame.size());
            }
            firstFrame = false;
            logger->info("Initialized particle filter");
        }

        // Measure frame times
        auto tickStart = clock::now();
        // Update the particle filter
        std::tie(bboxCenter, xVar, yVar) = pf->tick(frame);
        auto tickEnd = clock::now();
        std::chrono::duration<double, std::milli> frameTime = tickEnd - tickStart;
        frameTimes += frameTime;

        // Draw tracked particles and bounding box
        pf->drawParticles(frame, cv::Scalar(0, 255, 0, 0));
        // Get the upper-left coordinates of the new bounding box
        cv::Point2f bbox(bboxCenter.x - tracking._bboxSize.width / 2,
                         bboxCenter.y - tracking._bboxSize.height / 2);
        cv::rectangle(frame, cv::Rect(bbox, tracking._bboxSize), cv::Scalar(255, 0, 255, 0));

        // Write the frame time out to the console
        std::cout << "\rTick time = " << frameTime.count() << " ms (" << 1000.0 / frameTime.count()
                  << " fps)";

        // Save video if necessary
        if (saveVideo) videoOut.write(frame);
        // Save frame if necessary
        auto frameSearch = saveFrames.find(numFrames);
        if (frameSearch != saveFrames.end()) {
            cv::imwrite(outputPrefix + "-f" + std::to_string(numFrames) + ".png", frame);
        }
        // Display
        cv::imshow("Frame", frame);
        numFrames++;
        char c = char(cv::waitKey(1));
        if (c == 27) {
            break;
        }
    }
    // Reset the video stream
    tracking._cap.set(cv::CAP_PROP_POS_FRAMES, 0);
    cv::destroyAllWindows();

    std::cout << std::endl;

    double avgTime = frameTimes.count() / double(numFrames);
    logger->info("Average tick time was {} ms ({} fps)", avgTime, 1000.0 / avgTime);
}

void sol::runProblem1(Config& config) {
    // Time runtime
    using clock = std::chrono::high_resolution_clock;
    auto logger = spdlog::get(config::STDOUT_LOGGER);
    logger->info("Problem 1 begins");
    auto start = clock::now();

    pfDriver(config._debate,
             config._pfConf1,
             ParticleFilter::SimilarityMode::MEAN_SQ_ERR,
             config._outputPathPrefix + "/ps6-1-a",
             {{28, 84, 144}});

    // Part e, tracking head with noisy footage
    pfDriver(config._noisyDebate,
             config._pfConf1Noisy,
             ParticleFilter::SimilarityMode::MEAN_SQ_ERR,
             config._outputPathPrefix + "/ps6-1-e",
             {{14, 32, 46}});

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

    // Create a new tracking config object with Romney's hand as the bounding box
    Config::Tracking handTracking;
    handTracking._cap = config._debate._cap;
    handTracking._bbox = cv::Point2f(540, 385); // x,y coordinates of Mitt Romney's hand
    handTracking._bboxSize = cv::Size(73, 87);  // Width and height of Romney's hand

    pfDriver(handTracking,
             config._pfConf2,
             ParticleFilter::SimilarityMode::MEAN_SQ_ERR,
             config._outputPathPrefix + "/ps6-2-a",
             {{15, 50, 150}});

    // Noisy video
    handTracking._cap = config._noisyDebate._cap;
    pfDriver(handTracking,
             config._pfConf2Noisy,
             ParticleFilter::SimilarityMode::MEAN_SQ_ERR,
             config._outputPathPrefix + "/ps6-2-b",
             {{15, 50, 150}});

    auto finish = clock::now();
    std::chrono::duration<double, std::milli> runtime = finish - start;
    logger->info("Problem 2 runtime = {} ms", runtime.count());
}

void sol::runProblem3(Config& config) {
    // Time runtime
    using clock = std::chrono::high_resolution_clock;
    auto logger = spdlog::get(config::STDOUT_LOGGER);
    logger->info("Problem 3 begins");
    auto start = clock::now();

    pfDriver(config._debate,
             config._pfConf3Head,
             ParticleFilter::SimilarityMode::MEAN_SHIFT_LT,
             config._outputPathPrefix + "/ps6-3-a",
             {{28, 84, 144}});

    // Create a new tracking config object with Romney's hand as the bounding box
    Config::Tracking handTracking;
    handTracking._cap = config._debate._cap;
    handTracking._bbox = cv::Point2f(540, 385); // x,y coordinates of Mitt Romney's hand
    handTracking._bboxSize = cv::Size(73, 87);  // Width and height of Romney's hand

    // Track hand
    pfDriver(handTracking,
             config._pfConf3Hand,
             ParticleFilter::SimilarityMode::MEAN_SHIFT_LT,
             config._outputPathPrefix + "/ps6-3-b",
             {{15, 50, 140}});

    auto finish = clock::now();
    std::chrono::duration<double, std::milli> runtime = finish - start;
    logger->info("Problem 3 runtime = {} ms", runtime.count());
}