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

void sol::runProblem1(Config& config) {
    // Time runtime
    auto logger = spdlog::get(config::STDOUT_LOGGER);
    logger->info("Problem 1 begins");
    auto start = std::chrono::high_resolution_clock::now();

    std::unique_ptr<ParticleFilter> pf;
    static constexpr size_t numParticles = 300;
    cv::Point2f bboxCenter; // Center of bounding box around tracked point
    float xVar, yVar;       // x and y variances

    // Set up a video writer
    cv::VideoWriter videoOut;
    static constexpr int frameRate = 30; // 30 fps

    // Play back the first data set
    bool firstFrame = true;
    std::chrono::duration<double, std::milli> frameTimes;
    int numFrames = 0;
    while (1) {
        cv::Mat frame;
        config._debate._cap >> frame;
        if (frame.empty()) break;

        if (firstFrame) {
            cv::Mat model = frame(cv::Rect(config._debate._bbox, config._debate._bboxSize));
            // Initialize the particle filter
            pf = cpp_upstream::make_unique<ParticleFilter>(
                model, frame.size(), numParticles, 20.f, 10.f);
            // Initialize video writer
            videoOut = cv::VideoWriter(config._outputPathPrefix + "/romney_tracked.avi",
                                       cv::VideoWriter::fourcc('M', 'P', 'E', 'G'),
                                       frameRate,
                                       frame.size());
            firstFrame = false;
            logger->info("Initialized particle filter");
        }

        // Measure frame times
        auto tickStart = std::chrono::high_resolution_clock::now();
        // Update the particle filter
        std::tie(bboxCenter, xVar, yVar) = pf->tick(frame);
        auto tickEnd = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> frameTime = tickEnd - tickStart;
        frameTimes += frameTime;

        // Draw tracked particles and bounding box
        pf->drawParticles(frame, cv::Scalar(0, 255, 0, 0));
        // Get the upper-left coordinates of the new bounding box
        cv::Point2f bbox(bboxCenter.x - config._debate._bboxSize.width / 2,
                         bboxCenter.y - config._debate._bboxSize.height / 2);
        cv::rectangle(frame, cv::Rect(bbox, config._debate._bboxSize), cv::Scalar(255, 0, 255, 0));

        // Write the frame time out to the console
        std::cout << "\rFrame time = " << frameTime.count() << " ms (" << 1000.0 / frameTime.count()
                  << " fps)";

        // Save video and display
        videoOut.write(frame);
        cv::imshow("Frame", frame);
        numFrames++;
        char c = char(cv::waitKey(1));
        if (c == 27) {
            break;
        }
    }
    cv::destroyAllWindows();

    std::cout << std::endl;

    double avgTime = frameTimes.count() / double(numFrames);
    logger->info("Average frame time was {} ms ({} fps)", avgTime, 1000.0 / avgTime);

    auto finish = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> runtime = finish - start;
    logger->info("Problem 1 runtime = {} ms", runtime.count());
}
