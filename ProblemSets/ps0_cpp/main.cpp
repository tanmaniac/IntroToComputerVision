#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <iostream>

namespace Color {
enum Channels : int { BLUE = 0, GREEN = 1, RED = 2 };
};

void swapRedBlue(const cv::Mat& inputImage, cv::Mat& outputImage) {
    std::array<cv::Mat, 3> bgrSplit;
    cv::split(inputImage, bgrSplit.data());

    std::array<cv::Mat, 3> rgb = {bgrSplit[2], bgrSplit[1], bgrSplit[0]};
    cv::merge(rgb.data(), 3, outputImage);
}

void pixelReplacement(const cv::Mat& img1, const cv::Mat& img2, cv::Mat& outputImage) {
    static constexpr size_t NUM_CENTER_PIXELS = 100; // Copy a 100x100 pixel square

    // Compute the box containing the center pixels
    cv::Rect centerPixels((img1.cols / 2) - (NUM_CENTER_PIXELS / 2),
                          (img1.rows / 2) - (NUM_CENTER_PIXELS / 2),
                          NUM_CENTER_PIXELS,
                          NUM_CENTER_PIXELS);

    cv::Mat center(img1, centerPixels);

    outputImage = cv::Mat(img2);

    center.copyTo(outputImage(cv::Rect((img2.cols / 2) - (NUM_CENTER_PIXELS / 2),
                                       (img2.rows / 2) - (NUM_CENTER_PIXELS / 2),
                                       center.cols,
                                       center.rows)));
}

// Subtract the mean from all pixels, then divide by standard deviation, then multiply by 10 (if
// your image is 0 to 255) or by 0.05 (if your image ranges from 0.0 to 1.0). Now add the mean back
// in.
void doArithmeticOperations(const cv::Mat& inputImage,
                            const double mean,
                            const double stdDev,
                            cv::Mat& outputImage) {
    outputImage = cv::Mat(inputImage);
    // subtract(outputImage, mean, outputImage);
    // divide(outputImage, stdDev, outputImage);
    outputImage -= mean;
    outputImage /= stdDev;
    outputImage *= 10;
    outputImage += mean;
}

int main(int argc, char** argv) {
    if (argc != 3) {
        return -1;
    }

    cv::Mat image = cv::imread(argv[1], CV_LOAD_IMAGE_COLOR);
    cv::Mat image2 = cv::imread(argv[2], CV_LOAD_IMAGE_COLOR);

    if (!image.data) {
        std::cerr << "Could not open or find image \"" << argv[1] << "\"" << std::endl;
    }
    if (!image2.data) {
        std::cerr << "Could not open or find image \"" << argv[2] << "\"" << std::endl;
    }

    cv::Mat swappedImg;
    // 2a. Swap red and blue pixels
    swapRedBlue(image, swappedImg);
    cv::imwrite("ps0-2-a-1.png", swappedImg);

    // 2b. Extract green pixels
    cv::Mat green;
    cv::extractChannel(image, green, Color::GREEN);
    cv::imwrite("ps0-2-b-1.png", green);

    // 2c. Extract red pixels
    cv::Mat red;
    extractChannel(image, red, Color::RED);
    cv::imwrite("ps0-2-c-1.png", red);

    // 3a. Take the center square region of 100x100 pixels of monochrome version of image 1 and
    // insert them into the center of monochrome version of image 2
    cv::Mat replaced, redBG;
    extractChannel(image2, redBG, Color::RED);
    pixelReplacement(red, redBG, replaced);
    cv::imwrite("ps0-3-a-1.png", replaced);

    // 4a. What is the min and max of the pixel values of img1_green? What is the mean? What is the
    // standard deviation?
    double min, max;
    cv::minMaxLoc(green, &min, &max);
    cv::Scalar mean, stdDev;
    cv::meanStdDev(green, mean, stdDev);
    std::cout << "Min = " << min << ", Max = " << max << std::endl;
    std::cout << "Mean = " << mean[0] << ", StdDev = " << stdDev[0] << std::endl;

    // 4b. Subtract the mean from all pixels, then divide by standard deviation, then multiply by 10
    // (if your image is 0 to 255) or by 0.05 (if your image ranges from 0.0 to 1.0). Now add the
    // mean back in.
    cv::Mat arithmeticOps;
    doArithmeticOperations(green, mean[0], stdDev[0], arithmeticOps);
    cv::imwrite("ps0-4-b-1.png", arithmeticOps);

    return 0;
}