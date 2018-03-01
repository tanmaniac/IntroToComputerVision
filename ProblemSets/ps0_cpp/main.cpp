#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <array>
#include <iostream>
#include <sstream>

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

    outputImage = img2.clone();

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
    outputImage = inputImage.clone();
    // subtract(outputImage, mean, outputImage);
    // divide(outputImage, stdDev, outputImage);
    outputImage -= mean;
    outputImage /= stdDev;
    outputImage *= 10;
    outputImage += mean;
}

void translateImg(const cv::Mat& image, const int xOffset, const int yOffset, cv::Mat& output) {
    output = image.clone();
    cv::Mat translationMat = (cv::Mat_<double>(2, 3) << 1, 0, xOffset, 0, 1, yOffset);
    cv::warpAffine(image, output, translationMat, output.size());
}

void addGaussianNoise(const cv::Mat& image,
                      const double mean,
                      const double sigma,
                      cv::Mat& output) {
    if (image.channels() != 1) {
        std::cerr << "Input to addGaussianNoise() must have 1 channel" << std::endl;
        return;
    }
    output = image.clone();
    cv::Mat noise = cv::Mat(image.size(), CV_8SC1);
    // Generate noise
    cv::randn(noise, cv::Scalar::all(mean), cv::Scalar::all(sigma));
    image.convertTo(output, CV_8SC1);
    cv::addWeighted(output, 1.0, noise, 1.0, 0.0, output);
    output.convertTo(output, image.type());
}

std::string getUsage() {
    std::stringstream ss;
    ss << "Intro to Computer Vision Problem Set 0" << std::endl;
    ss << "    ./ps0 <path to first image> <path to second image>" << std::endl;
    ss << "Example:" << std::endl;
    ss << "    ./ps0 ../Resources/USC_Misc/4.1.03.png ../Resources/USC_Misc/4.1.07.png"
       << std::endl;
    return ss.str();
}

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "Not enough input arguments!" << std::endl;
        std::cerr << getUsage() << std::endl;
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

    // 4c. Shift img1_green to the left by 2 pixels
    cv::Mat translatedGreen;
    translateImg(green, -2, 0, translatedGreen);
    cv::imwrite("ps0-4-c-1.png", translatedGreen);

    // 4d. Subtract the shifted version of img1_green from the original, and save the difference
    // image
    cv::Mat translationDiff = green.clone();
    translationDiff -= translatedGreen;
    cv::imwrite("ps0-4-d-1.png", translationDiff);

    // 5a. Take the original colored image (image 1) and start adding Gaussian noise to the pixels
    // in the green channel. Increase sigma until the noise is somewhat visible
    cv::Mat noisyGreen;
    static constexpr int NOISE_SIGMA = 5;
    addGaussianNoise(green, 0, NOISE_SIGMA, noisyGreen);
    cv::imwrite("ps0-5-a-1.png", noisyGreen);

    // 5b. Now, instead add that amount of noise to the blue channel.
    cv::Mat blue, noisyBlue;
    extractChannel(image, blue, Color::BLUE);
    addGaussianNoise(blue, 0, NOISE_SIGMA, noisyBlue);
    cv::imwrite("ps0-5-b-1.png", noisyBlue);

    return 0;
}