#include "../include/FParse.h"

std::vector<cv::Mat> FParse::parse(const std::string& filePath) {
    return parseAs<float>(filePath);
}

std::vector<cv::Mat> FParse::parse(const std::string& filePath, const char delim) {
    return parseAs<float>(filePath, delim);
}

std::vector<cv::Mat> FParse::parse(const std::string& filePath, const std::string& delim) {
    return parseAs<float>(filePath, delim);
}