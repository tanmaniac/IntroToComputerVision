#include "../include/FParse.h"

cv::Mat FParse::parse(const std::string& filePath) {
    return parseAs<float>(filePath);
}

cv::Mat FParse::parse(const std::string& filePath, const char delim) {
    return parseAs<float>(filePath, delim);
}

cv::Mat FParse::parse(const std::string& filePath, const std::string& delim) {
    return parseAs<float>(filePath, delim);
}