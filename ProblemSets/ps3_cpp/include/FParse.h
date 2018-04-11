#pragma once

#include <opencv2/core/core.hpp>

#include <boost/lexical_cast.hpp>
#include <boost/tokenizer.hpp>

#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

// Parses files representing points in an image or in 3D space and stores the points in vectors of
// cv::Mat.

namespace FParse {
/**
 * \brief Parse an input file delimited with spaces into a vector of column-vectors representing 2D
 * or 3D points.
 *
 * \param filePath Path to the input file.
 */
cv::Mat parse(const std::string& filePath);

/**
 * \brief Parse an input file delimited with spaces into a vector of column-vectors representing 2D
 * or 3D points. While parsing, convert those values into a specific type.
 *
 * \param filePath Path to the input file.
 */
template <typename T>
cv::Mat parseAs(const std::string& filePath);

/**
 * \brief overload of parse() using a custom delimiter.
 */
cv::Mat parse(const std::string& filePath, const char delim);

/**
 * \brief Overload of parseAs<T>() using a custom delimiter.
 */
template <typename T>
cv::Mat parseAs(const std::string& filePath, const char delim);

/**
 * \brief Overload of parse() using a string-type delimiter.
 */
cv::Mat parse(const std::string& filePath, const std::string& delim);

/**
 * \brief Overload of parseAs<T>() using a string-type delimiter.
 */
template <typename T>
cv::Mat parseAs(const std::string& filePath, const std::string& delim);

// Deduces the OpenCV type (CV_32FC1, etc) of a given type.
template <typename T>
int detectCvType();

}; // namespace FParse

//---------- Implementations for templated functions ----------

template <typename T>
cv::Mat FParse::parseAs(const std::string& filePath) {
    return parseAs<T>(filePath, " ");
}

template <typename T>
cv::Mat FParse::parseAs(const std::string& filePath, const char delim) {
    return parseAs<T>(filePath, std::string(1, delim));
}

template <typename T>
cv::Mat FParse::parseAs(const std::string& filePath, const std::string& delim) {
    static_assert(std::is_arithmetic<T>::value, "Template parameter must be an arithmetic type");
    cv::Mat out;
    typedef boost::tokenizer<boost::char_separator<char>> tokenizer;
    boost::char_separator<char> separator(delim.c_str());

    // Open file
    std::ifstream input(filePath);
    if (input.is_open()) {
        std::string line;
        bool isFirstLine = true;
        while (std::getline(input, line)) {
            tokenizer tokens(line, separator);
            // Column-vector that we'll be pushing values to. Automatically deduce the type.
            cv::Mat point(0, 1, detectCvType<T>());
            // Iterate over the string and push back each tokenized value.
            size_t count = 0;
            for (auto tokenIter = tokens.begin(); tokenIter != tokens.end(); tokenIter++) {
                auto val = boost::lexical_cast<T>(*tokenIter);
                point.push_back(val);
                if (isFirstLine) count++;
            }
            if (isFirstLine) {
                // Initialize the output to be the correct size
                out.create(count, 0, detectCvType<T>());
                isFirstLine = false;
            }
            cv::hconcat(out, point, out);
        }
    }
    return out;
}

template <typename T>
int FParse::detectCvType() {
    static_assert(std::is_arithmetic<T>::value, "Template parameter must be an arithmetic type");
    if (std::is_same<T, int8_t>::value) {
        return CV_8SC1;
    } else if (std::is_same<T, uint8_t>::value) {
        return CV_8UC1;
    } else if (std::is_same<T, int16_t>::value) {
        return CV_16SC1;
    } else if (std::is_same<T, uint16_t>::value) {
        return CV_16UC1;
    } else if (std::is_same<T, int32_t>::value) {
        return CV_32SC1;
    } else if (std::is_same<T, float>::value) {
        return CV_32FC1;
    } else if (std::is_same<T, double>::value) {
        return CV_64FC1;
    } else {
        throw std::invalid_argument("Requested type has no corresponding matrix type in OpenCV");
    }
}