#pragma once

#include <algorithm>
#include <cmath>

// Useful utilities that can be used for both CUDA and CPU code

namespace common {

// Compute block size by dividing two numbers and round up, clipping the minimum output to 1.
template <typename numType, typename denomType>
std::size_t divRoundUp(const numType num, const denomType denom) {
    return std::max(std::size_t(1), std::size_t(ceil(float(num) / float(denom))));
}

// CUDA stream callback
void checkCopySuccess(int status, void* userData);

bool makeDir(const std::string& dirPath);

}; // namespace common