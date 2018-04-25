#include "../include/common/Utils.h"

#include <spdlog/spdlog.h>

#include <cuda_runtime.h>

// CUDA stream callback
void common::checkCopySuccess(int status, void* userData) {
    auto logger = spdlog::get("file_logger");
    if (status == cudaSuccess) {
        logger->info("Stream successfully copied data to GPU");
    } else {
        logger->error("Stream copy failed!");
    }
}

bool common::makeDir(const std::string& dirPath) {
    const int dirErr = mkdir(dirPath.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
    if (dirErr == -1) {
        // The directory already exists, so there's nothing to do anyway. Return true
        return errno == EEXIST;
    }
    return true;
}