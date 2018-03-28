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