#include "../include/common/GpuTimer.h"

GpuTimer::GpuTimer() {
    cudaEventCreate(&_start);
    cudaEventCreate(&_stop);
}

GpuTimer::~GpuTimer() {
    cudaEventSynchronize(_stop);
}

void GpuTimer::start() {
    cudaEventRecord(_start);
}

void GpuTimer::stop() {
    cudaEventRecord(_stop);
    cudaEventSynchronize(_stop);
}

float GpuTimer::getTime() {
    cudaEventElapsedTime(&_runtimeMs, _start, _stop);
    return _runtimeMs;
}