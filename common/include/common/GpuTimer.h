#pragma once

#include <cuda_runtime.h>

class GpuTimer {
private:
    float _runtimeMs;
    cudaEvent_t _start, _stop;

public:
    GpuTimer();
    ~GpuTimer();

    // Start timing GPU execution
    void start();

    // Stop timing GPU execution
    void stop();

    // Get runtime
    float getTime();
};