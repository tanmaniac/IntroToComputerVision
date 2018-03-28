# Problem Set 2: Window-based Stereo Matching

All output images are originally generated in the `bin/ps2_output` directory at the root of this repository. Additionally, program logfiles are generated in `bin`. For convenience, I have copied these outputs to the `output` directory here.

CUDA kernel execution times are listed in `output/ps2_gpu.log`.

Left-hand side disparity maps computed stereo correspondence from left to right; right-hand side computed it from right to left.

## Problem 1: Basic stereo with sum of squared differences

Input images:

<img src="../../Resources/ProblemSet2/pair0-L.png" width="300"/>    <img src="../../Resources/ProblemSet2/pair0-R.png" width="300"/>

Disparity maps with SSD:

<img src="./output/ps2-1-a-1.png" width="300"/>    <img src="./output/ps2-1-a-2.png" width="300"/>

- Runtime with CPU: 366.367 ms
- Runtime with GPU (CUDA): 7.18332 ms

**Speedup with GPU: 51x**

## Problem 2: More realistic images

Input images:

<img src="../../Resources/ProblemSet2/pair1-L.png" width="300"/>    <img src="../../Resources/ProblemSet2/pair1-R.png" width="300"/>

Ground truths:

<img src="../../Resources/ProblemSet2/pair1-D_L.png" width="300"/>    <img src="../../Resources/ProblemSet2/pair1-D_R.png" width="300"/>

Disparity maps with SSD:

<img src="./output/ps2-2-a-1-inverted.png" width="300"/>    <img src="./output/ps2-2-a-2.png" width="300"/>

- Runtime with CPU: 161714 ms
- Runtime with GPU (CUDA): 56.5523 ms

**Speedup with GPU: 2859.55x**

## Problem 3: Adding noise and contrast to the previous image

With added Gaussian noise (mean = 0, sigma = 10)

<img src="./output/ps2-3-a-1-inverted.png" width="300"/>    <img src="./output/ps2-3-a-2.png" width="300"/>

With boosted contrast (10% increase)

<img src="./output/ps2-3-b-1-inverted.png" width="300"/>    <img src="./output/ps2-3-b-2.png" width="300"/>

- Runtime with CPU: 373934 ms
- Runtime with GPU (CUDA): 118.377 ms

**Speedup with GPU: 3158.84x**

## Problem 4: Implementing stereo correspondence with normalized cross-correlation

Note: I used the built-in OpenCV `cv::matchTemplate` function to run the CPU version of the normalized cross-correlation matcher, but there is no fast parallelized implementation in OpenCV that would work for computing a full disparity map. I wrote another kernel similar to the sum of squared differences kernel in Problem 1 to handle normalized cross-correlation on the GPU.

Window matching with the images from Problem 2:

<img src="./output/ps2-4-a-1-inverted.png" width="300"/>    <img src="./output/ps2-4-a-2.png" width="300"/>

With increased Gaussian noise (mean = 0, sigma = 10)

<img src="./output/ps2-4-b-1-inverted.png" width="300"/>    <img src="./output/ps2-4-b-2.png" width="300"/>

With a 10% increase in contrast

<img src="./output/ps2-4-c-1-inverted.png" width="300"/>    <img src="./output/ps2-4-c-2.png" width="300"/>

- Runtime with CPU: 49778.3 ms
- Runtime with GPU: 215.4 ms

**Speedup with GPU: 231.10x**

## Problem 5: Normalized cross-correlation on a different pair of images

Input images:

<img src="../../Resources/ProblemSet2/pair2-L.png" width="300"/>    <img src="../../Resources/ProblemSet2/pair2-R.png" width="300"/>

Ground truths:

<img src="../../Resources/ProblemSet2/pair2-D_L.png" width="300"/>    <img src="../../Resources/ProblemSet2/pair2-D_R.png" width="300"/>

Output using normalized cross correlation:

<img src="./output/ps2-5-a-1-inverted.png" width="300"/>    <img src="./output/ps2-5-a-2.png" width="300"/>

- Runtime with CPU: 16845.6 ms
- Runtime with GPU: 61.6843 ms

**Speedup with GPU: 273.09x**