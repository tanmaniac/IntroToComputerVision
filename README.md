# Intro to Computer Vision (Udacity UD810)

Lecture slides, problem sets, and C++ solutions to UD810.

- [Original Google Doc](https://docs.google.com/spreadsheets/d/1ecUGIyhYOfQPi3HPXb-7NndrLgpX_zgkwsqzfqHPaus/pubhtml) with problem sets and slides
- [Miscellaneous images](http://sipi.usc.edu/database/database.php?volume=misc) from USC
- [Computer vision image database](http://homepages.inf.ed.ac.uk/rbf/CVonline/Imagedbase.htm) from the University of Edinburgh

## Prerequisites

I try to accelerate all of the assignments with CUDA, so you will need an Nvidia GPU along with CUDA toolkit installed. This repository is developed on:
- Ubuntu 17.10 + CUDA 9.1; Intel Core i7 6800k + GeForce GTX 1080

Follow [Nvidia's CUDA installation instructions](http://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html) to install CUDA. 

[Git LFS](https://git-lfs.github.com/) is also required to clone the images and lecture slides.

From here, there are two options: building in **Docker** or just building locally on a host machine.

### Containerized Build in Docker
This is the recommended approach to building this project since all of the dependencies are included in the Docker image.

- Install Docker-CE as described for your distribution on the [Docker docs](https://docs.docker.com/install/).
    - Follow the [Optional Linux post-installation](https://docs.docker.com/install/linux/linux-postinstall/) steps to run Docker without `sudo`.
- Install nvidia-docker2 as described in the [`nvidia-docker` docs](https://github.com/nvidia/nvidia-docker/wiki/Installation-(version-2.0)). The container provided in this repo needs the `nvidia` Docker runtime to run.
- Clone this repo and build/run the Docker container:

```bash
git clone --recursive https://github.com/tanmaniac/IntroToComputerVision.git
cd IntroToComputerVision/Docker
# Build the Docker container
./build.sh
# Run the container
./run.sh
```

This will drop you into a shell where you can follow the build steps below. The `IntroToComputerVision` directory (this one) is mapped to `${HOME}/IntroToComputerVision` in the Docker container.

### Build on host

- Build OpenCV 3.4.1 as directed in the OpenCV documentation. Make sure to add the `-DWITH_CUDA=ON` CMake flag to compile CUDA features.
- Install [Eigen](http://eigen.tuxfamily.org/) as described in its documentation.

## Building

```bash
# Navigate to wherever you cloned this repo first
mkdir build && cd build
cmake ..
make -j
```

If you want to use NVIDIA debugging tools, like `cuda-memcheck` or the NVIDIA Visual Profiler, compile with Debug flags:
```bash
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Debug ..
make -j
```
Note that this increases runtime of some kernels by around 100x, so only compile in debug mode if you need it.


## Running

Build outputs are placed in the `bin` directory. All of the executables are configured with YAML files in the `config` directory. You can edit these to change the input parameters to each of the assignments. For example, with Problem Set 0, all you need to do (after building) is:

```bash
cd bin
./ps0
```
