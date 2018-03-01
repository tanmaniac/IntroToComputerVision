# Intro to Computer Vision (Udacity UD810)

Lecture slides, problem sets, and C++ solutions to UD810.

- [Original Google Doc](https://docs.google.com/spreadsheets/d/1ecUGIyhYOfQPi3HPXb-7NndrLgpX_zgkwsqzfqHPaus/pubhtml) with problem sets and slides
- [Miscellaneous images](http://sipi.usc.edu/database/database.php?volume=misc) from USC
- [Computer vision image database](http://homepages.inf.ed.ac.uk/rbf/CVonline/Imagedbase.htm) from the University of Edinburgh

## Prerequisites

I try to accelerate all of the assignments with CUDA, so you will need an Nvidia GPU along with CUDA toolkit installed. This repository is developed on:
- Ubuntu 17.10 + CUDA 9.1; Intel Core i7 6800k + GeForce GTX 1080

Follow [Nvidia's CUDA installation instructions](http://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html) to install CUDA. 

You should be able to pull the rest of the necessary dependencies with
```bash
sudo apt-get update && sudo apt-get install -y libopencv-dev python-opencv build-essential cmake clang-format
```

## Building

```bash
mkdir build && cd build
cmake ..
make -j
```

## Running

Build outputs are placed in the `bin` directory. All of the executables are configured with YAML files in the `config` directory. You can edit these to change the input parameters to each of the assignments. For example, with Problem Set 0, all you need to do (after building) is:

```bash
cd bin
./ps0
```
