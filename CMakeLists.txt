############################################################################
# Modified from Udacity's CS344 repository:
#     https://github.com/udacity/cs344
############################################################################

cmake_minimum_required(VERSION 2.6 FATAL_ERROR)
project(IntroToComputerVision)

find_package(OpenCV REQUIRED core imgproc imgcodecs highgui cudafilters cudaimgproc)
find_package(CUDA REQUIRED)

link_libraries(${OpenCV_LIBS} )

set (EXECUTABLE_OUTPUT_PATH "${CMAKE_SOURCE_DIR}/bin/")

#######################################
# Externally-sourced libraries
#######################################
add_subdirectory(external)

include_directories(external/yaml-cpp/include)
include_directories(external/spdlog/include)

#######################################
# Utility library
#######################################
add_subdirectory(common)
include_directories(common/include)

#######################################
# Look for CUDA
#######################################

if(CUDA_FOUND)
  set(CUDA_NVCC_FLAGS "
  -ccbin g++;
  -gencode;arch=compute_50,code=sm_50;
  -gencode;arch=compute_52,code=sm_52;
  -gencode;arch=compute_60,code=sm_60;
  -gencode;arch=compute_61,code=sm_61;
  -gencode;arch=compute_62,code=sm_62")

  # add -Wextra compiler flag for gcc compilations
  if (UNIX)
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -Xcompiler -Wextra")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fprofile-arcs -ftest-coverage")
  endif (UNIX)

  # Support g++ versions less than 6.0
  if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    if (CMAKE_CXX_COMPILER_VERSION VERSION_LESS 6.0)
      set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
      set (CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -std=c++11")
      set (CUDA_PROPAGATE_HOST_FLAGS OFF)
    endif()
  endif()

  # add debugging to CUDA NVCC flags.  For NVidia's NSight tools.
  set(CUDA_NVCC_FLAGS_DEBUG ${CUDA_NVCC_FLAGS_DEBUG} "-G")

  add_subdirectory (ProblemSets)

else(CUDA_FOUND)
  message("CUDA is not installed on this system.")
endif()
