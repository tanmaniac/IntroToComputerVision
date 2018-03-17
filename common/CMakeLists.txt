file(GLOB cuda_hdr include/common/*.cuh)
file(GLOB cpp_hdr include/common/*.h include/common/*.hpp)
file(GLOB cuda_src src/*.cu)
file(GLOB cpp_src src/*.cpp)

include_directories("${CUDA_INCLUDE_DIRS}")

add_library(cv_common ${cpp_hdr} ${cuda_hdr} ${cpp_src} ${cuda_src})