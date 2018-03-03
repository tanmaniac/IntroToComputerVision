file(GLOB hdr *.h *.hpp)
file(GLOB src *.cpp)
file(GLOB cu *.cu *.cuh)

cuda_add_executable(canny ${hdr} ${cu} ${src})

target_link_libraries(canny yaml-cpp)

set(EXECUTABLE_OUTPUT_PATH "${EXECUTABLE_OUTPUT_PATH}/explore/")