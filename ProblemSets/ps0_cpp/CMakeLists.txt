file(GLOB hdr *.h *.hpp *.cuh)
file(GLOB src *.cpp *.cu)

add_executable(ps0 ${hdr} ${src})

target_link_libraries(ps0 yaml-cpp)

#set(EXECUTABLE_OUTPUT_PATH "${EXECUTABLE_OUTPUT_PATH}/ps0")