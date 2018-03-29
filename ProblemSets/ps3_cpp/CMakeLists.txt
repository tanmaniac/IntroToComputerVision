file(GLOB lib_hdr include/*.h include/*.hpp)
file(GLOB lib_src lib/*.cpp)
file(GLOB lib_cu lib/*.cu lib/*.cuh)

file(GLOB exe_hdr src/*.h src/*.hpp)
file(GLOB exe_src src/*.cpp)

cuda_add_library(ps3sol ${lib_hdr} ${lib_cu} ${lib_src})

cuda_add_executable(ps3 ${exe_hdr} ${exe_src})

target_link_libraries(ps3sol yaml-cpp cv_common)
target_link_libraries(ps3 ps3sol)