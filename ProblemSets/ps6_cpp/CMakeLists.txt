file(GLOB lib_hdr include/*.h include/*.hpp)
file(GLOB lib_src lib/*.cpp)
file(GLOB lib_cu lib/*.cu lib/*.cuh)

file(GLOB exe_hdr src/*.h src/*.hpp)
file(GLOB exe_src src/*.cpp)

add_library(ps6sol ${lib_hdr} ${lib_cu} ${lib_src})

add_executable(ps6 ${exe_hdr} ${exe_src})

target_link_libraries(ps6sol yaml-cpp cv_common)
target_link_libraries(ps6 ps6sol)