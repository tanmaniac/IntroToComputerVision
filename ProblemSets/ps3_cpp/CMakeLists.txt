file(GLOB lib_hdr include/*.h include/*.hpp)
file(GLOB lib_src lib/*.cpp)

file(GLOB exe_hdr src/*.h src/*.hpp)
file(GLOB exe_src src/*.cpp)

add_library(ps3sol ${lib_hdr} ${lib_src})

add_executable(ps3 ${exe_hdr} ${exe_src})

target_link_libraries(ps3sol yaml-cpp cv_common Eigen3::Eigen)
target_link_libraries(ps3 ps3sol)