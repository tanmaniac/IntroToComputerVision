file(GLOB hdr src/*.h src/*.hpp)
file(GLOB src src/*.cpp)
file(GLOB cu src/*.cu src/*.cuh)

cuda_add_executable(ps1 ${hdr} ${cu} ${src})

target_link_libraries(ps1 yaml-cpp cv_common)