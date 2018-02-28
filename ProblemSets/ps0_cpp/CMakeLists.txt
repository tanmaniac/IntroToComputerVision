file(GLOB hdr *.h *.hpp *.cuh)
file(GLOB src *.cpp *.cu)

add_executable(ps0 ${hdr} ${src})