cmake_minimum_required (VERSION 3.1)
project(anabrute)

find_package(OpenCL REQUIRED)

add_executable (anabrute #[[opencl_main.c ocl_layer.c hashes.c]] anabrute.c anatypes.c seedphrase.c fact.c)
target_include_directories (anabrute PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries (anabrute pthread)
#target_link_libraries (main ${OpenCL_LIBRARY})
target_link_libraries (anabrute "C:/Program Files/NVIDIA Corporation/OpenCL/OpenCL64.dll")
set_property(TARGET anabrute PROPERTY C_STANDARD 99)
