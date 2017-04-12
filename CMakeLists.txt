cmake_minimum_required (VERSION 3.1)
project(anabrute)

find_package(OpenCL REQUIRED)

add_executable (main main.c ocl_layer.c hashes.c)
target_include_directories (main PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries (main pthread)
#target_link_libraries (main ${OpenCL_LIBRARY})
target_link_libraries (main "C:/Program Files/NVIDIA Corporation/OpenCL/OpenCL64.dll")
