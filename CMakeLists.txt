cmake_minimum_required (VERSION 3.1)
project(opencl-test)

find_package(OpenCL REQUIRED)

add_executable (main main.c)
target_include_directories (main PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries (main ${OpenCL_LIBRARY})
#target_link_libraries (main "C:/Program Files/NVIDIA Corporation/OpenCL/OpenCL64.dll")


add_executable (kernel_debug kernel.c)
target_include_directories (kernel_debug PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
