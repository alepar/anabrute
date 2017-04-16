cmake_minimum_required (VERSION 3.1)
project(anabrute)

find_package(OpenCL REQUIRED)

add_executable (anabrute gpu_cruncher.c hashes.c main.c permut_types.c seedphrase.c fact.c cpu_cruncher.c os.c task_buffers.c)
set_property(TARGET anabrute PROPERTY C_STANDARD 99)
target_include_directories (anabrute PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries (anabrute pthread)
target_link_libraries (anabrute ${OpenCL_LIBRARY})
#target_link_libraries (anabrute "C:/Program Files/NVIDIA Corporation/OpenCL/OpenCL64.dll")

add_executable (kernel_debug gpu_cruncher.c hashes.c opencl_main.c permut_types.c seedphrase.c fact.c)
set_property(TARGET kernel_debug PROPERTY C_STANDARD 99)
target_include_directories (kernel_debug PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries (kernel_debug pthread)
#target_link_libraries (kernel_debug ${OpenCL_LIBRARY})
target_link_libraries (kernel_debug "C:/Program Files/NVIDIA Corporation/OpenCL/OpenCL64.dll")
