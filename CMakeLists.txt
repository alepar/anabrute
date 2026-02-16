cmake_minimum_required (VERSION 3.5)
project(anabrute)

find_package(OpenCL REQUIRED)

# homebrew includes (event.h, unitypes.h on macOS)
if(APPLE AND EXISTS "/opt/homebrew/include")
    include_directories("/opt/homebrew/include")
endif()

# suppress int-to-pointer warnings from ret_iferr macro in void* functions (known bug)
add_compile_options(-Wno-int-conversion)

add_executable (anabrute main.c gpu_cruncher.c hashes.c dict.c permut_types.c seedphrase.c fact.c cpu_cruncher.c os.c task_buffers.c)
set_property(TARGET anabrute PROPERTY C_STANDARD 99)
target_include_directories (anabrute PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries (anabrute pthread)
target_link_libraries (anabrute ${OpenCL_LIBRARY})
#target_link_libraries (anabrute "C:/Program Files/NVIDIA Corporation/OpenCL/OpenCL64.dll")

add_executable (kernel_debug kernel_debug.c gpu_cruncher.c hashes.c dict.c permut_types.c seedphrase.c fact.c cpu_cruncher.c os.c task_buffers.c)
set_property(TARGET kernel_debug PROPERTY C_STANDARD 99)
target_include_directories (kernel_debug PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries (kernel_debug pthread)
target_link_libraries (kernel_debug ${OpenCL_LIBRARY})
#target_link_libraries (kernel_debug "C:/Program Files/NVIDIA Corporation/OpenCL/OpenCL64.dll")

# === Tests ===
enable_testing()

add_executable(test_hash_parsing tests/test_hash_parsing.c hashes.c)
set_property(TARGET test_hash_parsing PROPERTY C_STANDARD 99)
target_include_directories(test_hash_parsing PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_options(test_hash_parsing PRIVATE -fsanitize=address -fsanitize=undefined -fno-omit-frame-pointer)
target_link_options(test_hash_parsing PRIVATE -fsanitize=address -fsanitize=undefined)
add_test(NAME hash_parsing COMMAND test_hash_parsing)

add_executable(test_dict_parsing tests/test_dict_parsing.c dict.c permut_types.c seedphrase.c)
set_property(TARGET test_dict_parsing PROPERTY C_STANDARD 99)
target_include_directories(test_dict_parsing PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_options(test_dict_parsing PRIVATE -fsanitize=address -fsanitize=undefined -fno-omit-frame-pointer)
target_link_options(test_dict_parsing PRIVATE -fsanitize=address -fsanitize=undefined)
add_test(NAME dict_parsing COMMAND test_dict_parsing)

add_executable(test_cpu_enumeration tests/test_cpu_enumeration.c
    cpu_cruncher.c task_buffers.c dict.c permut_types.c seedphrase.c fact.c os.c hashes.c)
set_property(TARGET test_cpu_enumeration PROPERTY C_STANDARD 99)
target_include_directories(test_cpu_enumeration PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_options(test_cpu_enumeration PRIVATE -fsanitize=address -fsanitize=undefined -fno-omit-frame-pointer)
target_link_options(test_cpu_enumeration PRIVATE -fsanitize=address -fsanitize=undefined)
target_link_libraries(test_cpu_enumeration pthread)
add_test(NAME cpu_enumeration COMMAND test_cpu_enumeration)
