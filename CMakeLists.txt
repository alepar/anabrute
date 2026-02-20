cmake_minimum_required (VERSION 3.5)
project(anabrute)

if(APPLE)
    enable_language(OBJC)
    set_source_files_properties(metal_cruncher.m PROPERTIES COMPILE_FLAGS "-fobjc-arc")
endif()

find_package(OpenCL QUIET)

if(OpenCL_FOUND)
    message(STATUS "OpenCL found: ${OpenCL_LIBRARY}")
    add_compile_definitions(HAVE_OPENCL)
else()
    message(STATUS "OpenCL NOT found — OpenCL backend disabled, skipping kernel_debug target")
endif()

# homebrew includes (event.h, unitypes.h on macOS)
if(APPLE AND EXISTS "/opt/homebrew/include")
    include_directories("/opt/homebrew/include")
endif()

# suppress int-to-pointer warnings from ret_iferr macro in void* functions (known bug)
add_compile_options(-Wno-int-conversion)

# Enable AVX2 for avx_cruncher.c on x86_64
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
    set_source_files_properties(avx_cruncher.c PROPERTIES COMPILE_FLAGS "-mavx2 -mavx512f")
endif()

# === Main binary (works with or without OpenCL) ===
add_executable (anabrute main.c opencl_cruncher.c gpu_cruncher.c avx_cruncher.c hashes.c dict.c permut_types.c seedphrase.c fact.c cpu_cruncher.c os.c task_buffers.c)
set_property(TARGET anabrute PROPERTY C_STANDARD 99)
target_include_directories (anabrute PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries (anabrute pthread m)
if(OpenCL_FOUND)
    target_link_libraries (anabrute ${OpenCL_LIBRARY})
endif()
if(APPLE)
    target_sources(anabrute PRIVATE metal_cruncher.m)
    target_link_libraries(anabrute "-framework Metal" "-framework Foundation")
endif()

# === kernel_debug (requires OpenCL) ===
if(OpenCL_FOUND)
    add_executable (kernel_debug kernel_debug.c opencl_cruncher.c gpu_cruncher.c avx_cruncher.c hashes.c dict.c permut_types.c seedphrase.c fact.c cpu_cruncher.c os.c task_buffers.c)
    set_property(TARGET kernel_debug PROPERTY C_STANDARD 99)
    target_include_directories (kernel_debug PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
    target_link_libraries (kernel_debug pthread)
    target_link_libraries (kernel_debug ${OpenCL_LIBRARY})
    if(APPLE)
        target_sources(kernel_debug PRIVATE metal_cruncher.m)
        target_link_libraries(kernel_debug "-framework Metal" "-framework Foundation")
    endif()
endif()

# === Benchmark ===
# bench_enum is portable (no intrinsics)
add_executable(bench_enum bench_enum.c cpu_cruncher.c task_buffers.c dict.c permut_types.c seedphrase.c fact.c os.c hashes.c)
set_property(TARGET bench_enum PROPERTY C_STANDARD 99)
target_include_directories(bench_enum PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(bench_enum pthread)
target_compile_options(bench_enum PRIVATE -O2)

# bench_avx and bench_breakdown use AVX2/AVX512 intrinsics — x86_64 only
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
    add_executable(bench_avx bench_avx.c avx_cruncher.c task_buffers.c hashes.c permut_types.c seedphrase.c fact.c os.c)
    set_property(TARGET bench_avx PROPERTY C_STANDARD 99)
    target_include_directories(bench_avx PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
    target_link_libraries(bench_avx pthread)
    target_compile_options(bench_avx PRIVATE -O2)
    set_source_files_properties(bench_avx.c PROPERTIES COMPILE_FLAGS "-mavx2")

    add_executable(bench_breakdown bench_breakdown.c os.c)
    set_property(TARGET bench_breakdown PROPERTY C_STANDARD 99)
    target_include_directories(bench_breakdown PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
    target_compile_options(bench_breakdown PRIVATE -O2 -mavx2)
endif()

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

add_executable(test_cruncher tests/test_cruncher.c
    opencl_cruncher.c gpu_cruncher.c avx_cruncher.c task_buffers.c hashes.c permut_types.c seedphrase.c fact.c os.c)
set_property(TARGET test_cruncher PROPERTY C_STANDARD 99)
target_include_directories(test_cruncher PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
if(APPLE)
    # ASAN causes massive memory overhead with Metal runtime; disable for this target on macOS
    target_compile_options(test_cruncher PRIVATE -fsanitize=undefined -fno-omit-frame-pointer)
    target_link_options(test_cruncher PRIVATE -fsanitize=undefined)
    target_sources(test_cruncher PRIVATE metal_cruncher.m)
    target_link_libraries(test_cruncher "-framework Metal" "-framework Foundation")
else()
    target_compile_options(test_cruncher PRIVATE -fsanitize=address -fsanitize=undefined -fno-omit-frame-pointer)
    target_link_options(test_cruncher PRIVATE -fsanitize=address -fsanitize=undefined)
endif()
target_link_libraries(test_cruncher pthread)
if(OpenCL_FOUND)
    target_link_libraries(test_cruncher ${OpenCL_LIBRARY})
endif()
add_test(NAME cruncher COMMAND test_cruncher)
set_tests_properties(cruncher PROPERTIES WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
