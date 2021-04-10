cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(parallelprimitives LANGUAGES CXX CUDA)

find_package(CUDA  REQUIRED)
include_directories("${CUDA_INCLUDE_DIRS}")

file(GLOB SOURCES "src/*.cu" "include/*.hpp")
set (tests test_reduction test_scan test_histogram test_gemm) 

foreach(test IN LISTS tests)

    add_executable(${test} test/${test}.cu)
    set_target_properties(${test} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    target_include_directories(${test} PRIVATE include)
    target_compile_options(${test} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
                        -arch=sm_52
                        >)
endforeach()

