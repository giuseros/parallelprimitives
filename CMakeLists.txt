cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(parallelprimitives LANGUAGES CXX CUDA)

find_package(CUDA  REQUIRED)
include_directories("${CUDA_INCLUDE_DIRS}")

#set(CMAKE_CXX_FLAGS "-O0 -g")

file(GLOB SOURCES "src/*.cu" "include/*.hpp")
#add_library(parallelprimitives SHARED ${SOURCES})
#target_compile_features(parallelprimitives PUBLIC cxx_std_14)
#target_include_directories(parallelprimitives PRIVATE include)
#set_target_properties(parallelprimitives PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set (tests test_reduction test_scan test_histogram test_gemm) 

foreach(test IN LISTS tests)

    add_executable(${test} test/${test}.cu)
    set_target_properties(${test} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    target_include_directories(${test} PRIVATE include)
    target_compile_options(${test} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
                        -arch=sm_52
                        >)
endforeach()


#target_compile_options(parallelprimitives PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
#                       -G
#                       --generate-line-info
#                       --use_fast_math
#                       --relocatable-device-code=true
#                       >)

