cmake_minimum_required(VERSION 3.16)
project(mini_ml_compiler)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(include)

file(GLOB_RECURSE SOURCES src/*.cpp)

add_library(mlcompiler ${SOURCES})

add_executable(run_demo apps/run_demo.cpp)
target_link_libraries(run_demo mlcompiler)

add_executable(benchmark_matmul apps/benchmark_matmul.cpp)
target_link_libraries(benchmark_matmul mlcompiler)

add_executable(benchmark_kernels apps/benchmark_kernels.cpp)
target_link_libraries(benchmark_kernels mlcompiler)

add_executable(benchmark_threads apps/benchmark_threads.cpp)
target_link_libraries(benchmark_threads mlcompiler)