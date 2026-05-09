cmake_minimum_required(VERSION 3.16)
project(mini_ml_compiler)

if (MSVC)
    add_compile_options(/arch:AVX2)
endif()

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

add_executable(benchmark_gpu_sim apps/benchmark_gpu_sim.cpp)
target_link_libraries(benchmark_gpu_sim mlcompiler)

add_executable(benchmark_gpu_tiled_sim apps/benchmark_gpu_tiled_sim.cpp)
target_link_libraries(benchmark_gpu_tiled_sim mlcompiler)

target_sources(mlcompiler PRIVATE src/runtime/thread_pool.cpp)

add_executable(benchmark_thread_pool apps/benchmark_thread_pool.cpp)
target_link_libraries(benchmark_thread_pool mlcompiler)

add_executable(run_mlir_model apps/run_mlir_model.cpp)
target_link_libraries(run_mlir_model mlcompiler)

add_executable(run_attention_demo apps/run_attention_demo.cpp)
target_link_libraries(run_attention_demo mlcompiler)

add_executable(run_causal_attention_demo apps/run_causal_attention_demo.cpp)
target_link_libraries(run_causal_attention_demo mlcompiler)

add_executable(run_kv_cache_demo apps/run_kv_cache_demo.cpp)
target_link_libraries(run_kv_cache_demo mlcompiler)

add_executable(benchmark_simd_relu apps/benchmark_simd_relu.cpp)
target_link_libraries(benchmark_simd_relu mlcompiler)

add_executable(benchmark_simd_add apps/benchmark_simd_add.cpp)
target_link_libraries(benchmark_simd_add mlcompiler)

add_executable(run_backend_demo apps/run_backend_demo.cpp)
target_link_libraries(run_backend_demo mlcompiler)

add_executable(run_scheduled_backend_demo apps/run_scheduled_backend_demo.cpp)
target_link_libraries(run_scheduled_backend_demo mlcompiler)

add_executable(run_partition_demo apps/run_partition_demo.cpp)
target_link_libraries(run_partition_demo mlcompiler)