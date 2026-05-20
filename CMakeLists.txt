cmake_minimum_required(VERSION 3.16)
project(mini_ml_compiler)

if (MSVC)
    add_compile_options(/arch:AVX2)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(include)

file(GLOB_RECURSE SOURCES
    src/*.cpp
    src/*.mm
)

add_library(mlcompiler ${SOURCES})
if(APPLE)
    target_link_libraries(mlcompiler
        PUBLIC
        "-framework Foundation"
        "-framework Metal"
        objc
    )
endif()

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

add_executable(benchmark_avx2_matmul apps/benchmark_avx2_matmul.cpp)
target_link_libraries(benchmark_avx2_matmul mlcompiler)

add_executable(benchmark_int8_matmul apps/benchmark_int8_matmul.cpp)
target_link_libraries(benchmark_int8_matmul mlcompiler)

add_executable(run_dynamic_shape_demo apps/run_dynamic_shape_demo.cpp)
target_link_libraries(run_dynamic_shape_demo mlcompiler)

add_executable(benchmark_neon_add apps/benchmark_neon_add.cpp)
target_link_libraries(benchmark_neon_add mlcompiler)

add_executable(run_dce_demo apps/run_dce_demo.cpp)
target_link_libraries(run_dce_demo mlcompiler)

add_executable(run_algebraic_simplification_demo apps/run_algebraic_simplification_demo.cpp)
target_link_libraries(run_algebraic_simplification_demo mlcompiler)

add_executable(run_layernorm_demo apps/run_layernorm_demo.cpp)
target_link_libraries(run_layernorm_demo mlcompiler)

add_executable(run_fused_attention_demo apps/run_fused_attention_demo.cpp)
target_link_libraries(run_fused_attention_demo mlcompiler)

add_executable(run_async_demo apps/run_async_demo.cpp)
target_link_libraries(run_async_demo mlcompiler)

add_executable(run_execution_provider_demo apps/run_execution_provider_demo.cpp)
target_link_libraries(run_execution_provider_demo mlcompiler)

add_executable(run_parallel_demo apps/run_parallel_demo.cpp)
target_link_libraries(run_parallel_demo mlcompiler)

add_executable(run_mlir_emitter_demo apps/run_mlir_emitter_demo.cpp)
target_link_libraries(run_mlir_emitter_demo mlcompiler)

add_executable(run_subgraph_partition_demo apps/run_subgraph_partition_demo.cpp)
target_link_libraries(run_subgraph_partition_demo mlcompiler)

add_executable(run_allocator_v2_demo apps/run_allocator_v2_demo.cpp)
target_link_libraries(run_allocator_v2_demo mlcompiler)

add_executable(run_tiled_attention_demo apps/run_tiled_attention_demo.cpp)
target_link_libraries(run_tiled_attention_demo mlcompiler)

add_executable(run_llm_scheduler_demo apps/run_llm_scheduler_demo.cpp)
target_link_libraries(run_llm_scheduler_demo mlcompiler)

add_executable(run_paged_kv_demo apps/run_paged_kv_demo.cpp)
target_link_libraries(run_paged_kv_demo mlcompiler)

add_executable(run_compiler_pipeline_demo apps/run_compiler_pipeline_demo.cpp)
target_link_libraries(run_compiler_pipeline_demo mlcompiler)

add_executable(run_runtime_trace_demo apps/run_runtime_trace_demo.cpp)
target_link_libraries(run_runtime_trace_demo mlcompiler)

add_executable(run_fusion_bias_demo apps/run_fusion_bias_demo.cpp)
target_link_libraries(run_fusion_bias_demo mlcompiler)

if(APPLE)
    add_executable(run_metal_vector_add apps/run_metal_vector_add.mm)
    find_library(FOUNDATION_FRAMEWORK Foundation)
    find_library(METAL_FRAMEWORK Metal)
    target_link_libraries(run_metal_vector_add "-framework Foundation" "-framework Metal")
endif()

if(APPLE)
    add_executable(run_metal_backend_demo
        apps/run_metal_backend_demo.cpp
    )

    target_link_libraries(
        run_metal_backend_demo
        mlcompiler
    )
endif()

if(APPLE)
    add_executable(run_metal_matmul apps/run_metal_matmul.mm)

    target_link_libraries(
        run_metal_matmul
        "-framework Foundation"
        "-framework Metal"
        objc
    )
endif()

add_executable(
    run_metal_kernel_profile
    apps/run_metal_kernel_profile.mm
)

target_link_libraries(
    run_metal_kernel_profile
    "-framework Metal"
    "-framework Foundation"
)

add_executable(
    run_metal_vector_add_profile
    apps/run_metal_vector_add_profile.mm
)

target_link_libraries(
    run_metal_vector_add_profile
    "-framework Metal"
    "-framework Foundation"
)