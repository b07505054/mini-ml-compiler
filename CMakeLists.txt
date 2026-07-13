cmake_minimum_required(VERSION 3.16)
project(mini_ml_compiler LANGUAGES CXX)

option(ENABLE_CUDA "Build CUDA targets" OFF)
include(CTest)
find_package(Python3 REQUIRED COMPONENTS Interpreter)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(ENABLE_CUDA)
    enable_language(CUDA)

    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    set(CMAKE_CUDA_ARCHITECTURES 75)
endif()

include_directories(include)

file(GLOB_RECURSE SOURCES
    src/*.cpp
)
list(FILTER SOURCES EXCLUDE REGEX "src/runtime/metal_backend_stub\\.cpp$")

if(NOT APPLE)
    list(APPEND SOURCES
        src/runtime/metal_backend_stub.cpp
    )
endif()

if(APPLE)
    file(GLOB_RECURSE METAL_SOURCES
        src/*.mm
    )
    list(APPEND SOURCES ${METAL_SOURCES})
endif()

add_library(mlcompiler ${SOURCES})

if(MSVC)
    target_compile_options(mlcompiler PRIVATE /arch:AVX2)
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
    target_compile_options(mlcompiler PRIVATE -mavx2)
endif()

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

add_executable(benchmark_prefetch_matmul apps/benchmark_prefetch_matmul.cpp)
target_link_libraries(benchmark_prefetch_matmul mlcompiler)

add_executable(benchmark_threads apps/benchmark_threads.cpp)
target_link_libraries(benchmark_threads mlcompiler)

add_executable(benchmark_gpu_sim apps/benchmark_gpu_sim.cpp)
target_link_libraries(benchmark_gpu_sim mlcompiler)

add_executable(benchmark_gpu_tiled_sim apps/benchmark_gpu_tiled_sim.cpp)
target_link_libraries(benchmark_gpu_tiled_sim mlcompiler)

add_executable(benchmark_thread_pool apps/benchmark_thread_pool.cpp)
target_link_libraries(benchmark_thread_pool mlcompiler)

add_executable(run_mlir_model apps/run_mlir_model.cpp)
target_link_libraries(run_mlir_model mlcompiler)

add_executable(run_mlir_fused_kernel_benchmark apps/run_mlir_fused_kernel_benchmark.cpp)
target_link_libraries(run_mlir_fused_kernel_benchmark mlcompiler)
add_test(
    NAME mlir_fused_kernel_benchmark_smoke
    COMMAND run_mlir_fused_kernel_benchmark
        --pattern all
        --variant all
        --warmup 1
        --iterations 1
        --repeats 2
        --output ${CMAKE_BINARY_DIR}/matmul_postop_relu_kernel_profile_smoke.json
        --report-output ${CMAKE_BINARY_DIR}/matmul_postop_relu_benchmark_report_smoke.md
)
set_tests_properties(mlir_fused_kernel_benchmark_smoke PROPERTIES
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
)
add_test(
    NAME mlir_fused_kernel_benchmark_json_schema
    COMMAND
        ${Python3_EXECUTABLE}
        ${CMAKE_SOURCE_DIR}/tests/validate_matmul_postop_benchmark_json.py
        $<TARGET_FILE:run_mlir_fused_kernel_benchmark>
        ${CMAKE_BINARY_DIR}/matmul_postop_relu_kernel_profile_schema.json
        ${CMAKE_BINARY_DIR}/matmul_postop_relu_benchmark_report_schema.md
)
set_tests_properties(mlir_fused_kernel_benchmark_json_schema PROPERTIES
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
)
add_test(
    NAME matmul_postop_multi_shape_infrastructure
    COMMAND
        ${Python3_EXECUTABLE}
        ${CMAKE_SOURCE_DIR}/tests/test_matmul_postop_multi_shape.py
)
set_tests_properties(matmul_postop_multi_shape_infrastructure PROPERTIES
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
)
add_test(
    NAME triton_matmul_bias_relu_benchmark_infrastructure
    COMMAND
        ${Python3_EXECUTABLE}
        ${CMAKE_SOURCE_DIR}/tests/test_triton_matmul_bias_relu_benchmark.py
)
set_tests_properties(triton_matmul_bias_relu_benchmark_infrastructure PROPERTIES
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
)
add_test(
    NAME triton_matmul_bias_relu_exact_selection_infrastructure
    COMMAND
        ${Python3_EXECUTABLE}
        ${CMAKE_SOURCE_DIR}/tests/test_triton_exact_selection.py
)
set_tests_properties(triton_matmul_bias_relu_exact_selection_infrastructure PROPERTIES
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
)
add_test(
    NAME mlir_fused_kernel_benchmark_rejects_invalid_variant
    COMMAND run_mlir_fused_kernel_benchmark --variant not-a-variant
)
set_tests_properties(mlir_fused_kernel_benchmark_rejects_invalid_variant PROPERTIES
    WILL_FAIL TRUE
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
)
add_test(
    NAME mlir_fused_kernel_benchmark_rejects_invalid_bias_shape
    COMMAND run_mlir_fused_kernel_benchmark
        --pattern bias
        --variant naive-unfused
        --warmup 1
        --iterations 1
        --repeats 1
        --force-invalid-bias-shape
        --output ${CMAKE_BINARY_DIR}/invalid_bias_shape.json
        --report-output ${CMAKE_BINARY_DIR}/invalid_bias_shape.md
)
set_tests_properties(mlir_fused_kernel_benchmark_rejects_invalid_bias_shape PROPERTIES
    WILL_FAIL TRUE
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
)
add_test(
    NAME mlir_fused_kernel_benchmark_rejects_invalid_elementwise_shape
    COMMAND run_mlir_fused_kernel_benchmark
        --pattern elementwise-add
        --variant naive-unfused
        --warmup 1
        --iterations 1
        --repeats 1
        --force-invalid-elementwise-shape
        --output ${CMAKE_BINARY_DIR}/invalid_elementwise_shape.json
        --report-output ${CMAKE_BINARY_DIR}/invalid_elementwise_shape.md
)
set_tests_properties(mlir_fused_kernel_benchmark_rejects_invalid_elementwise_shape PROPERTIES
    WILL_FAIL TRUE
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
)

# ---------------------------------------------------------------------------
# Phase 1: CPU fused schedule candidate discovery and real oracle.
# Self-contained; does not depend on or modify run_mlir_fused_kernel_benchmark.
# ---------------------------------------------------------------------------
add_executable(run_cpu_fused_schedule_discovery apps/run_cpu_fused_schedule_discovery.cpp)
target_link_libraries(run_cpu_fused_schedule_discovery mlcompiler)

add_test(
    NAME cpu_fused_schedule_discovery_smoke
    COMMAND run_cpu_fused_schedule_discovery
        --mode discover
        --smoke
        --output-dir ${CMAKE_BINARY_DIR}/cpu_fused_schedule_discovery_smoke
)
set_tests_properties(cpu_fused_schedule_discovery_smoke PROPERTIES
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
)
add_test(
    NAME cpu_fused_schedule_discovery_json_schema
    COMMAND
        ${Python3_EXECUTABLE}
        ${CMAKE_SOURCE_DIR}/tests/validate_cpu_fused_schedule_discovery_json.py
        $<TARGET_FILE:run_cpu_fused_schedule_discovery>
        ${CMAKE_BINARY_DIR}/cpu_fused_schedule_discovery_schema_check
)
set_tests_properties(cpu_fused_schedule_discovery_json_schema PROPERTIES
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
)
add_test(
    NAME cpu_fused_schedule_discovery_rejects_unknown_candidate
    COMMAND
        ${Python3_EXECUTABLE}
        ${CMAKE_SOURCE_DIR}/tests/assert_cpu_fused_schedule_rejects_unknown_candidate.py
        $<TARGET_FILE:run_cpu_fused_schedule_discovery>
        ${CMAKE_BINARY_DIR}/cpu_fused_schedule_discovery_unknown_candidate
)
set_tests_properties(cpu_fused_schedule_discovery_rejects_unknown_candidate PROPERTIES
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
)
add_test(
    NAME cpu_fused_schedule_discovery_analysis_unit_tests
    COMMAND
        ${Python3_EXECUTABLE}
        -m unittest
        ${CMAKE_SOURCE_DIR}/tests/test_cpu_fused_schedule_discovery_analysis.py
)
set_tests_properties(cpu_fused_schedule_discovery_analysis_unit_tests PROPERTIES
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
)

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
    target_link_libraries(run_metal_vector_add "-framework Foundation" "-framework Metal")

    add_executable(run_metal_backend_demo apps/run_metal_backend_demo.cpp)
    target_link_libraries(run_metal_backend_demo mlcompiler)

    add_executable(run_metal_matmul apps/run_metal_matmul.mm)
    target_link_libraries(run_metal_matmul "-framework Foundation" "-framework Metal" objc)

    add_executable(run_metal_kernel_profile apps/run_metal_kernel_profile.mm)
    target_link_libraries(run_metal_kernel_profile "-framework Metal" "-framework Foundation")

    add_executable(run_metal_vector_add_profile apps/run_metal_vector_add_profile.mm)
    target_link_libraries(run_metal_vector_add_profile "-framework Metal" "-framework Foundation")

    add_executable(benchmark_metal_rmsnorm apps/benchmark_metal_rmsnorm.mm)
    target_link_libraries(benchmark_metal_rmsnorm "-framework Metal" "-framework Foundation")
    add_test(NAME metal_rmsnorm_correctness COMMAND benchmark_metal_rmsnorm)
    set_tests_properties(metal_rmsnorm_correctness PROPERTIES
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    )

    add_executable(run_metal_rmsnorm_plan apps/run_metal_rmsnorm_plan.mm)
    target_link_libraries(run_metal_rmsnorm_plan mlcompiler)
    add_test(NAME metal_rmsnorm_plan_dispatch COMMAND run_metal_rmsnorm_plan)
    set_tests_properties(metal_rmsnorm_plan_dispatch PROPERTIES
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        SKIP_RETURN_CODE 77
    )
endif()

if(ENABLE_CUDA)

    add_executable(
        run_cuda_vector_add_profile
        apps/run_cuda_vector_add_profile.cu
    )

    set_target_properties(
        run_cuda_vector_add_profile
        PROPERTIES
        CUDA_STANDARD 17
        CUDA_STANDARD_REQUIRED ON
    )

    add_executable(
        run_cuda_matmul_profile
        apps/run_cuda_matmul_profile.cu
    )

    set_target_properties(
        run_cuda_matmul_profile
        PROPERTIES
        CUDA_STANDARD 17
        CUDA_STANDARD_REQUIRED ON
    )

endif()

add_executable(run_cv_graph_demo apps/run_cv_graph_demo.cpp)
target_link_libraries(run_cv_graph_demo mlcompiler)

add_executable(run_llm_graph_demo apps/run_llm_graph_demo.cpp)
target_link_libraries(run_llm_graph_demo mlcompiler)

add_executable(
    run_dynamic_cv_planning_demo
    apps/run_dynamic_cv_planning_demo.cpp
)

target_link_libraries(
    run_dynamic_cv_planning_demo
    mlcompiler
)

# Frontend GraphBuilder unit tests
add_executable(test_cv_graph_builder
    tests/frontend/test_cv_graph_builder.cpp)
target_link_libraries(test_cv_graph_builder mlcompiler)
add_test(NAME cv_graph_builder COMMAND test_cv_graph_builder)

add_executable(test_llm_graph_builder
    tests/frontend/test_llm_graph_builder.cpp)
target_link_libraries(test_llm_graph_builder mlcompiler)
add_test(NAME llm_graph_builder COMMAND test_llm_graph_builder)

add_executable(test_backend_placement
    tests/frontend/test_backend_placement.cpp)
target_link_libraries(test_backend_placement mlcompiler)
add_test(NAME backend_placement COMMAND test_backend_placement)
