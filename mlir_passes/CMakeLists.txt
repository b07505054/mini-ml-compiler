cmake_minimum_required(VERSION 3.20)
project(hir_mlir_passes LANGUAGES CXX C)

find_package(LLVM REQUIRED CONFIG)
find_package(MLIR REQUIRED CONFIG)

message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")
message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")

include(AddLLVM)
include(AddMLIR)
include(TableGen)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/include)

add_definitions(${LLVM_DEFINITIONS})

file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/include/HIR/IR)

set(LLVM_TARGET_DEFINITIONS include/HIR/IR/HIROps.td)
mlir_tablegen(include/HIR/IR/HIROps.h.inc -gen-op-decls)
mlir_tablegen(include/HIR/IR/HIROps.cpp.inc -gen-op-defs)
mlir_tablegen(include/HIR/IR/HIROpsDialect.h.inc -gen-dialect-decls -dialect=hir)
mlir_tablegen(include/HIR/IR/HIROpsDialect.cpp.inc -gen-dialect-defs -dialect=hir)
add_public_tablegen_target(HIRDialectIncGen)

set(LLVM_TARGET_DEFINITIONS include/FusionPasses.td)
mlir_tablegen(include/FusionPasses.h.inc -gen-pass-decls)
add_public_tablegen_target(HIRFusionPassIncGen)

file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/include)

add_library(HIRMatMulBiasReluFusionPass MODULE
  lib/HIR/IR/HIRDialect.cpp
  lib/HIR/IR/HIROps.cpp
  lib/MatMulBiasReluFusionPass.cpp
  lib/QuantizationPlanningPass.cpp
  lib/ServingPhaseAnalysisPass.cpp
  lib/serving/KVLayoutPlanningPass.cpp
  lib/serving/ReplayEligibilityPass.cpp
  lib/serving/ServingExecutionPlanBuilder.cpp
  lib/serving/ServingPipeline.cpp
  lib/serving/TargetConstraints.cpp
  lib/serving/ExecutionProviderPlanningPass.cpp
  lib/serving/LLMFrontendNormalizationPass.cpp
  lib/serving/CVFrontendNormalizationPass.cpp
  lib/serving/CVShapeInferencePass.cpp
  lib/serving/CVMemoryPlanningPass.cpp
  lib/serving/CVExecutionDomainPlanningPass.cpp
  lib/serving/CVExecutionPlanBuilder.cpp
)
llvm_update_compile_flags(HIRMatMulBiasReluFusionPass)

add_dependencies(HIRMatMulBiasReluFusionPass
  HIRDialectIncGen
  HIRFusionPassIncGen
)

if(APPLE)
  target_link_options(HIRMatMulBiasReluFusionPass
    PRIVATE
    "-undefined"
    "dynamic_lookup"
  )
endif()

target_include_directories(HIRMatMulBiasReluFusionPass
  PRIVATE
  ${LLVM_INCLUDE_DIRS}
  ${MLIR_INCLUDE_DIRS}
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${CMAKE_CURRENT_BINARY_DIR}/include
)

target_compile_definitions(HIRMatMulBiasReluFusionPass
  PRIVATE
  ${LLVM_DEFINITIONS}
)

if(APPLE)
  set_target_properties(HIRMatMulBiasReluFusionPass PROPERTIES
    PREFIX ""
    SUFFIX ".dylib"
  )
endif()

# ---------------------------------------------------------------------------
# CTest: ServingExecutionPlanBuilder unit test
# Compiles ServingPhaseAnalysisPass and ServingExecutionPlanBuilder directly
# (no plugin loading, no Python, no JSON).
# ---------------------------------------------------------------------------
enable_testing()

add_executable(ServingExecutionPlanBuilderTest
  lib/ServingPhaseAnalysisPass.cpp
  lib/serving/KVLayoutPlanningPass.cpp
  lib/serving/ReplayEligibilityPass.cpp
  lib/serving/ExecutionProviderPlanningPass.cpp
  lib/serving/ServingExecutionPlanBuilder.cpp
  test/serving/ServingExecutionPlanBuilderTest.cpp
)
llvm_update_compile_flags(ServingExecutionPlanBuilderTest)
add_dependencies(ServingExecutionPlanBuilderTest
  HIRFusionPassIncGen
)
target_include_directories(ServingExecutionPlanBuilderTest
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_BINARY_DIR}/include
    ${LLVM_INCLUDE_DIRS}
    ${MLIR_INCLUDE_DIRS}
)
target_compile_definitions(ServingExecutionPlanBuilderTest
  PRIVATE ${LLVM_DEFINITIONS}
)
target_link_libraries(ServingExecutionPlanBuilderTest
  PRIVATE
    MLIRIR
    MLIRParser
    MLIRPass
    MLIRFuncDialect
    MLIRTransforms
    MLIRSupport
)
add_test(NAME ServingExecutionPlanBuilderTest
         COMMAND ServingExecutionPlanBuilderTest)

# ---------------------------------------------------------------------------
# CTest: TargetConstraints unit test
# Exercises TargetConstraints::fromModule and ::attachToModule directly.
# No passes, no plugin loading, no Python, no JSON.
# ---------------------------------------------------------------------------
add_executable(TargetConstraintsTest
  lib/serving/TargetConstraints.cpp
  test/serving/TargetConstraintsTest.cpp
)
llvm_update_compile_flags(TargetConstraintsTest)
target_include_directories(TargetConstraintsTest
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_BINARY_DIR}/include
    ${LLVM_INCLUDE_DIRS}
    ${MLIR_INCLUDE_DIRS}
)
target_compile_definitions(TargetConstraintsTest
  PRIVATE ${LLVM_DEFINITIONS}
)
target_link_libraries(TargetConstraintsTest
  PRIVATE
    MLIRIR
    MLIRParser
    MLIRSupport
)
add_test(NAME TargetConstraintsTest
         COMMAND TargetConstraintsTest)

# ---------------------------------------------------------------------------
# CTest: CVExecutionPlanBuilder unit test
# Runs the four CV planning passes, then builds an in-memory typed plan.
# No exporter, JSON, artifact generation, or pass registration.
# ---------------------------------------------------------------------------
add_executable(CVExecutionPlanBuilderTest
  lib/serving/CVFrontendNormalizationPass.cpp
  lib/serving/CVShapeInferencePass.cpp
  lib/serving/CVMemoryPlanningPass.cpp
  lib/serving/CVExecutionDomainPlanningPass.cpp
  lib/serving/CVExecutionPlanBuilder.cpp
  test/serving/CVExecutionPlanBuilderTest.cpp
)
llvm_update_compile_flags(CVExecutionPlanBuilderTest)
add_dependencies(CVExecutionPlanBuilderTest
  HIRFusionPassIncGen
)
target_include_directories(CVExecutionPlanBuilderTest
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_BINARY_DIR}/include
    ${LLVM_INCLUDE_DIRS}
    ${MLIR_INCLUDE_DIRS}
)
target_compile_definitions(CVExecutionPlanBuilderTest
  PRIVATE ${LLVM_DEFINITIONS}
)
target_link_libraries(CVExecutionPlanBuilderTest
  PRIVATE
    MLIRIR
    MLIRParser
    MLIRPass
    MLIRFuncDialect
    MLIRTransforms
    MLIRSupport
)
add_test(NAME CVExecutionPlanBuilderTest
         COMMAND CVExecutionPlanBuilderTest)

# ---------------------------------------------------------------------------
# compile-for-target: driver tool
#
# Wires: TargetDeviceProfile JSON → TargetConstraints → serving passes →
#        ServingExecutionPlanBuilder → ServingExecutionPlanExporter →
#        canonical artifact + summary artifact.
#
# Follows the same direct-source pattern as ServingExecutionPlanBuilderTest
# (no HIRServingCore static library to avoid MODULE duplicate-symbol issues).
# JSON construction is entirely owned by ServingExecutionPlanExporter; main.cpp
# never constructs llvm::json types directly.
# ---------------------------------------------------------------------------
set(REPO_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/..")

add_executable(compile-for-target
  lib/ServingPhaseAnalysisPass.cpp
  lib/serving/KVLayoutPlanningPass.cpp
  lib/serving/ReplayEligibilityPass.cpp
  lib/serving/ExecutionProviderPlanningPass.cpp
  lib/serving/ServingExecutionPlanBuilder.cpp
  lib/serving/ServingExecutionPlanExporter.cpp
  lib/serving/TargetConstraints.cpp
  tools/compile-for-target/main.cpp
)
llvm_update_compile_flags(compile-for-target)
add_dependencies(compile-for-target HIRFusionPassIncGen)
target_include_directories(compile-for-target
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_BINARY_DIR}/include
    ${LLVM_INCLUDE_DIRS}
    ${MLIR_INCLUDE_DIRS}
)
target_compile_definitions(compile-for-target PRIVATE ${LLVM_DEFINITIONS})
target_link_libraries(compile-for-target
  PRIVATE
    MLIRIR
    MLIRParser
    MLIRPass
    MLIRFuncDialect
    MLIRTransforms
    MLIRSupport
    LLVMSupport
)

# ---------------------------------------------------------------------------
# CTest: CompileForTargetTest
# Runs compile-for-target on the A17 Pro profile + tiny_gpt_serving.mlir,
# then greps both output artifacts for required fields.
# Writes output to build dir; does not modify the repo artifact.
# ---------------------------------------------------------------------------
set(_CFT_CANONICAL "${CMAKE_CURRENT_BINARY_DIR}/test_serving_plan_iphone.json")
set(_CFT_SUMMARY   "${CMAKE_CURRENT_BINARY_DIR}/serving_execution_plan_summary.json")

add_test(NAME CompileForTargetTest
  COMMAND ${CMAKE_COMMAND}
    -DTOOL=$<TARGET_FILE:compile-for-target>
    -DPROFILE=${REPO_ROOT}/configs/target_profiles/apple_a17pro_mobile.json
    -DMLIR=${REPO_ROOT}/mlir/tiny_gpt_serving.mlir
    -DOUT=${_CFT_CANONICAL}
    -DSUMMARY=${_CFT_SUMMARY}
    -P ${CMAKE_CURRENT_SOURCE_DIR}/test/serving/RunCompileForTargetTest.cmake
)
set_tests_properties(CompileForTargetTest PROPERTIES
  DEPENDS compile-for-target
)

# ---------------------------------------------------------------------------
# qwen-to-serving-mlir: Qwen model-spec frontend
#
# Reads configs/models/qwen_0_5b_spec.json and emits serving-aware MLIR text.
# Does not import weights or lower operators.
# Truth boundary: declared model config -> serving-aware IR.
# ---------------------------------------------------------------------------
add_executable(qwen-to-serving-mlir
  tools/qwen-to-serving-mlir/main.cpp
)
llvm_update_compile_flags(qwen-to-serving-mlir)
target_include_directories(qwen-to-serving-mlir
  PRIVATE
    ${LLVM_INCLUDE_DIRS}
    ${MLIR_INCLUDE_DIRS}
)
target_compile_definitions(qwen-to-serving-mlir PRIVATE ${LLVM_DEFINITIONS})
target_link_libraries(qwen-to-serving-mlir
  PRIVATE
    LLVMSupport
)

# ---------------------------------------------------------------------------
# CTest: QwenToServingMlirTest
# Runs qwen-to-serving-mlir and checks output MLIR for required attrs/ops.
# ---------------------------------------------------------------------------
set(_QWEN_MLIR_UNIT "${CMAKE_CURRENT_BINARY_DIR}/qwen_0_5b_unit.mlir")

add_test(NAME QwenToServingMlirTest
  COMMAND ${CMAKE_COMMAND}
    -DTOOL=$<TARGET_FILE:qwen-to-serving-mlir>
    -DSPEC=${REPO_ROOT}/configs/models/qwen_0_5b_spec.json
    -DOUT=${_QWEN_MLIR_UNIT}
    -P ${CMAKE_CURRENT_SOURCE_DIR}/test/serving/RunQwenToServingMlirTest.cmake
)
set_tests_properties(QwenToServingMlirTest PROPERTIES
  DEPENDS qwen-to-serving-mlir
)

# ---------------------------------------------------------------------------
# CTest: QwenCompileForTargetTest
# End-to-end: qwen-to-serving-mlir -> compile-for-target -> JSON artifact check.
# Writes to a separate subdirectory to avoid summary file collision with
# CompileForTargetTest (which also writes serving_execution_plan_summary.json).
# ---------------------------------------------------------------------------
set(_QWEN_INT_DIR  "${CMAKE_CURRENT_BINARY_DIR}/qwen_integration")
set(_QWEN_INT_MLIR "${_QWEN_INT_DIR}/qwen_0_5b.mlir")
set(_QWEN_INT_JSON "${_QWEN_INT_DIR}/qwen_serving_plan.json")

add_test(NAME QwenCompileForTargetTest
  COMMAND ${CMAKE_COMMAND}
    -DQWEN_TOOL=$<TARGET_FILE:qwen-to-serving-mlir>
    -DCFT_TOOL=$<TARGET_FILE:compile-for-target>
    -DSPEC=${REPO_ROOT}/configs/models/qwen_0_5b_spec.json
    -DPROFILE=${REPO_ROOT}/configs/target_profiles/apple_a17pro_mobile.json
    -DMLIR_OUT=${_QWEN_INT_MLIR}
    -DJSON_OUT=${_QWEN_INT_JSON}
    -P ${CMAKE_CURRENT_SOURCE_DIR}/test/serving/RunQwenCompileForTargetTest.cmake
)
set_tests_properties(QwenCompileForTargetTest PROPERTIES
  DEPENDS "qwen-to-serving-mlir;compile-for-target"
)

# ---------------------------------------------------------------------------
# CTest: LLMFrontendNormalizationTest
# Two-step integration: mlir-opt (normalization pass) -> compile-for-target.
# Requires mlir-opt from the LLVM build. Skipped if mlir-opt is not found.
# ---------------------------------------------------------------------------
find_program(MLIR_OPT_BIN
  NAMES mlir-opt
  HINTS "${LLVM_TOOLS_BINARY_DIR}"
  DOC "Path to mlir-opt (required for LLMFrontendNormalizationTest)"
)

set(_LLMFN_INT_DIR  "${CMAKE_CURRENT_BINARY_DIR}/llm_frontend_norm_integration")
set(_LLMFN_NORM_MLIR "${_LLMFN_INT_DIR}/normalized.mlir")
set(_LLMFN_JSON      "${_LLMFN_INT_DIR}/raw_qwen_frontend_serving_plan.json")

if(MLIR_OPT_BIN)
  add_test(NAME LLMFrontendNormalizationTest
    COMMAND ${CMAKE_COMMAND}
      -DMLIR_OPT=${MLIR_OPT_BIN}
      -DPLUGIN=$<TARGET_FILE:HIRMatMulBiasReluFusionPass>
      -DCFT_TOOL=$<TARGET_FILE:compile-for-target>
      -DRAW_MLIR=${REPO_ROOT}/mlir/raw_qwen_frontend_input.mlir
      -DPROFILE=${REPO_ROOT}/configs/target_profiles/apple_a17pro_mobile.json
      -DNORM_OUT=${_LLMFN_NORM_MLIR}
      -DJSON_OUT=${_LLMFN_JSON}
      -P ${CMAKE_CURRENT_SOURCE_DIR}/test/serving/RunLLMFrontendNormalizationTest.cmake
  )
  set_tests_properties(LLMFrontendNormalizationTest PROPERTIES
    DEPENDS "HIRMatMulBiasReluFusionPass;compile-for-target"
  )
else()
  message(STATUS
    "mlir-opt not found; LLMFrontendNormalizationTest will not be added. "
    "Set MLIR_OPT_BIN or ensure mlir-opt is in PATH."
  )
endif()

# ---------------------------------------------------------------------------
# CTest: CVFrontendNormalizationTest
# Runs mlir-opt with cv-frontend-normalization on cv_raw_yoloseg.mlir, then
# checks the normalized output for expected func.func attrs.
# No compile-for-target step (no CV plan builder yet).
# Skipped if mlir-opt is not found.
# ---------------------------------------------------------------------------
set(_CVFN_INT_DIR  "${CMAKE_CURRENT_BINARY_DIR}/cv_frontend_norm_integration")
set(_CVFN_NORM_OUT "${_CVFN_INT_DIR}/cv_normalized.mlir")

if(MLIR_OPT_BIN)
  add_test(NAME CVFrontendNormalizationTest
    COMMAND ${CMAKE_COMMAND}
      -DMLIR_OPT=${MLIR_OPT_BIN}
      -DPLUGIN=$<TARGET_FILE:HIRMatMulBiasReluFusionPass>
      -DRAW_MLIR=${REPO_ROOT}/mlir/cv_raw_yoloseg.mlir
      -DNORM_OUT=${_CVFN_NORM_OUT}
      -P ${CMAKE_CURRENT_SOURCE_DIR}/test/serving/RunCVFrontendNormalizationTest.cmake
  )
  set_tests_properties(CVFrontendNormalizationTest PROPERTIES
    DEPENDS HIRMatMulBiasReluFusionPass
  )
endif()

# ---------------------------------------------------------------------------
# CTest: CVShapeInferenceTest
# Runs mlir-opt with cv-shape-inference on the dedicated small-shape test
# file and checks that num_elements, bytes_estimate, and function summary
# attrs are emitted correctly.
# No compile-for-target step (no CV plan builder yet).
# Skipped if mlir-opt is not found.
# ---------------------------------------------------------------------------
set(_CVSI_INT_DIR  "${CMAKE_CURRENT_BINARY_DIR}/cv_shape_inference_integration")
set(_CVSI_NORM_OUT "${_CVSI_INT_DIR}/cv_shape_inferred.mlir")

if(MLIR_OPT_BIN)
  add_test(NAME CVShapeInferenceTest
    COMMAND ${CMAKE_COMMAND}
      -DMLIR_OPT=${MLIR_OPT_BIN}
      -DPLUGIN=$<TARGET_FILE:HIRMatMulBiasReluFusionPass>
      -DINPUT_MLIR=${CMAKE_CURRENT_SOURCE_DIR}/test/serving/cv_shape_inference.mlir
      -DNORM_OUT=${_CVSI_NORM_OUT}
      -P ${CMAKE_CURRENT_SOURCE_DIR}/test/serving/RunCVShapeInferenceTest.cmake
  )
  set_tests_properties(CVShapeInferenceTest PROPERTIES
    DEPENDS HIRMatMulBiasReluFusionPass
  )
endif()

# ---------------------------------------------------------------------------
# CTest: CVMemoryPlanningTest
# Primary: runs cv-memory-planning on the dedicated small-shape test file and
#   checks buffer assignment, lifetime, and reuse attrs.
# Secondary smoke: runs cv-shape-inference,cv-memory-planning on
#   cv_raw_yoloseg.mlir and checks for "completed" status.
# Skipped if mlir-opt is not found.
# ---------------------------------------------------------------------------
set(_CVMP_INT_DIR  "${CMAKE_CURRENT_BINARY_DIR}/cv_memory_planning_integration")
set(_CVMP_PLAN_OUT "${_CVMP_INT_DIR}/cv_memory_planned.mlir")
set(_CVMP_SMOKE_OUT "${_CVMP_INT_DIR}/cv_yoloseg_memory_planned.mlir")

if(MLIR_OPT_BIN)
  add_test(NAME CVMemoryPlanningTest
    COMMAND ${CMAKE_COMMAND}
      -DMLIR_OPT=${MLIR_OPT_BIN}
      -DPLUGIN=$<TARGET_FILE:HIRMatMulBiasReluFusionPass>
      -DINPUT_MLIR=${CMAKE_CURRENT_SOURCE_DIR}/test/serving/cv_memory_planning.mlir
      -DPLAN_OUT=${_CVMP_PLAN_OUT}
      -DYOLOSEG_MLIR=${REPO_ROOT}/mlir/cv_raw_yoloseg.mlir
      -DSMOKE_OUT=${_CVMP_SMOKE_OUT}
      -P ${CMAKE_CURRENT_SOURCE_DIR}/test/serving/RunCVMemoryPlanningTest.cmake
  )
  set_tests_properties(CVMemoryPlanningTest PROPERTIES
    DEPENDS HIRMatMulBiasReluFusionPass
  )
endif()

# ---------------------------------------------------------------------------
# CTest: CVExecutionDomainPlanningTest
# Primary: runs cv-execution-domain-planning on the dedicated test file and
#   checks that execution domain classification attrs are emitted correctly.
# Secondary smoke: runs the full 4-pass CV pipeline on cv_raw_yoloseg.mlir
#   and checks for "completed" status.
# Skipped if mlir-opt is not found.
# ---------------------------------------------------------------------------
set(_CVEDP_INT_DIR   "${CMAKE_CURRENT_BINARY_DIR}/cv_execution_domain_planning_integration")
set(_CVEDP_PLAN_OUT  "${_CVEDP_INT_DIR}/cv_execution_domain_planned.mlir")
set(_CVEDP_SMOKE_OUT "${_CVEDP_INT_DIR}/cv_yoloseg_execution_domain_planned.mlir")

if(MLIR_OPT_BIN)
  add_test(NAME CVExecutionDomainPlanningTest
    COMMAND ${CMAKE_COMMAND}
      -DMLIR_OPT=${MLIR_OPT_BIN}
      -DPLUGIN=$<TARGET_FILE:HIRMatMulBiasReluFusionPass>
      -DINPUT_MLIR=${CMAKE_CURRENT_SOURCE_DIR}/test/serving/cv_execution_domain_planning.mlir
      -DPLAN_OUT=${_CVEDP_PLAN_OUT}
      -DYOLOSEG_MLIR=${REPO_ROOT}/mlir/cv_raw_yoloseg.mlir
      -DSMOKE_OUT=${_CVEDP_SMOKE_OUT}
      -P ${CMAKE_CURRENT_SOURCE_DIR}/test/serving/RunCVExecutionDomainPlanningTest.cmake
  )
  set_tests_properties(CVExecutionDomainPlanningTest PROPERTIES
    DEPENDS HIRMatMulBiasReluFusionPass
  )
endif()
