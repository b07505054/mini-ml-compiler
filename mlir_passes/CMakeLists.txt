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
file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/include/CV/IR)

set(LLVM_TARGET_DEFINITIONS include/HIR/IR/HIROps.td)
mlir_tablegen(include/HIR/IR/HIROps.h.inc -gen-op-decls)
mlir_tablegen(include/HIR/IR/HIROps.cpp.inc -gen-op-defs)
mlir_tablegen(include/HIR/IR/HIROpsDialect.h.inc -gen-dialect-decls -dialect=hir)
mlir_tablegen(include/HIR/IR/HIROpsDialect.cpp.inc -gen-dialect-defs -dialect=hir)
add_public_tablegen_target(HIRDialectIncGen)

set(LLVM_TARGET_DEFINITIONS include/CV/IR/CVOps.td)
mlir_tablegen(include/CV/IR/CVOps.h.inc -gen-op-decls)
mlir_tablegen(include/CV/IR/CVOps.cpp.inc -gen-op-defs)
mlir_tablegen(include/CV/IR/CVOpsDialect.h.inc -gen-dialect-decls -dialect=cv)
mlir_tablegen(include/CV/IR/CVOpsDialect.cpp.inc -gen-dialect-defs -dialect=cv)
add_public_tablegen_target(CVDialectIncGen)

set(LLVM_TARGET_DEFINITIONS include/FusionPasses.td)
mlir_tablegen(include/FusionPasses.h.inc -gen-pass-decls)
add_public_tablegen_target(HIRFusionPassIncGen)

file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/include)

set(ML_GRAPH_COMPILER_PASS_SOURCES
  lib/CV/IR/CVDialect.cpp
  lib/CV/IR/CVOps.cpp
  lib/HIR/IR/HIRDialect.cpp
  lib/HIR/IR/HIROps.cpp
  lib/MatMulBiasReluFusionPass.cpp
  lib/NativeCodegenPipeline.cpp
  lib/QuantizationCompilerPasses.cpp
  lib/QuantizationPlanningPass.cpp
  lib/ServingPhaseAnalysisPass.cpp
  lib/serving/KVLayoutPlanningPass.cpp
  lib/serving/ReplayEligibilityPass.cpp
  lib/serving/ServingPipeline.cpp
  lib/serving/TargetConstraints.cpp
  lib/serving/ExecutionProviderPlanningPass.cpp
  lib/serving/RepresentationPlanningPass.cpp
  lib/serving/LayoutPlanningPass.cpp
  lib/serving/BoundaryPlanningPass.cpp
  lib/serving/BoundaryMaterializationPass.cpp
  lib/serving/TilePlanningPass.cpp
  lib/serving/KernelSelectionPass.cpp
  lib/serving/QuantizationCoDesignPass.cpp
  lib/serving/WeightClassificationPlanningPass.cpp
  lib/serving/QuantizationStrategyPlanningPass.cpp
  lib/serving/CVExecutionPlanAttrsPass.cpp
  lib/serving/KernelAvailabilityPlanningPass.cpp
  lib/serving/LoweringDecisionPlanningPass.cpp
  lib/serving/QuantizedBoundaryRefinementPass.cpp
  lib/serving/AlternativeLoweringPlanningPass.cpp
  lib/serving/CandidateGenerationPass.cpp
  lib/serving/ServingCostModelPass.cpp
  lib/serving/PlanSelectionPass.cpp
  lib/serving/LLMFrontendNormalizationPass.cpp
  lib/serving/AttentionImplementationPasses.cpp
  lib/serving/CVFrontendNormalizationPass.cpp
  lib/serving/CVShapeInferencePass.cpp
  lib/serving/CVMemoryPlanningPass.cpp
  lib/serving/CVExecutionDomainPlanningPass.cpp
  lib/serving/CVSemanticAnnotationPass.cpp
  lib/serving/CVExecutionPlanBuilder.cpp
  lib/serving/DistributedPlanning.cpp
  lib/serving/DistributedStrategyPlanningPass.cpp
)

add_library(MLGraphCompilerPasses STATIC
  ${ML_GRAPH_COMPILER_PASS_SOURCES}
)
set_target_properties(MLGraphCompilerPasses PROPERTIES
  POSITION_INDEPENDENT_CODE ON
)
llvm_update_compile_flags(MLGraphCompilerPasses)

add_dependencies(MLGraphCompilerPasses
  CVDialectIncGen
  HIRDialectIncGen
  HIRFusionPassIncGen
)

target_include_directories(MLGraphCompilerPasses
  PUBLIC
  ${LLVM_INCLUDE_DIRS}
  ${MLIR_INCLUDE_DIRS}
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${CMAKE_CURRENT_BINARY_DIR}/include
)

target_compile_definitions(MLGraphCompilerPasses
  PRIVATE
  ${LLVM_DEFINITIONS}
)

target_link_libraries(MLGraphCompilerPasses
  PUBLIC
    MLIRIR
    MLIRParser
    MLIRPass
    MLIRFuncDialect
    MLIRTensorDialect
    MLIRLinalgDialect
    MLIRArithDialect
    MLIRMathDialect
    MLIRSCFDialect
    MLIRVectorDialect
    MLIRTransforms
    MLIRSupport
    LLVMSupport
)

add_library(HIRMatMulBiasReluFusionPass MODULE
  lib/PluginRegistration.cpp
)
llvm_update_compile_flags(HIRMatMulBiasReluFusionPass)
add_dependencies(HIRMatMulBiasReluFusionPass
  CVDialectIncGen
  HIRDialectIncGen
  HIRFusionPassIncGen
)
target_link_libraries(HIRMatMulBiasReluFusionPass
  PRIVATE MLGraphCompilerPasses
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

enable_testing()

add_executable(MatMulLoweringCostModelV2Test
  test/MatMulLoweringCostModelV2Test.cpp
)
target_include_directories(MatMulLoweringCostModelV2Test PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)
add_test(NAME MatMulLoweringCostModelV2Test
  COMMAND MatMulLoweringCostModelV2Test
)

# ---------------------------------------------------------------------------
# CTest: ExecutionPlanBuilder collector test (Qwen 2.5-0.5B)
# Exercises ExecutionPlanBuilder as a collector/packer on Qwen serving MLIR.
# Runs the 4-pass serving pipeline, then verifies the V2 plan structure.
# No exporter, JSON, or artifact generation.  V1 builder untouched.
# ---------------------------------------------------------------------------
add_executable(PlanBuilderQwenTest
  lib/ServingPhaseAnalysisPass.cpp
  lib/serving/KVLayoutPlanningPass.cpp
  lib/serving/ReplayEligibilityPass.cpp
  lib/serving/ExecutionProviderPlanningPass.cpp
  lib/serving/ExecutionPlanBuilder.cpp
  test/serving/plan_builder_qwen_test.cpp
)
llvm_update_compile_flags(PlanBuilderQwenTest)
add_dependencies(PlanBuilderQwenTest
  HIRFusionPassIncGen
)
target_include_directories(PlanBuilderQwenTest
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_BINARY_DIR}/include
    ${LLVM_INCLUDE_DIRS}
    ${MLIR_INCLUDE_DIRS}
)
target_compile_definitions(PlanBuilderQwenTest
  PRIVATE ${LLVM_DEFINITIONS}
)
target_link_libraries(PlanBuilderQwenTest
  PRIVATE
    MLIRIR
    MLIRParser
    MLIRPass
    MLIRFuncDialect
    MLIRTensorDialect
    MLIRLinalgDialect
    MLIRArithDialect
    MLIRMathDialect
    MLIRTransforms
    MLIRSupport
)
add_test(NAME PlanBuilderQwenTest
         COMMAND PlanBuilderQwenTest)

# ---------------------------------------------------------------------------
# CTest: ExecutionPlanBuilder full 16-pass pipeline test (Qwen 2.5-0.5B)
# Runs the complete serving pipeline including KernelAvailabilityPlanningPass
# and LoweringDecisionPlanningPass, then verifies that per-op KernelDecisions
# are collected from lowering.decision / kernel.* attrs.
# Requires target.kernel_libraries.cpu in the MLIR to activate
# KernelAvailabilityPlanningPass (standard pipeline mechanism, not faking).
# No V1 builder linked.  No exporter, JSON, or artifact generation.
# ---------------------------------------------------------------------------
add_executable(PlanBuilderQwenFullPipelineTest
  lib/ServingPhaseAnalysisPass.cpp
  lib/serving/KVLayoutPlanningPass.cpp
  lib/serving/ReplayEligibilityPass.cpp
  lib/serving/ExecutionProviderPlanningPass.cpp
  lib/serving/RepresentationPlanningPass.cpp
  lib/serving/LayoutPlanningPass.cpp
  lib/serving/BoundaryPlanningPass.cpp
  lib/serving/WeightClassificationPlanningPass.cpp
  lib/serving/QuantizationStrategyPlanningPass.cpp
  lib/serving/CVExecutionPlanAttrsPass.cpp
  lib/serving/KernelAvailabilityPlanningPass.cpp
  lib/serving/LoweringDecisionPlanningPass.cpp
  lib/serving/QuantizedBoundaryRefinementPass.cpp
  lib/serving/AlternativeLoweringPlanningPass.cpp
  lib/serving/CandidateGenerationPass.cpp
  lib/serving/ServingCostModelPass.cpp
  lib/serving/PlanSelectionPass.cpp
  lib/serving/ExecutionPlanBuilder.cpp
  test/serving/plan_builder_qwen_full_pipeline_test.cpp
)
llvm_update_compile_flags(PlanBuilderQwenFullPipelineTest)
add_dependencies(PlanBuilderQwenFullPipelineTest
  HIRFusionPassIncGen
)
target_include_directories(PlanBuilderQwenFullPipelineTest
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_BINARY_DIR}/include
    ${LLVM_INCLUDE_DIRS}
    ${MLIR_INCLUDE_DIRS}
)
target_compile_definitions(PlanBuilderQwenFullPipelineTest
  PRIVATE ${LLVM_DEFINITIONS}
)
target_link_libraries(PlanBuilderQwenFullPipelineTest
  PRIVATE
    MLIRIR
    MLIRParser
    MLIRPass
    MLIRFuncDialect
    MLIRTensorDialect
    MLIRLinalgDialect
    MLIRArithDialect
    MLIRMathDialect
    MLIRTransforms
    MLIRSupport
)
add_test(NAME PlanBuilderQwenFullPipelineTest
         COMMAND PlanBuilderQwenFullPipelineTest)

# ---------------------------------------------------------------------------
# CTest: ServingStaticCostModelV1Test
# Verifies ServingCostModelPass cost components, total == sum invariant,
# cost_model_id, truth_boundary, and V0 backward-compat penalty_score.
# ---------------------------------------------------------------------------
add_executable(ServingStaticCostModelV1Test
  lib/serving/ServingCostModelPass.cpp
  test/serving/serving_static_cost_model_v1_test.cpp
)
llvm_update_compile_flags(ServingStaticCostModelV1Test)
add_dependencies(ServingStaticCostModelV1Test
  HIRFusionPassIncGen
)
target_include_directories(ServingStaticCostModelV1Test
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_BINARY_DIR}/include
    ${LLVM_INCLUDE_DIRS}
    ${MLIR_INCLUDE_DIRS}
)
target_compile_definitions(ServingStaticCostModelV1Test
  PRIVATE ${LLVM_DEFINITIONS}
)
target_link_libraries(ServingStaticCostModelV1Test
  PRIVATE
    MLIRIR
    MLIRParser
    MLIRPass
    MLIRFuncDialect
    MLIRTransforms
    MLIRSupport
)
add_test(NAME ServingStaticCostModelV1Test
         COMMAND ServingStaticCostModelV1Test)

# ---------------------------------------------------------------------------
# CTest: ImplementationCandidate unit test
# Verifies the compiler-internal candidate encode/decode boundary, stable
# fallback identity, feasibility mapping, and PolicyResult separation.
# No Runtime, JSON, target profile, or ExecutionPlan export.
# ---------------------------------------------------------------------------
add_executable(ImplementationCandidateTest
  test/serving/ImplementationCandidateTest.cpp
)
llvm_update_compile_flags(ImplementationCandidateTest)
target_include_directories(ImplementationCandidateTest
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_BINARY_DIR}/include
    ${LLVM_INCLUDE_DIRS}
    ${MLIR_INCLUDE_DIRS}
)
target_compile_definitions(ImplementationCandidateTest
  PRIVATE ${LLVM_DEFINITIONS}
)
target_link_libraries(ImplementationCandidateTest
  PRIVATE
    MLIRIR
    MLIRSupport
)
add_test(NAME ImplementationCandidateTest
         COMMAND ImplementationCandidateTest)

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
# CTest: DistributedPlanning unit test (D1: compiler-planned TP=2
# multi-process simulation). Pure C++ structs — no MLIR IR, no Python.
# ---------------------------------------------------------------------------
add_executable(DistributedPlanningTest
  lib/serving/DistributedPlanning.cpp
  lib/serving/ExecutionPlanExporter.cpp
  test/serving/DistributedPlanningTest.cpp
)
llvm_update_compile_flags(DistributedPlanningTest)
target_include_directories(DistributedPlanningTest
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_BINARY_DIR}/include
    ${LLVM_INCLUDE_DIRS}
    ${MLIR_INCLUDE_DIRS}
)
target_compile_definitions(DistributedPlanningTest
  PRIVATE ${LLVM_DEFINITIONS}
)
target_link_libraries(DistributedPlanningTest
  PRIVATE
    MLIRIR
    MLIRSupport
    LLVMSupport
)
add_test(NAME DistributedPlanningTest
         COMMAND DistributedPlanningTest)

# ---------------------------------------------------------------------------
# CTest: DistributedStrategyPlanningTest (D2 Qwen-aware legality/cost unit
# test). Pure C++ structs -- no MLIR IR.
# ---------------------------------------------------------------------------
add_executable(DistributedStrategyPlanningTest
  lib/serving/DistributedPlanning.cpp
  test/serving/DistributedStrategyPlanningTest.cpp
)
llvm_update_compile_flags(DistributedStrategyPlanningTest)
target_include_directories(DistributedStrategyPlanningTest
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_BINARY_DIR}/include
    ${LLVM_INCLUDE_DIRS}
    ${MLIR_INCLUDE_DIRS}
)
target_compile_definitions(DistributedStrategyPlanningTest
  PRIVATE ${LLVM_DEFINITIONS}
)
target_link_libraries(DistributedStrategyPlanningTest
  PRIVATE
    MLIRIR
    MLIRSupport
    LLVMSupport
)
add_test(NAME DistributedStrategyPlanningTest
         COMMAND DistributedStrategyPlanningTest)

# ---------------------------------------------------------------------------
# CTest: DistributedProfitabilityTest (D6 distributed_profitability_
# contract_v1 unit test). Pure C++ structs -- no MLIR IR.
# ---------------------------------------------------------------------------
add_executable(DistributedProfitabilityTest
  lib/serving/DistributedPlanning.cpp
  test/serving/DistributedProfitabilityTest.cpp
)
llvm_update_compile_flags(DistributedProfitabilityTest)
target_include_directories(DistributedProfitabilityTest
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_BINARY_DIR}/include
    ${LLVM_INCLUDE_DIRS}
    ${MLIR_INCLUDE_DIRS}
)
target_compile_definitions(DistributedProfitabilityTest
  PRIVATE ${LLVM_DEFINITIONS}
)
target_link_libraries(DistributedProfitabilityTest
  PRIVATE
    MLIRIR
    MLIRSupport
    LLVMSupport
)
add_test(NAME DistributedProfitabilityTest
         COMMAND DistributedProfitabilityTest)

# ---------------------------------------------------------------------------
# CTest: DistributedStrategyPlanningPipelineTest (D2 end-to-end integration
# test against the real per-layer Qwen ONNX graph and the real
# compile-for-target production pipeline).
# ---------------------------------------------------------------------------
set(_D2_PIPELINE_DIR "${CMAKE_CURRENT_BINARY_DIR}/distributed_strategy_planning_pipeline")
# D6 rationale for the FACTS/PROFILE_TP2/MODEL_PROFILE/WORKLOAD_PROFILE
# changes below: this test used to prove "opt_in=true legally forces TP2"
# against the 0.5B model on the (single, laptop-GPU) GTX1650 opt-in
# profile. D6 replaces that mechanism with a real profitability
# comparison, and the real, measured D5 result is that the 0.5B model
# never profitably selects TP2 in the calibrated workload range (TP1
# wins 35/36 measured cells -- see
# heterogeneous-inference-runtime/results/runtime_paths/
# distributed_d5_compiler_tp_policy/). Forcing a TP2 assertion for 0.5B
# here would require either an artificial workload/calibration input or
# reverting to the old opt-in-forces-TP2 mechanism -- both dishonest.
# This test now uses the 7B model against the real, D5-calibrated RTX4090
# profile, where TP2 genuinely and profitably wins (12/12 measured D5
# cells) -- the same real crossover the D6 held-out reproduction found.
# The TP1 (non-opt-in) backward-compatibility assertion is unaffected by
# this change: opt_in=false still means no "distributed" key regardless
# of model or calibration.
add_test(NAME DistributedStrategyPlanningPipelineTest
  COMMAND ${CMAKE_COMMAND}
    -DONNX_TOOL=$<TARGET_FILE:qwen-onnx-to-serving-mlir>
    -DCFT_TOOL=$<TARGET_FILE:compile-for-target>
    -DFACTS=${CMAKE_CURRENT_SOURCE_DIR}/../configs/models/qwen_7b_onnx_graph_facts.json
    -DPROFILE_TP1=${CMAKE_CURRENT_SOURCE_DIR}/../configs/target_profiles/nvidia_gtx1650_maxq.json
    -DPROFILE_TP2=${CMAKE_CURRENT_SOURCE_DIR}/../configs/target_profiles/nvidia_rtx4090_d6_distributed_profitability.json
    -DMODEL_PROFILE=${CMAKE_CURRENT_SOURCE_DIR}/../configs/models/qwen_7b_model_profile.json
    -DWORKLOAD_PROFILE=${CMAKE_CURRENT_SOURCE_DIR}/../configs/workloads/qwen7b_in32_out32_c1.json
    -DRAW_OUT=${_D2_PIPELINE_DIR}/raw.mlir
    -DJSON_TP1_OUT=${_D2_PIPELINE_DIR}/real_qwen_tp1_execution_plan.json
    -DJSON_TP2_OUT=${_D2_PIPELINE_DIR}/real_qwen_tp2_execution_plan.json
    -DEVIDENCE_OUT=${_D2_PIPELINE_DIR}/distributed_evidence_report.json
    -P ${CMAKE_CURRENT_SOURCE_DIR}/test/serving/RunDistributedStrategyPlanningPipelineTest.cmake
)
set_tests_properties(DistributedStrategyPlanningPipelineTest PROPERTIES
  DEPENDS "qwen-onnx-to-serving-mlir;compile-for-target"
)

# ---------------------------------------------------------------------------
# CTest: CVExecutionPlanBuilder unit test
# Runs the four CV planning passes, then builds an in-memory typed plan.
# No exporter, JSON, artifact generation, or pass registration.
# ---------------------------------------------------------------------------
add_executable(CVExecutionPlanBuilderTest
  lib/CV/IR/CVDialect.cpp
  lib/CV/IR/CVOps.cpp
  lib/serving/CVFrontendNormalizationPass.cpp
  lib/serving/CVShapeInferencePass.cpp
  lib/serving/CVMemoryPlanningPass.cpp
  lib/serving/CVExecutionDomainPlanningPass.cpp
  lib/serving/CVExecutionPlanBuilder.cpp
  test/serving/CVExecutionPlanBuilderTest.cpp
)
llvm_update_compile_flags(CVExecutionPlanBuilderTest)
add_dependencies(CVExecutionPlanBuilderTest
  CVDialectIncGen
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
# CTest: CVExecutionPlanExporter unit test
# Builds a typed CV plan, exports JSON, and checks schema sentinel fields.
# No artifact generator CLI, pass registration, or backend mapping.
# ---------------------------------------------------------------------------
add_executable(CVExecutionPlanExporterTest
  lib/CV/IR/CVDialect.cpp
  lib/CV/IR/CVOps.cpp
  lib/serving/CVFrontendNormalizationPass.cpp
  lib/serving/CVShapeInferencePass.cpp
  lib/serving/CVMemoryPlanningPass.cpp
  lib/serving/CVExecutionDomainPlanningPass.cpp
  lib/serving/CVExecutionPlanBuilder.cpp
  lib/serving/CVExecutionPlanExporter.cpp
  test/serving/CVExecutionPlanExporterTest.cpp
)
llvm_update_compile_flags(CVExecutionPlanExporterTest)
add_dependencies(CVExecutionPlanExporterTest
  CVDialectIncGen
  HIRFusionPassIncGen
)
target_include_directories(CVExecutionPlanExporterTest
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_BINARY_DIR}/include
    ${LLVM_INCLUDE_DIRS}
    ${MLIR_INCLUDE_DIRS}
)
target_compile_definitions(CVExecutionPlanExporterTest
  PRIVATE ${LLVM_DEFINITIONS}
)
target_link_libraries(CVExecutionPlanExporterTest
  PRIVATE
    MLIRIR
    MLIRParser
    MLIRPass
    MLIRFuncDialect
    MLIRTransforms
    MLIRSupport
    LLVMSupport
)
add_test(NAME CVExecutionPlanExporterTest
         COMMAND CVExecutionPlanExporterTest)

# ---------------------------------------------------------------------------
# emit-cv-execution-plan: CV artifact generator CLI
#
# Reads CV MLIR, runs the CV planning pipeline, builds CVExecutionPlan, and
# exports the JSON plan through CVExecutionPlanExporter.
# ---------------------------------------------------------------------------
add_executable(emit-cv-execution-plan
  lib/CV/IR/CVDialect.cpp
  lib/CV/IR/CVOps.cpp
  lib/serving/CVFrontendNormalizationPass.cpp
  lib/serving/CVShapeInferencePass.cpp
  lib/serving/CVMemoryPlanningPass.cpp
  lib/serving/CVExecutionDomainPlanningPass.cpp
  lib/serving/CVExecutionPlanBuilder.cpp
  lib/serving/CVExecutionPlanExporter.cpp
  tools/emit-cv-execution-plan/main.cpp
)
llvm_update_compile_flags(emit-cv-execution-plan)
add_dependencies(emit-cv-execution-plan
  CVDialectIncGen
  HIRFusionPassIncGen
)
target_include_directories(emit-cv-execution-plan
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_BINARY_DIR}/include
    ${LLVM_INCLUDE_DIRS}
    ${MLIR_INCLUDE_DIRS}
)
target_compile_definitions(emit-cv-execution-plan PRIVATE ${LLVM_DEFINITIONS})
target_link_libraries(emit-cv-execution-plan
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
# emit-distributed-execution-plan: D1 distributed candidate export CLI
#
# Generates the fixed {TP1, TP2} distributed candidates for a synthetic
# sharded-matmul problem, applies D1 legality filtering, and exports the
# selected candidate as a real ExecutionPlan schema_version 2.0.0 artifact.
# See mlir_passes/tools/emit-distributed-execution-plan/main.cpp.
# ---------------------------------------------------------------------------
add_executable(emit-distributed-execution-plan
  lib/serving/DistributedPlanning.cpp
  lib/serving/ExecutionPlanExporter.cpp
  tools/emit-distributed-execution-plan/main.cpp
)
llvm_update_compile_flags(emit-distributed-execution-plan)
target_include_directories(emit-distributed-execution-plan
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_BINARY_DIR}/include
    ${LLVM_INCLUDE_DIRS}
    ${MLIR_INCLUDE_DIRS}
)
target_compile_definitions(emit-distributed-execution-plan PRIVATE ${LLVM_DEFINITIONS})
target_link_libraries(emit-distributed-execution-plan
  PRIVATE
    MLIRIR
    MLIRSupport
    LLVMSupport
)

set(_CV_PLAN_TOOL_DIR "${CMAKE_CURRENT_BINARY_DIR}/cv_execution_plan_artifact_tool")
set(_CV_PLAN_TOOL_OUT "${_CV_PLAN_TOOL_DIR}/cv_execution_plan.json")

add_test(NAME CVExecutionPlanArtifactToolTest
  COMMAND ${CMAKE_COMMAND}
    -DTOOL=$<TARGET_FILE:emit-cv-execution-plan>
    -DINPUT_MLIR=${CMAKE_CURRENT_SOURCE_DIR}/../mlir/cv_raw_yoloseg.mlir
    -DOUT=${_CV_PLAN_TOOL_OUT}
    -P ${CMAKE_CURRENT_SOURCE_DIR}/test/serving/RunCVExecutionPlanArtifactToolTest.cmake
)
set_tests_properties(CVExecutionPlanArtifactToolTest PROPERTIES
  DEPENDS emit-cv-execution-plan
)

# ---------------------------------------------------------------------------
# compile-for-target: driver tool
#
# Wires: TargetDeviceProfile JSON → TargetConstraints + CapabilityBundle →
#        serving passes (15-pass pipeline) →
#        ExecutionPlanBuilder → ExecutionPlanExporter →
#        canonical artifact (execution_plan.json).
#
# Pass and pipeline implementations come from MLGraphCompilerPasses, shared
# with the plugin; plugin entry points remain isolated in PluginRegistration.
# JSON construction is entirely owned by ExecutionPlanExporter; main.cpp
# never constructs llvm::json types directly.
# ---------------------------------------------------------------------------
set(REPO_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/..")

add_executable(compile-for-target
  lib/serving/ExecutionPlanBuilder.cpp
  lib/serving/ExecutionPlanExporter.cpp
  tools/compile-for-target/main.cpp
)
llvm_update_compile_flags(compile-for-target)
add_dependencies(compile-for-target HIRDialectIncGen HIRFusionPassIncGen)
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
    MLGraphCompilerPasses
    MLIR
    MLIRIR
    MLIRParser
    MLIRPass
    MLIRFuncDialect
    MLIRTensorDialect
    MLIRLinalgDialect
    MLIRArithDialect
    MLIRMathDialect
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
set(_CFT_CANONICAL "${CMAKE_CURRENT_BINARY_DIR}/test_execution_plan.json")

add_test(NAME CompileForTargetTest
  COMMAND ${CMAKE_COMMAND}
    -DTOOL=$<TARGET_FILE:compile-for-target>
    -DPROFILE=${REPO_ROOT}/configs/target_profiles/apple_a17pro_mobile.json
    -DMLIR=${REPO_ROOT}/mlir/tiny_gpt_serving.mlir
    -DOUT=${_CFT_CANONICAL}
    -P ${CMAKE_CURRENT_SOURCE_DIR}/test/serving/RunCompileForTargetTest.cmake
)
add_test(NAME AttentionCPUContractTest
  COMMAND ${CMAKE_COMMAND}
    -DTOOL=$<TARGET_FILE:compile-for-target>
    -DPROFILE=${REPO_ROOT}/configs/target_profiles/apple_a17pro_mobile.json
    -DINPUT=${REPO_ROOT}/mlir/attention_cpu_supported.mlir
    -DOUT=${CMAKE_CURRENT_BINARY_DIR}/attention_cpu_execution_plan.json
    -DDUMP=${CMAKE_CURRENT_BINARY_DIR}/attention_cpu_lowered.mlir
    -P ${CMAKE_CURRENT_SOURCE_DIR}/test/serving/RunAttentionCPUContractTest.cmake)
add_test(NAME AttentionCPUFailClosedTest
  COMMAND ${CMAKE_COMMAND}
    -DTOOL=$<TARGET_FILE:compile-for-target>
    -DPROFILE=${REPO_ROOT}/configs/target_profiles/apple_a17pro_mobile.json
    -DINPUT=${REPO_ROOT}/mlir/attention_cpu_unsupported.mlir
    -DOUT=${CMAKE_CURRENT_BINARY_DIR}/attention_cpu_unsupported_plan.json
    -DDUMP=${CMAKE_CURRENT_BINARY_DIR}/attention_cpu_unsupported_lowered.mlir
    -P ${CMAKE_CURRENT_SOURCE_DIR}/test/serving/RunAttentionCPUFailClosedTest.cmake)
add_test(NAME ContiguousKVLegalityTest
  COMMAND ${CMAKE_COMMAND}
    -DTOOL=$<TARGET_FILE:compile-for-target>
    -DPROFILE=${REPO_ROOT}/configs/target_profiles/apple_a17pro_mobile.json
    -DINPUT=${REPO_ROOT}/mlir/attention_cpu_supported.mlir
    -DOUT_DIR=${CMAKE_CURRENT_BINARY_DIR}
    -P ${CMAKE_CURRENT_SOURCE_DIR}/test/serving/RunContiguousKVLegalityTest.cmake)
add_test(NAME PagedKVContractTest
  COMMAND ${CMAKE_COMMAND}
    -DTOOL=$<TARGET_FILE:compile-for-target>
    -DPROFILE=${REPO_ROOT}/configs/target_profiles/apple_a17pro_mobile.json
    -DINPUT=${REPO_ROOT}/mlir/attention_cpu_paged.mlir
    -DOUT=${CMAKE_CURRENT_BINARY_DIR}/attention_cpu_paged_plan.json
    -DDUMP=${CMAKE_CURRENT_BINARY_DIR}/attention_cpu_paged_lowered.mlir
    -P ${CMAKE_CURRENT_SOURCE_DIR}/test/serving/RunPagedKVContractTest.cmake)
set_tests_properties(CompileForTargetTest PROPERTIES
  DEPENDS compile-for-target
)

# ---------------------------------------------------------------------------
# CTest: RaspberryPiFusedMatMulBiasReluKernelSelectionTest (Phase P1B)
# Proves the ONE evidence-backed runtime kernel declaration in
# raspberry_pi5_cortex_a76_cpu.json (kernel_selection_contract_v1) is
# actually selected end to end by the real, unmodified compile-for-target
# pipeline for hir.fused_matmul_bias_relu, and that no other op receives a
# fabricated selection. See DOC/result/ for the full P1B report.
# ---------------------------------------------------------------------------
set(_P1B_CANONICAL "${CMAKE_CURRENT_BINARY_DIR}/test_p1b_raspberry_pi_execution_plan.json")

add_test(NAME RaspberryPiFusedMatMulBiasReluKernelSelectionTest
  COMMAND ${CMAKE_COMMAND}
    -DTOOL=$<TARGET_FILE:compile-for-target>
    -DPROFILE=${REPO_ROOT}/configs/target_profiles/raspberry_pi5_cortex_a76_cpu.json
    -DMLIR=${REPO_ROOT}/mlir/p1b_fused_matmul_bias_relu_cpu.mlir
    -DOUT=${_P1B_CANONICAL}
    -P ${CMAKE_CURRENT_SOURCE_DIR}/test/serving/RunRaspberryPiFusedMatMulBiasReluKernelSelectionTest.cmake
)
set_tests_properties(RaspberryPiFusedMatMulBiasReluKernelSelectionTest PROPERTIES
  DEPENDS compile-for-target
)

# ---------------------------------------------------------------------------
# CTest: Slice3DQuantizedIRLoweringTest
# Proves a selected packed INT8 fused MatMul+Bias+ReLU decision is
# materialized into explicit HIR Q/DQ + packed weight IR, lowered to the
# selected portable CPU INT8 runtime-call contract, and exported with ordered
# runtime execution stages. Also checks selected INT8 fails when the packed
# artifact contract is incomplete.
# ---------------------------------------------------------------------------
set(_S3D_JSON "${CMAKE_CURRENT_BINARY_DIR}/slice3d_quantized_ir_plan.json")
set(_S3D_MLIR "${CMAKE_CURRENT_BINARY_DIR}/slice3d_quantized_ir_lowered.mlir")

add_test(NAME Slice3DQuantizedIRLoweringTest
  COMMAND ${CMAKE_COMMAND}
    -DTOOL=$<TARGET_FILE:compile-for-target>
    -DPROFILE=${REPO_ROOT}/configs/target_profiles/raspberry_pi5_cortex_a76_cpu.json
    -DMLIR=${CMAKE_CURRENT_SOURCE_DIR}/test/serving/slice3d_int8_materialization.mlir
    -DBAD_MLIR=${CMAKE_CURRENT_SOURCE_DIR}/test/serving/slice3d_int8_missing_packed_artifact.mlir
    -DOUT=${_S3D_JSON}
    -DDUMP_MLIR=${_S3D_MLIR}
    -P ${CMAKE_CURRENT_SOURCE_DIR}/test/serving/RunSlice3DQuantizedIRLoweringTest.cmake
)
set_tests_properties(Slice3DQuantizedIRLoweringTest PROPERTIES
  DEPENDS compile-for-target
)

# ---------------------------------------------------------------------------
# CTest: QuantStrategyExportTest
# Verifies per_op_quantization_decisions appear in the exported JSON when
# QuantizationStrategyPlanningPass has run (Commit H).
# ---------------------------------------------------------------------------
set(_QSE_OUT "${CMAKE_CURRENT_BINARY_DIR}/quant_strategy_plan.json")

add_test(NAME QuantStrategyExportTest
  COMMAND ${CMAKE_COMMAND}
    -DTOOL=$<TARGET_FILE:compile-for-target>
    -DPROFILE=${REPO_ROOT}/configs/target_profiles/apple_a17pro_mobile.json
    -DMLIR=${REPO_ROOT}/mlir/quant_strategy_serving.mlir
    -DOUT=${_QSE_OUT}
    -P ${CMAKE_CURRENT_SOURCE_DIR}/test/serving/RunQuantStrategyExportTest.cmake
)
set_tests_properties(QuantStrategyExportTest PROPERTIES
  DEPENDS compile-for-target
)

# ---------------------------------------------------------------------------
# CTest: LoweringDecisionExportTest
# Verifies per_op_decisions[].kernel appear in the exported V2 JSON when
# LoweringDecisionPlanningPass has run.
# Covers: direct_lower, rewrite_then_lower, fallback_backend, unsupported.
# ---------------------------------------------------------------------------
set(_LDE_OUT "${CMAKE_CURRENT_BINARY_DIR}/lowering_decision_plan.json")

add_test(NAME LoweringDecisionExportTest
  COMMAND ${CMAKE_COMMAND}
    -DTOOL=$<TARGET_FILE:compile-for-target>
    -DPROFILE=${REPO_ROOT}/configs/target_profiles/lowering_decision_export_test.json
    -DMLIR=${REPO_ROOT}/mlir/lowering_decision_export_test.mlir
    -DOUT=${_LDE_OUT}
    -P ${CMAKE_CURRENT_SOURCE_DIR}/test/serving/RunLoweringDecisionExportTest.cmake
)
set_tests_properties(LoweringDecisionExportTest PROPERTIES
  DEPENDS compile-for-target
)

# ---------------------------------------------------------------------------
# CTest: AppleQuantLoweringDemoTest
# End-to-end Apple/CoreML quantization-lowering demo artifact (Commit L).
# Shows: hardware profile -> representation -> layout -> boundary ->
#        quant strategy -> kernel availability -> lowering decision -> JSON.
# Input: integration_bundle/apple_quant_lowering_demo/{input.mlir, target_profile.json}
# Tests: quant+lowering arrays present, weight_only_int8, fp16_fallback,
#        direct_lower, fallback_backend, truth_boundary fields, coreml primary.
# ---------------------------------------------------------------------------
set(_AQLD_OUT "${CMAKE_CURRENT_BINARY_DIR}/apple_quant_lowering_demo_plan.json")

add_test(NAME AppleQuantLoweringDemoTest
  COMMAND ${CMAKE_COMMAND}
    -DTOOL=$<TARGET_FILE:compile-for-target>
    -DPROFILE=${REPO_ROOT}/integration_bundle/apple_quant_lowering_demo/target_profile.json
    -DMLIR=${REPO_ROOT}/integration_bundle/apple_quant_lowering_demo/input.mlir
    -DOUT=${_AQLD_OUT}
    -P ${CMAKE_CURRENT_SOURCE_DIR}/test/serving/RunAppleQuantLoweringDemoTest.cmake
)
set_tests_properties(AppleQuantLoweringDemoTest PROPERTIES
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
# End-to-end: qwen-to-serving-mlir -> compile-for-target -> V2 JSON artifact check.
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
# CTest: QwenGTX1650CompileTest
# End-to-end: qwen-to-serving-mlir -> compile-for-target with GTX 1650 Max-Q profile.
# Asserts execution_plan.json exists, schema 2.0.0, strategy=none, no measured fields.
# ---------------------------------------------------------------------------
set(_QWEN_GTX_DIR  "${CMAKE_CURRENT_BINARY_DIR}/qwen_gtx1650_integration")
set(_QWEN_GTX_MLIR "${_QWEN_GTX_DIR}/qwen_0_5b_gtx1650.mlir")
set(_QWEN_GTX_JSON "${_QWEN_GTX_DIR}/execution_plan.json")

add_test(NAME QwenGTX1650CompileTest
  COMMAND ${CMAKE_COMMAND}
    -DQWEN_TOOL=$<TARGET_FILE:qwen-to-serving-mlir>
    -DCFT_TOOL=$<TARGET_FILE:compile-for-target>
    -DSPEC=${REPO_ROOT}/configs/models/qwen_0_5b_spec.json
    -DPROFILE=${REPO_ROOT}/configs/target_profiles/nvidia_gtx1650_maxq.json
    -DMLIR_OUT=${_QWEN_GTX_MLIR}
    -DJSON_OUT=${_QWEN_GTX_JSON}
    -P ${CMAKE_CURRENT_SOURCE_DIR}/test/serving/RunQwenGTX1650CompileTest.cmake
)
set_tests_properties(QwenGTX1650CompileTest PROPERTIES
  DEPENDS "qwen-to-serving-mlir;compile-for-target"
)

# ---------------------------------------------------------------------------
# CTest: QwenAwqForcedCompileTest
# End-to-end: qwen-to-serving-mlir -> compile-for-target with the experimental
# forced-AWQ GTX 1650 profile (Phase C, minimal, AWQ only).
# Asserts global_decisions.quantization carries strategy=weight_only_int4,
# algorithm=awq, quantized_model_artifact_ref, and the forced-quant
# truth_boundary; no measured fields present.
# ---------------------------------------------------------------------------
set(_QWEN_AWQ_DIR  "${CMAKE_CURRENT_BINARY_DIR}/qwen_awq_forced_integration")
set(_QWEN_AWQ_MLIR "${_QWEN_AWQ_DIR}/qwen_0_5b_awq.mlir")
set(_QWEN_AWQ_JSON "${_QWEN_AWQ_DIR}/execution_plan.json")

add_test(NAME QwenAwqForcedCompileTest
  COMMAND ${CMAKE_COMMAND}
    -DQWEN_TOOL=$<TARGET_FILE:qwen-to-serving-mlir>
    -DCFT_TOOL=$<TARGET_FILE:compile-for-target>
    -DSPEC=${REPO_ROOT}/configs/models/qwen_0_5b_spec.json
    -DPROFILE=${REPO_ROOT}/configs/target_profiles/nvidia_gtx1650_maxq_awq_forced.json
    -DMLIR_OUT=${_QWEN_AWQ_MLIR}
    -DJSON_OUT=${_QWEN_AWQ_JSON}
    -P ${CMAKE_CURRENT_SOURCE_DIR}/test/serving/RunQwenAwqForcedCompileTest.cmake
)
set_tests_properties(QwenAwqForcedCompileTest PROPERTIES
  DEPENDS "qwen-to-serving-mlir;compile-for-target"
)

# ---------------------------------------------------------------------------
# qwen-onnx-to-serving-mlir: ONNX-graph-facts frontend (Phase 1 scaffold)
#
# Reads configs/models/qwen_0_5b_onnx_graph_facts.json (a hand-authored
# ONNX-shaped fixture, NOT a real ONNX protobuf parse) and emits full
# per-layer-expanded, flat, unrolled RAW serving MLIR. Must be run through
# llm-frontend-normalization before compile-for-target.
# ---------------------------------------------------------------------------
add_executable(qwen-onnx-to-serving-mlir
  tools/qwen-onnx-to-serving-mlir/main.cpp
)
llvm_update_compile_flags(qwen-onnx-to-serving-mlir)
target_include_directories(qwen-onnx-to-serving-mlir
  PRIVATE
    ${LLVM_INCLUDE_DIRS}
    ${MLIR_INCLUDE_DIRS}
)
target_compile_definitions(qwen-onnx-to-serving-mlir PRIVATE ${LLVM_DEFINITIONS})
target_link_libraries(qwen-onnx-to-serving-mlir
  PRIVATE
    LLVMSupport
)

# ---------------------------------------------------------------------------
# CTest: QwenOnnxToServingMlirTest
# Runs qwen-onnx-to-serving-mlir and checks output MLIR for required attrs,
# full per-layer expansion (serving.layer_index 0..num_layers-1), and roles.
# ---------------------------------------------------------------------------
set(_QWEN_ONNX_MLIR_UNIT "${CMAKE_CURRENT_BINARY_DIR}/qwen_0_5b_onnx_unit.mlir")

add_test(NAME QwenOnnxToServingMlirTest
  COMMAND ${CMAKE_COMMAND}
    -DTOOL=$<TARGET_FILE:qwen-onnx-to-serving-mlir>
    -DFACTS=${REPO_ROOT}/configs/models/qwen_0_5b_onnx_graph_facts.json
    -DOUT=${_QWEN_ONNX_MLIR_UNIT}
    -P ${CMAKE_CURRENT_SOURCE_DIR}/test/serving/RunQwenOnnxToServingMlirTest.cmake
)
set_tests_properties(QwenOnnxToServingMlirTest PROPERTIES
  DEPENDS qwen-onnx-to-serving-mlir
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
find_program(FILECHECK_BIN
  NAMES FileCheck
  HINTS "${LLVM_TOOLS_BINARY_DIR}"
  DOC "Path to FileCheck (required for generic existing-dialect prototype tests)"
)

set(_LLMFN_INT_DIR  "${CMAKE_CURRENT_BINARY_DIR}/llm_frontend_norm_integration")
set(_LLMFN_NORM_MLIR "${_LLMFN_INT_DIR}/normalized.mlir")
set(_LLMFN_JSON      "${_LLMFN_INT_DIR}/raw_qwen_frontend_serving_plan.json")

if(MLIR_OPT_BIN)
  add_test(NAME CVDialectOpsTest
    COMMAND ${CMAKE_COMMAND}
      -DMLIR_OPT=${MLIR_OPT_BIN}
      -DPLUGIN=$<TARGET_FILE:HIRMatMulBiasReluFusionPass>
      -DINPUT=${CMAKE_CURRENT_SOURCE_DIR}/test/cv/cv_dialect_ops.mlir
      -P ${CMAKE_CURRENT_SOURCE_DIR}/test/cv/RunCVDialectOpsTest.cmake
  )
  set_tests_properties(CVDialectOpsTest PROPERTIES
    DEPENDS HIRMatMulBiasReluFusionPass
  )

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

  # -------------------------------------------------------------------------
  # CTest: QwenOnnxServingPlanExportTest
  # Full ONNX-graph-facts pipeline: qwen-onnx-to-serving-mlir ->
  # llm-frontend-normalization -> compile-for-target -> execution_plan.json.
  # Validates full per-layer expansion (24 distinct serving.layer_index
  # values) survives normalization and reaches the exported plan.
  # -------------------------------------------------------------------------
  set(_QWEN_ONNX_INT_DIR   "${CMAKE_CURRENT_BINARY_DIR}/qwen_onnx_serving_plan_integration")
  set(_QWEN_ONNX_RAW_MLIR  "${_QWEN_ONNX_INT_DIR}/qwen_0_5b_onnx_raw.mlir")
  set(_QWEN_ONNX_NORM_MLIR "${_QWEN_ONNX_INT_DIR}/qwen_0_5b_onnx_normalized.mlir")
  set(_QWEN_ONNX_JSON      "${_QWEN_ONNX_INT_DIR}/execution_plan.json")

  add_test(NAME QwenOnnxServingPlanExportTest
    COMMAND ${CMAKE_COMMAND}
      -DONNX_TOOL=$<TARGET_FILE:qwen-onnx-to-serving-mlir>
      -DMLIR_OPT=${MLIR_OPT_BIN}
      -DPLUGIN=$<TARGET_FILE:HIRMatMulBiasReluFusionPass>
      -DCFT_TOOL=$<TARGET_FILE:compile-for-target>
      -DFACTS=${REPO_ROOT}/configs/models/qwen_0_5b_onnx_graph_facts.json
      -DPROFILE=${REPO_ROOT}/configs/target_profiles/nvidia_gtx1650_maxq.json
      -DRAW_OUT=${_QWEN_ONNX_RAW_MLIR}
      -DNORM_OUT=${_QWEN_ONNX_NORM_MLIR}
      -DJSON_OUT=${_QWEN_ONNX_JSON}
      -P ${CMAKE_CURRENT_SOURCE_DIR}/test/serving/RunQwenOnnxServingPlanExportTest.cmake
  )
  set_tests_properties(QwenOnnxServingPlanExportTest PROPERTIES
    DEPENDS "qwen-onnx-to-serving-mlir;HIRMatMulBiasReluFusionPass;compile-for-target"
  )
else()
  message(STATUS
    "mlir-opt not found; LLMFrontendNormalizationTest and "
    "QwenOnnxServingPlanExportTest will not be added. "
    "Set MLIR_OPT_BIN or ensure mlir-opt is in PATH."
  )
endif()

if(MLIR_OPT_BIN AND FILECHECK_BIN)
  foreach(prototype IN ITEMS
      generic_resize_nearest_2x_prototype
      generic_conv_transpose2d_stride2_prototype)
    add_test(NAME ${prototype}Test
      COMMAND ${CMAKE_COMMAND}
        -DMLIR_OPT=${MLIR_OPT_BIN}
        -DFILECHECK=${FILECHECK_BIN}
        -DINPUT=${REPO_ROOT}/mlir/${prototype}.mlir
        -DOUTPUT=${CMAKE_CURRENT_BINARY_DIR}/${prototype}.verified.mlir
        -P ${CMAKE_CURRENT_SOURCE_DIR}/test/RunGenericMLIRPrototypeTest.cmake
    )
  endforeach()
else()
  message(STATUS
    "mlir-opt or FileCheck not found; generic existing-dialect prototype "
    "tests will not be added."
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

# ---------------------------------------------------------------------------
# CTest: TargetProfileValidationTest
# Pure-Python: validates all 5 deployment-family profile JSON files.
# Checks: JSON validity, required fields, no unknown cost fields, and
# representative_devices metadata is present (ignored safely by the parser).
# Does not require mlir-opt or the plugin.
# ---------------------------------------------------------------------------
find_program(PYTHON3_BIN
  NAMES python3 python
  DOC "Path to Python 3 (required for TargetProfileValidationTest)"
)

if(PYTHON3_BIN)
  add_test(NAME TargetProfileValidationTest
    COMMAND "${PYTHON3_BIN}"
      "${CMAKE_CURRENT_SOURCE_DIR}/test/serving/validate_target_profiles.py"
  )
endif()

# ---------------------------------------------------------------------------
# CTest: NvidiaProfileLoweringTest
# Runs compile-for-target with nvidia_datacenter_gpu.json and verifies that
# backendCapabilities are lowered to target.backend_capabilities.* attrs in
# the annotated MLIR dump.  Also confirms unknown cost fields are absent.
# ---------------------------------------------------------------------------
set(_NVIDA_INT_DIR  "${CMAKE_CURRENT_BINARY_DIR}/nvidia_profile_lowering")
set(_NVIDIA_JSON    "${_NVIDA_INT_DIR}/nvidia_serving_plan.json")
set(_NVIDIA_MLIR    "${_NVIDA_INT_DIR}/nvidia_annotated.mlir")

add_test(NAME NvidiaProfileLoweringTest
  COMMAND ${CMAKE_COMMAND}
    -DTOOL=$<TARGET_FILE:compile-for-target>
    -DPROFILE=${REPO_ROOT}/configs/target_profiles/nvidia_datacenter_gpu.json
    -DMLIR=${REPO_ROOT}/mlir/tiny_gpt_serving.mlir
    -DOUT=${_NVIDIA_JSON}
    -DDUMP_MLIR=${_NVIDIA_MLIR}
    -P ${CMAKE_CURRENT_SOURCE_DIR}/test/serving/RunNvidiaProfileLoweringTest.cmake
)
set_tests_properties(NvidiaProfileLoweringTest PROPERTIES
  DEPENDS compile-for-target
)

# ---------------------------------------------------------------------------
# CTest: DispatchUnitBuilderTest (Phase 26)
# Verifies dispatch-unit materialization in ExecutionPlanBuilder for CV
# full-graph functions carrying the emitter source-provenance contract:
# grouping, op classification totality, tensor binding ABI, corrected
# memory metrics, and backend/kernel truth. Pure C++ + MLIR parser; no
# passes, no exporter, no JSON.
# ---------------------------------------------------------------------------
add_executable(DispatchUnitBuilderTest
  lib/serving/ExecutionPlanBuilder.cpp
  test/serving/DispatchUnitBuilderTest.cpp
)
llvm_update_compile_flags(DispatchUnitBuilderTest)
target_include_directories(DispatchUnitBuilderTest
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_BINARY_DIR}/include
    ${LLVM_INCLUDE_DIRS}
    ${MLIR_INCLUDE_DIRS}
)
target_compile_definitions(DispatchUnitBuilderTest
  PRIVATE ${LLVM_DEFINITIONS}
)
target_link_libraries(DispatchUnitBuilderTest
  PRIVATE
    MLIRIR
    MLIRParser
    MLIRFuncDialect
    MLIRTensorDialect
    MLIRArithDialect
    MLIRSupport
)
add_test(NAME DispatchUnitBuilderTest
         COMMAND DispatchUnitBuilderTest)

add_test(NAME TritonShadowCandidateProviderTest
         COMMAND python3
                 ${CMAKE_CURRENT_SOURCE_DIR}/../tests/test_triton_shadow_candidate_provider.py)
set_tests_properties(TritonShadowCandidateProviderTest PROPERTIES
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/..
)
