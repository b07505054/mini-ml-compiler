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
  lib/ServingPhaseAnalysisPass.cpp
  lib/serving/KVLayoutPlanningPass.cpp
  lib/serving/ReplayEligibilityPass.cpp
  lib/serving/ServingExecutionPlanBuilder.cpp
  lib/serving/ServingPipeline.cpp
  lib/serving/TargetConstraints.cpp
  lib/serving/ExecutionProviderPlanningPass.cpp
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
