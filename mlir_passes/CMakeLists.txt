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

set(LLVM_TARGET_DEFINITIONS include/FusionPasses.td)
mlir_tablegen(include/FusionPasses.h.inc -gen-pass-decls)
add_public_tablegen_target(HIRFusionPassIncGen)

file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/include)

add_library(HIRMatMulBiasReluFusionPass MODULE
  lib/MatMulBiasReluFusionPass.cpp
)
llvm_update_compile_flags(HIRMatMulBiasReluFusionPass)

add_dependencies(HIRMatMulBiasReluFusionPass
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
