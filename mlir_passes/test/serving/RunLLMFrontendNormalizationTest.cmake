# Integration test: LLMFrontendNormalizationPass -> compile-for-target -> JSON check
#
# Steps:
#   1. mlir-opt: run llm-frontend-normalization on raw input MLIR
#   2. compile-for-target: run serving-optimization-pipeline -> JSON plan
#   3. grep: check JSON for canonical serving phase fields
#
# Required variables (passed via -D):
#   MLIR_OPT    -- path to mlir-opt
#   PLUGIN      -- path to HIRMatMulBiasReluFusionPass plugin (.dylib / .so)
#   CFT_TOOL    -- path to compile-for-target
#   RAW_MLIR    -- path to raw_qwen_frontend_input.mlir
#   PROFILE     -- path to apple_a17pro_mobile.json target profile
#   NORM_OUT    -- output path for normalized MLIR
#   JSON_OUT    -- output path for serving plan JSON

cmake_minimum_required(VERSION 3.20)

foreach(var MLIR_OPT PLUGIN CFT_TOOL RAW_MLIR PROFILE NORM_OUT JSON_OUT)
  if(NOT DEFINED ${var})
    message(FATAL_ERROR "Missing required variable: ${var}")
  endif()
endforeach()

# ---- Step 1: normalize raw LLM graph to canonical IR ----
get_filename_component(NORM_DIR "${NORM_OUT}" DIRECTORY)
file(MAKE_DIRECTORY "${NORM_DIR}")

execute_process(
  COMMAND "${MLIR_OPT}"
    --allow-unregistered-dialect
    "--load-pass-plugin=${PLUGIN}"
    "--pass-pipeline=builtin.module(llm-frontend-normalization)"
    "${RAW_MLIR}"
    -o "${NORM_OUT}"
  RESULT_VARIABLE rc
  OUTPUT_VARIABLE stdout
  ERROR_VARIABLE  stderr
)
if(NOT rc EQUAL 0)
  message(FATAL_ERROR
    "[LLMFrontendNormalizationTest] mlir-opt normalization failed (rc=${rc}):\n${stderr}")
endif()

# ---- Step 2: run serving planning pipeline on normalized MLIR ----
get_filename_component(JSON_DIR "${JSON_OUT}" DIRECTORY)
file(MAKE_DIRECTORY "${JSON_DIR}")

execute_process(
  COMMAND "${CFT_TOOL}"
    "--device-profile=${PROFILE}"
    "--mlir=${NORM_OUT}"
    "--out=${JSON_OUT}"
  RESULT_VARIABLE rc
  OUTPUT_VARIABLE stdout
  ERROR_VARIABLE  stderr
)
if(NOT rc EQUAL 0)
  message(FATAL_ERROR
    "[LLMFrontendNormalizationTest] compile-for-target failed (rc=${rc}):\n${stderr}")
endif()

# ---- Step 3: verify JSON plan fields ----
file(READ "${JSON_OUT}" json_content)

foreach(field
    "serving_execution_plan"
    "qwen2.5-0.5b"
    "raw_qwen_prefill_graph"
    "raw_qwen_decode_graph"
    "serving-phase-analysis"
    "prefill"
    "decode"
  )
  string(FIND "${json_content}" "${field}" pos)
  if(pos EQUAL -1)
    message(FATAL_ERROR
      "[LLMFrontendNormalizationTest] JSON missing expected field: '${field}'")
  endif()
endforeach()

message(STATUS "[LLMFrontendNormalizationTest] passed — normalized MLIR -> serving plan OK")
