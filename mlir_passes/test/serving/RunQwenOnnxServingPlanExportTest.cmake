# RunQwenOnnxServingPlanExportTest.cmake
#
# Full ONNX-graph-facts pipeline, end to end:
#   qwen-onnx-to-serving-mlir (raw, full per-layer expansion)
#   -> mlir-opt --pass-pipeline='builtin.module(llm-frontend-normalization)'
#   -> compile-for-target
#   -> execution_plan.json
#
# Plus validation that this is a genuinely full-expanded plan, not a single
# hand-templated block: distinct serving.layer_index values found in the
# normalized MLIR must equal the declared llm.num_layers, and the exported
# plan's per_op_decisions must cover more than the single rmsnorm-only entry
# the legacy ModelSpec path produces.
#
# Required variables (passed via -D):
#   ONNX_TOOL  -- path to qwen-onnx-to-serving-mlir
#   MLIR_OPT   -- path to mlir-opt
#   PLUGIN     -- path to HIRMatMulBiasReluFusionPass plugin (.dylib / .so)
#   CFT_TOOL   -- path to compile-for-target
#   FACTS      -- path to configs/models/qwen_0_5b_onnx_graph_facts.json
#   PROFILE    -- path to a target profile JSON
#   RAW_OUT    -- output path for raw (pre-normalization) MLIR
#   NORM_OUT   -- output path for normalized MLIR
#   JSON_OUT   -- output path for execution_plan.json

cmake_minimum_required(VERSION 3.20)

foreach(var ONNX_TOOL MLIR_OPT PLUGIN CFT_TOOL FACTS PROFILE RAW_OUT NORM_OUT JSON_OUT)
  if(NOT DEFINED ${var})
    message(FATAL_ERROR "Missing required variable: ${var}")
  endif()
endforeach()

# ---- Step 1: import ONNX-shaped graph facts -> raw serving MLIR ----
get_filename_component(RAW_DIR "${RAW_OUT}" DIRECTORY)
file(MAKE_DIRECTORY "${RAW_DIR}")

execute_process(
  COMMAND "${ONNX_TOOL}" --graph-facts "${FACTS}" --out "${RAW_OUT}"
  RESULT_VARIABLE rc
  OUTPUT_VARIABLE stdout
  ERROR_VARIABLE  stderr
)
if(NOT rc EQUAL 0)
  message(FATAL_ERROR
    "[QwenOnnxServingPlanExportTest] qwen-onnx-to-serving-mlir failed (rc=${rc}):\n${stderr}")
endif()

# ---- Step 2: normalize raw attention pattern -> canonical IR ----
get_filename_component(NORM_DIR "${NORM_OUT}" DIRECTORY)
file(MAKE_DIRECTORY "${NORM_DIR}")

execute_process(
  COMMAND "${MLIR_OPT}"
    --allow-unregistered-dialect
    "--load-pass-plugin=${PLUGIN}"
    "--pass-pipeline=builtin.module(llm-frontend-normalization)"
    "${RAW_OUT}"
    -o "${NORM_OUT}"
  RESULT_VARIABLE rc
  OUTPUT_VARIABLE stdout
  ERROR_VARIABLE  stderr
)
if(NOT rc EQUAL 0)
  message(FATAL_ERROR
    "[QwenOnnxServingPlanExportTest] mlir-opt normalization failed (rc=${rc}):\n${stderr}")
endif()

# ---- Step 3: validate full per-layer expansion survived normalization ----
file(READ "${NORM_OUT}" norm_content)

string(REGEX MATCHALL "serving\\.layer_index = [0-9]+" _layer_index_matches "${norm_content}")
set(_layer_indices "")
foreach(_match ${_layer_index_matches})
  string(REGEX REPLACE "serving\\.layer_index = " "" _num "${_match}")
  list(APPEND _layer_indices "${_num}")
endforeach()
list(REMOVE_DUPLICATES _layer_indices)
list(LENGTH _layer_indices _distinct_layer_count)

if(NOT _distinct_layer_count EQUAL 24)
  message(FATAL_ERROR
    "[QwenOnnxServingPlanExportTest] expected 24 distinct serving.layer_index "
    "values (matching llm.num_layers) in normalized MLIR, found ${_distinct_layer_count}: "
    "${_layer_indices}")
endif()

foreach(_required "0" "23")
  list(FIND _layer_indices "${_required}" _pos)
  if(_pos EQUAL -1)
    message(FATAL_ERROR
      "[QwenOnnxServingPlanExportTest] serving.layer_index = ${_required} not found in normalized MLIR")
  endif()
endforeach()

# Normalization must have consumed the raw attention pattern.
foreach(_forbidden "llm.attention_scores" "llm.softmax" "llm.kv_cache_write" "llm.kv_cache_read")
  string(FIND "${norm_content}" "${_forbidden}" _pos)
  if(NOT _pos EQUAL -1)
    message(FATAL_ERROR
      "[QwenOnnxServingPlanExportTest] normalized MLIR still contains '${_forbidden}'; "
      "llm-frontend-normalization did not fully consume the raw attention pattern")
  endif()
endforeach()

foreach(_required "llm.attention_prefill" "llm.attention_decode" "llm.q_proj" "llm.o_proj" "llm.mlp")
  string(FIND "${norm_content}" "${_required}" _pos)
  if(_pos EQUAL -1)
    message(FATAL_ERROR
      "[QwenOnnxServingPlanExportTest] normalized MLIR missing expected op '${_required}'")
  endif()
endforeach()

# ---- Step 4: run serving planning pipeline on normalized MLIR ----
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
    "[QwenOnnxServingPlanExportTest] compile-for-target failed (rc=${rc}):\n${stderr}")
endif()

# ---- Step 5: verify the exported plan and its per-op decision coverage ----
if(NOT EXISTS "${JSON_OUT}")
  message(FATAL_ERROR "[QwenOnnxServingPlanExportTest] execution plan not found: ${JSON_OUT}")
endif()

file(READ "${JSON_OUT}" json_content)

foreach(_field "\"schema\"" "\"execution_plan\"" "prefill" "decode")
  string(FIND "${json_content}" "${_field}" _pos)
  if(_pos EQUAL -1)
    message(FATAL_ERROR
      "[QwenOnnxServingPlanExportTest] execution_plan.json missing expected field: ${_field}")
  endif()
endforeach()

# The legacy ModelSpec path exports exactly one per_op_decisions entry per
# phase (op_1, always llm.rmsnorm). A genuinely full-expanded, op-coverage-
# extended plan must export more than one op_N entry per phase — this is the
# concrete signal that real per-layer fidelity (not a single hand-templated
# block) reached the exported artifact.
string(REGEX MATCHALL "\"op_name\": \"op_[0-9]+\"" _op_name_matches "${json_content}")
list(LENGTH _op_name_matches _op_name_count)
if(_op_name_count LESS_EQUAL 2)
  message(FATAL_ERROR
    "[QwenOnnxServingPlanExportTest] expected more than 2 total per_op_decisions "
    "entries across both phases (found ${_op_name_count}); full per-layer "
    "expansion and op-coverage extension should produce many more than the "
    "legacy ModelSpec path's one-rmsnorm-per-phase output")
endif()

message(STATUS
  "[QwenOnnxServingPlanExportTest] passed — ${_distinct_layer_count} distinct "
  "layers, ${_op_name_count} per_op_decisions entries")
