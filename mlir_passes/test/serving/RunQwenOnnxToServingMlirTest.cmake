# RunQwenOnnxToServingMlirTest.cmake
# Invoked by CTest via add_test(...COMMAND cmake -P ...).
# Variables passed from CMakeLists.txt:
#   TOOL  — path to qwen-onnx-to-serving-mlir executable
#   FACTS — path to configs/models/qwen_0_5b_onnx_graph_facts.json
#   OUT   — output MLIR path (written to build dir)

execute_process(
  COMMAND "${TOOL}" --graph-facts "${FACTS}" --out "${OUT}"
  RESULT_VARIABLE _rc
  OUTPUT_VARIABLE _stdout
  ERROR_VARIABLE  _stderr
)

if(NOT _rc EQUAL 0)
  message(FATAL_ERROR
    "qwen-onnx-to-serving-mlir exited with code ${_rc}\n"
    "stdout:\n${_stdout}\n"
    "stderr:\n${_stderr}")
endif()

if(NOT EXISTS "${OUT}")
  message(FATAL_ERROR "output MLIR not found: ${OUT}")
endif()

file(READ "${OUT}" _contents)

macro(assert_contains _needle)
  string(FIND "${_contents}" "${_needle}" _pos)
  if(_pos EQUAL -1)
    message(FATAL_ERROR "expected to find '${_needle}' in ${OUT}")
  endif()
endmacro()

macro(assert_not_contains _needle)
  string(FIND "${_contents}" "${_needle}" _pos)
  if(NOT _pos EQUAL -1)
    message(FATAL_ERROR "expected NOT to find '${_needle}' in ${OUT}")
  endif()
endmacro()

# --- Module attribute checks ---
assert_contains("llm.model = \"qwen2.5-0.5b\"")
assert_contains("llm.num_layers = 24")
assert_contains("llm.hidden_size = 896")
assert_contains("llm.frontend_source = \"qwen_onnx_graph_import\"")
assert_contains("llm.truth_boundary = \"onnx_shaped_fixture_not_real_onnx_protobuf_import\"")

# --- Full per-layer expansion: layer_index 0 and 23 must both be present,
#     proving this is not a single hand-templated block. ---
assert_contains("serving.layer_index = 0 : i64")
assert_contains("serving.layer_index = 23 : i64")
assert_not_contains("serving.layer_index = 24 : i64")

# --- Layer role coverage ---
assert_contains("serving.layer_role = \"decoder_layer\"")
assert_contains("serving.layer_role = \"embedding\"")
assert_contains("serving.layer_role = \"final_norm\"")
assert_contains("serving.layer_role = \"lm_head\"")

# --- Raw (pre-normalization) attention pattern present per layer ---
assert_contains("llm.q_proj")
assert_contains("llm.k_proj")
assert_contains("llm.v_proj")
assert_contains("llm.attention_scores")
assert_contains("llm.attention_output")
assert_contains("llm.kv_cache_write")
assert_contains("llm.kv_cache_read")

# --- Non-attention per-layer ops that must survive normalization untouched ---
assert_contains("llm.o_proj")
assert_contains("llm.mlp")
assert_contains("llm.rmsnorm")
assert_contains("llm.lm_head_proj")

# --- Two phases, both present ---
assert_contains("func.func @qwen_prefill")
assert_contains("func.func @qwen_decode")

message(STATUS "QwenOnnxToServingMlirTest: PASS")
