# RunQwenToServingMlirTest.cmake
# Invoked by CTest via add_test(...COMMAND cmake -P ...).
# Variables passed from CMakeLists.txt:
#   TOOL — path to qwen-to-serving-mlir executable
#   SPEC — path to configs/models/qwen_0_5b_spec.json
#   OUT  — output MLIR path (written to build dir)

execute_process(
  COMMAND "${TOOL}" --model-spec "${SPEC}" --out "${OUT}"
  RESULT_VARIABLE _rc
  OUTPUT_VARIABLE _stdout
  ERROR_VARIABLE  _stderr
)

if(NOT _rc EQUAL 0)
  message(FATAL_ERROR
    "qwen-to-serving-mlir exited with code ${_rc}\n"
    "stdout:\n${_stdout}\n"
    "stderr:\n${_stderr}")
endif()

if(NOT EXISTS "${OUT}")
  message(FATAL_ERROR "output MLIR not found: ${OUT}")
endif()

macro(assert_contains _file _needle)
  file(READ "${_file}" _contents)
  string(FIND "${_contents}" "${_needle}" _pos)
  if(_pos EQUAL -1)
    message(FATAL_ERROR
      "expected to find '${_needle}' in ${_file}\n"
      "actual contents:\n${_contents}")
  endif()
endmacro()

# --- Module attribute checks ---
assert_contains("${OUT}" "llm.model = \"qwen2.5-0.5b\"")
assert_contains("${OUT}" "llm.num_layers = 24")
assert_contains("${OUT}" "llm.hidden_size = 896")
assert_contains("${OUT}" "llm.num_attention_heads = 14")
assert_contains("${OUT}" "llm.num_key_value_heads = 2")
assert_contains("${OUT}" "llm.frontend_source = \"qwen_model_spec\"")
assert_contains("${OUT}" "llm.truth_boundary = \"declared_model_config_not_full_graph_import\"")

# --- Serving op checks ---
assert_contains("${OUT}" "llm.attention_prefill")
assert_contains("${OUT}" "llm.attention_decode")
assert_contains("${OUT}" "kv_cache.role = \"producer\"")
assert_contains("${OUT}" "kv_cache.role = \"consumer\"")
assert_contains("${OUT}" "serving.phase = \"prefill\"")
assert_contains("${OUT}" "serving.phase = \"decode\"")
assert_contains("${OUT}" "serving.prompt_tokens")
assert_contains("${OUT}" "serving.output_tokens")

# --- GQA KV projection dimension (kv_proj_dim = 2 * 64 = 128) ---
assert_contains("${OUT}" "tensor<?x128xf16>")
assert_contains("${OUT}" "tensor<1x128xf16>")

message(STATUS "QwenToServingMlirTest: PASS")
