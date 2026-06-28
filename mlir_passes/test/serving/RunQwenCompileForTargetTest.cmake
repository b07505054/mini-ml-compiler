# RunQwenCompileForTargetTest.cmake
# End-to-end integration test: qwen-to-serving-mlir -> compile-for-target.
# Invoked by CTest via add_test(...COMMAND cmake -P ...).
# Variables passed from CMakeLists.txt:
#   QWEN_TOOL — path to qwen-to-serving-mlir executable
#   CFT_TOOL  — path to compile-for-target executable
#   SPEC      — path to configs/models/qwen_0_5b_spec.json
#   PROFILE   — path to configs/target_profiles/apple_a17pro_mobile.json
#   MLIR_OUT  — intermediate MLIR path (written to build dir)
#   JSON_OUT  — canonical serving plan JSON path (written to build dir)

# Step 1: Generate serving-aware MLIR from Qwen model spec.
execute_process(
  COMMAND "${QWEN_TOOL}" --model-spec "${SPEC}" --out "${MLIR_OUT}"
  RESULT_VARIABLE _rc1
  OUTPUT_VARIABLE _stdout1
  ERROR_VARIABLE  _stderr1
)

if(NOT _rc1 EQUAL 0)
  message(FATAL_ERROR
    "qwen-to-serving-mlir failed (exit ${_rc1})\n"
    "stdout:\n${_stdout1}\n"
    "stderr:\n${_stderr1}")
endif()

if(NOT EXISTS "${MLIR_OUT}")
  message(FATAL_ERROR "MLIR output not found: ${MLIR_OUT}")
endif()

# Step 2: Compile generated MLIR through the serving pass pipeline.
execute_process(
  COMMAND "${CFT_TOOL}"
    --device-profile "${PROFILE}"
    --mlir           "${MLIR_OUT}"
    --out            "${JSON_OUT}"
  RESULT_VARIABLE _rc2
  OUTPUT_VARIABLE _stdout2
  ERROR_VARIABLE  _stderr2
)

if(NOT _rc2 EQUAL 0)
  message(FATAL_ERROR
    "compile-for-target failed (exit ${_rc2})\n"
    "stdout:\n${_stdout2}\n"
    "stderr:\n${_stderr2}")
endif()

if(NOT EXISTS "${JSON_OUT}")
  message(FATAL_ERROR "JSON output not found: ${JSON_OUT}")
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

# --- Canonical artifact structural checks ---
assert_contains("${JSON_OUT}" "\"artifact_type\": \"serving_execution_plan\"")
assert_contains("${JSON_OUT}" "\"schema_version\": \"1.0.0\"")
assert_contains("${JSON_OUT}" "\"target_profile_id\": \"apple-a17pro-mobile\"")

# --- Qwen model identity: proves this is not the tiny-gpt fixture ---
assert_contains("${JSON_OUT}" "\"model_name\": \"qwen2.5-0.5b\"")

# --- Plan structure ---
assert_contains("${JSON_OUT}" "\"function_plans\"")
assert_contains("${JSON_OUT}" "\"backend_execution_plan\"")
assert_contains("${JSON_OUT}" "\"kv_plan\"")
assert_contains("${JSON_OUT}" "\"replay_plan\"")
assert_contains("${JSON_OUT}" "\"source_passes\"")

# --- GQA metadata in artifact ---
assert_contains("${JSON_OUT}" "\"num_kv_heads\": 2")

message(STATUS "QwenCompileForTargetTest: PASS")
