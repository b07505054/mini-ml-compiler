# RunCompileForTargetTest.cmake
# Invoked by CTest via add_test(...COMMAND cmake -P ...).
# Variables passed in from CMakeLists.txt:
#   TOOL     — path to compile-for-target executable
#   PROFILE  — path to apple_a17pro_mobile.json
#   MLIR     — path to tiny_gpt_serving.mlir
#   OUT      — canonical output path (written to build dir)
#   SUMMARY  — expected summary path (sibling of OUT)

# Run the tool.
execute_process(
  COMMAND "${TOOL}"
    --device-profile "${PROFILE}"
    --mlir           "${MLIR}"
    --out            "${OUT}"
  RESULT_VARIABLE _rc
  OUTPUT_VARIABLE _stdout
  ERROR_VARIABLE  _stderr
)

if(NOT _rc EQUAL 0)
  message(FATAL_ERROR
    "compile-for-target exited with code ${_rc}\n"
    "stdout:\n${_stdout}\n"
    "stderr:\n${_stderr}")
endif()

# Verify canonical artifact exists.
if(NOT EXISTS "${OUT}")
  message(FATAL_ERROR "canonical artifact not found: ${OUT}")
endif()

# Verify summary artifact exists (same dir as OUT).
if(NOT EXISTS "${SUMMARY}")
  message(FATAL_ERROR "summary artifact not found: ${SUMMARY}")
endif()

# Helper: grep a file for a literal string.
macro(assert_contains _file _needle)
  file(READ "${_file}" _contents)
  string(FIND "${_contents}" "${_needle}" _pos)
  if(_pos EQUAL -1)
    message(FATAL_ERROR
      "expected to find '${_needle}' in ${_file}\n"
      "actual contents:\n${_contents}")
  endif()
endmacro()

# --- Canonical artifact checks ---
assert_contains("${OUT}" "\"artifact_type\": \"serving_execution_plan\"")
assert_contains("${OUT}" "\"schema_version\": \"1.0.0\"")
assert_contains("${OUT}" "\"target_profile_id\": \"apple-a17pro-mobile\"")
assert_contains("${OUT}" "\"primary_backend\": \"coreml\"")
assert_contains("${OUT}" "\"decision_source\": \"target_preferred\"")
assert_contains("${OUT}" "\"kv_plan\"")
assert_contains("${OUT}" "\"replay_plan\"")
assert_contains("${OUT}" "\"source_passes\"")
# Profile provides cost attrs → cost_source must be target_profile_formula_estimate.
assert_contains("${OUT}" "\"cost_source\": \"target_profile_formula_estimate\"")
# Quantization plan is now included in every function plan.
assert_contains("${OUT}" "\"quantization_plan\"")
assert_contains("${OUT}" "\"plan_dtype\"")
assert_contains("${OUT}" "\"dtype_bytes\"")
assert_contains("${OUT}" "precision_selection_from_target_profile_not_calibrated")

# --- Summary artifact checks ---
assert_contains("${SUMMARY}" "\"artifact_type\": \"serving_execution_plan_summary\"")
assert_contains("${SUMMARY}" "\"schema_version\": \"1.0.0\"")
assert_contains("${SUMMARY}" "\"NOT_A_RUNTIME_INPUT\"")
assert_contains("${SUMMARY}" "\"target_profile_id\": \"apple-a17pro-mobile\"")

message(STATUS "CompileForTargetTest: PASS")
