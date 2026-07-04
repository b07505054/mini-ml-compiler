# RunQuantStrategyExportTest.cmake
# CTest driver for Commit H: verifies per_op_quantization_decisions appear in
# the exported serving execution plan JSON.
#
# Variables passed in from CMakeLists.txt:
#   TOOL     — path to compile-for-target executable
#   PROFILE  — path to apple_a17pro_mobile.json
#   MLIR     — path to quant_strategy_serving.mlir
#   OUT      — canonical output path (written to build dir)

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

if(NOT EXISTS "${OUT}")
  message(FATAL_ERROR "canonical artifact not found: ${OUT}")
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

# Artifact structure checks.
assert_contains("${OUT}" "\"artifact_type\": \"serving_execution_plan\"")
assert_contains("${OUT}" "\"schema_version\": \"1.0.0\"")

# per_op_quantization_decisions array present.
assert_contains("${OUT}" "\"per_op_quantization_decisions\"")

# weight_only_int8 strategy from hir.matmul.
assert_contains("${OUT}" "\"strategy\": \"weight_only_int8\"")
assert_contains("${OUT}" "\"weight_dtype\": \"int8\"")
assert_contains("${OUT}" "\"activation_dtype\": \"fp16\"")

# fp16_fallback strategy from hir.softmax (accuracy-sensitive).
assert_contains("${OUT}" "\"strategy\": \"fp16_fallback\"")
assert_contains("${OUT}" "\"accuracy_risk\": \"medium\"")
assert_contains("${OUT}" "\"fallback_reason\": \"accuracy_sensitive_op\"")

# Op types appear in decisions.
assert_contains("${OUT}" "\"op_type\": \"hir.matmul\"")
assert_contains("${OUT}" "\"op_type\": \"hir.softmax\"")

# Truth boundary from QuantizationStrategyPlanningPass.
assert_contains("${OUT}" "quantization_strategy_static_not_accuracy_calibrated")

# Pass tracked in source_passes.
assert_contains("${OUT}" "\"quantization-strategy-planning\"")

message(STATUS "QuantStrategyExportTest: PASS")
