# RunQuantStrategyExportTest.cmake
# CTest driver: verifies per-op quantization decisions appear in the exported
# ExecutionPlanV2 JSON when QuantizationStrategyPlanningPass has run.
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

# V2 schema identity.
assert_contains("${OUT}" "\"schema\": \"execution_plan\"")
assert_contains("${OUT}" "\"schema_version\": \"2.0.0\"")

# per_op_decisions array present (V2 name for per-op decision bundles).
assert_contains("${OUT}" "\"per_op_decisions\"")

# Slice 1 candidate-owned quantization decision from hir.matmul. The Apple
# profile intentionally lacks the new explicit quantization capability fields,
# so the compiler preserves the FP32 fallback rather than inferring INT8 support.
assert_contains("${OUT}" "\"strategy\": \"fp32_baseline\"")
assert_contains("${OUT}" "\"scheme\": \"fp32_baseline\"")
assert_contains("${OUT}" "\"selected_candidate_id\"")
assert_contains("${OUT}" "\"considered_candidate_ids\"")
assert_contains("${OUT}" "\"rejected_candidate_reasons\"")
assert_contains("${OUT}" "\"weight_dtype\": \"fp32\"")
assert_contains("${OUT}" "\"activation_dtype\": \"fp32\"")

# fp16_fallback strategy from hir.softmax (accuracy-sensitive).
assert_contains("${OUT}" "\"strategy\": \"fp16_fallback\"")
assert_contains("${OUT}" "\"accuracy_risk\": \"medium\"")

# Op types appear in decisions.
assert_contains("${OUT}" "\"op_type\": \"hir.matmul\"")
assert_contains("${OUT}" "\"op_type\": \"hir.softmax\"")

# Truth boundary from QuantizationStrategyPlanningPass.
assert_contains("${OUT}" "quantization_strategy_static_not_accuracy_calibrated")

# Pass name tracked in per-op decision source_pass.
assert_contains("${OUT}" "\"quantization-strategy-planning\"")

message(STATUS "QuantStrategyExportTest: PASS")
