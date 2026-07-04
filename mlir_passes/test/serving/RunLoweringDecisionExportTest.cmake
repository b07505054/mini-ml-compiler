# RunLoweringDecisionExportTest.cmake
# CTest driver for Commit K: verifies per_op_lowering_decisions appear in the
# exported serving execution plan JSON.
#
# Achievable through compile-for-target (4 of 5 decision values):
#   direct_lower, rewrite_then_lower, fallback_backend, unsupported.
#
# dequant_then_lower is covered by ServingExecutionPlanBuilderTest (pre-annotated
# MLIR, no pipeline; required because weight_only_int8 keeps activations fp16).
#
# Variables passed in from CMakeLists.txt:
#   TOOL     -- path to compile-for-target executable
#   PROFILE  -- path to lowering_decision_export_test.json
#   MLIR     -- path to lowering_decision_export_test.mlir
#   OUT      -- canonical output path (written to build dir)

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

# Artifact structure.
assert_contains("${OUT}" "\"artifact_type\": \"serving_execution_plan\"")
assert_contains("${OUT}" "\"schema_version\": \"1.0.0\"")

# per_op_lowering_decisions array present.
assert_contains("${OUT}" "\"per_op_lowering_decisions\"")

# All 4 pipeline-achievable decision values.
assert_contains("${OUT}" "\"lowering_decision\": \"direct_lower\"")
assert_contains("${OUT}" "\"lowering_decision\": \"rewrite_then_lower\"")
assert_contains("${OUT}" "\"lowering_decision\": \"fallback_backend\"")
assert_contains("${OUT}" "\"lowering_decision\": \"unsupported\"")

# Required rewrite value for rewrite_then_lower case.
assert_contains("${OUT}" "\"required_rewrite\": \"fp16_to_fp32_cast\"")

# Op types covered.
assert_contains("${OUT}" "\"op_type\": \"hir.matmul\"")
assert_contains("${OUT}" "\"op_type\": \"hir.conv2d\"")
assert_contains("${OUT}" "\"op_type\": \"hir.softmax\"")

# Kernel provenance fields present.
assert_contains("${OUT}" "\"kernel_exists\"")
assert_contains("${OUT}" "\"kernel_lowering_status\"")
assert_contains("${OUT}" "\"target_backend\"")
assert_contains("${OUT}" "\"target_kernel_library\"")
assert_contains("${OUT}" "\"target_kernel\"")

# Truth boundary (from LoweringDecisionPlanningPass).
assert_contains("${OUT}" "lowering_decision_static_not_backend_codegen_verified")

# Pass tracked in source_passes.
assert_contains("${OUT}" "\"lowering-decision-planning\"")

message(STATUS "LoweringDecisionExportTest: PASS")
