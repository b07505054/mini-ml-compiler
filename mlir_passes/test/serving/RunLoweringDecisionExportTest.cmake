# RunLoweringDecisionExportTest.cmake
# CTest driver: verifies per-op kernel decisions appear in the exported
# ExecutionPlanV2 JSON when LoweringDecisionPlanningPass has run.
#
# Pipeline-achievable lowering_path values (4 of 5):
#   direct_lower, rewrite_then_lower, fallback_backend, unsupported.
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

# V2 schema identity.
assert_contains("${OUT}" "\"schema\": \"execution_plan\"")
assert_contains("${OUT}" "\"schema_version\": \"2.0.0\"")

# per_op_decisions array present.
assert_contains("${OUT}" "\"per_op_decisions\"")

# All 4 pipeline-achievable lowering_path values (V2 field name).
assert_contains("${OUT}" "\"lowering_path\": \"direct_lower\"")
assert_contains("${OUT}" "\"lowering_path\": \"rewrite_then_lower\"")
assert_contains("${OUT}" "\"lowering_path\": \"fallback_backend\"")
assert_contains("${OUT}" "\"lowering_path\": \"unsupported\"")

# Op types covered.
assert_contains("${OUT}" "\"op_type\": \"hir.matmul\"")
assert_contains("${OUT}" "\"op_type\": \"hir.conv2d\"")
assert_contains("${OUT}" "\"op_type\": \"hir.softmax\"")

# Kernel provenance fields (V2 names).
assert_contains("${OUT}" "\"kernel_exists\"")
assert_contains("${OUT}" "\"selected_kernel\"")
assert_contains("${OUT}" "\"kernel_library\"")
# fallback_backend field present in FallbackDecision for fallback_backend ops.
assert_contains("${OUT}" "\"fallback_backend\"")

# Truth boundary (from LoweringDecisionPlanningPass).
assert_contains("${OUT}" "lowering_decision_static_not_backend_codegen_verified")

# Pass name tracked in per-op decision source_pass.
assert_contains("${OUT}" "\"lowering-decision-planning\"")

message(STATUS "LoweringDecisionExportTest: PASS")
