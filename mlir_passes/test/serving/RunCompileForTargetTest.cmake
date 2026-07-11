# RunCompileForTargetTest.cmake
# Invoked by CTest via add_test(...COMMAND cmake -P ...).
# Variables passed in from CMakeLists.txt:
#   TOOL     — path to compile-for-target executable
#   PROFILE  — path to apple_a17pro_mobile.json
#   MLIR     — path to tiny_gpt_serving.mlir
#   OUT      — canonical output path (written to build dir)

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

# --- V2 schema identity ---
assert_contains("${OUT}" "\"schema\": \"execution_plan\"")
assert_contains("${OUT}" "\"schema_version\": \"2.0.0\"")

# --- Provenance: profile ID in capability_bundle ---
assert_contains("${OUT}" "\"hardware_profile_ref\": \"apple-a17pro-mobile\"")

# --- Plan structure ---
assert_contains("${OUT}" "\"plan_id\"")
assert_contains("${OUT}" "\"provenance\"")
assert_contains("${OUT}" "\"model_identity\"")
assert_contains("${OUT}" "\"global_decisions\"")
assert_contains("${OUT}" "\"function_plans\"")

# --- Backend decision: V2 field names ---
assert_contains("${OUT}" "\"selected_backend\"")
assert_contains("${OUT}" "\"fallback_backends\"")
# Profile requests coreml; decision_source maps to source_pass.
assert_contains("${OUT}" "\"target_preferred\"")

# --- Memory decision: kv_cache_layout in global_decisions.memory ---
assert_contains("${OUT}" "\"kv_cache_layout\"")

# --- Serving decision: replay_eligible in global_decisions.serving ---
assert_contains("${OUT}" "\"replay_eligible\"")

# --- Per-op decisions array present ---
assert_contains("${OUT}" "\"per_op_decisions\"")

# --- Truth boundary present ---
assert_contains("${OUT}" "\"truth_boundary\"")

# --- Kernel selection contract (kernel_selection_contract_v1) ---
# The a17pro profile declares exactly one runtime kernel (Metal RMSNorm f32);
# the coreml-primary plan must therefore carry kernel_selection objects with
# explicit statuses — never silent, never fabricated coverage.
assert_contains("${OUT}" "\"kernel_selection\"")
assert_contains("${OUT}" "\"contract_version\": \"kernel_selection_contract_v1\"")
assert_contains("${OUT}" "kernel_selection_static_descriptor_match_not_runtime_execution")

message(STATUS "CompileForTargetTest: PASS")
