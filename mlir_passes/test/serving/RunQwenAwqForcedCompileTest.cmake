# RunQwenAwqForcedCompileTest.cmake
# End-to-end integration test: qwen-to-serving-mlir -> compile-for-target with
# the experimental forced-AWQ GTX 1650 profile (Phase C, minimal, AWQ only).
#
# Variables passed from CMakeLists.txt:
#   QWEN_TOOL — path to qwen-to-serving-mlir executable
#   CFT_TOOL  — path to compile-for-target executable
#   SPEC      — path to configs/models/qwen_0_5b_spec.json
#   PROFILE   — path to configs/target_profiles/nvidia_gtx1650_maxq_awq_forced.json
#   MLIR_OUT  — intermediate MLIR path (written to build dir)
#   JSON_OUT  — execution plan JSON path (written to build dir)
#
# Assertions:
#   - execution_plan.json exists, schema == "execution_plan", schema_version == "2.0.0"
#   - global_decisions.quantization.strategy == "weight_only_int4"
#   - global_decisions.quantization.algorithm == "awq"
#   - global_decisions.quantization.quantized_model_artifact_ref == "artifacts/qwen_awq"
#   - global_decisions.quantization.truth_boundary ==
#       "experimental_forced_quant_not_native_int4_support_on_gtx1650"
#   - no measured fields present (same invariant as every other compiler plan)

# ---------------------------------------------------------------------------
# Step 1: Generate serving-aware MLIR from Qwen model spec.
# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
# Step 2: Compile through the serving pipeline with the forced-AWQ profile.
# ---------------------------------------------------------------------------
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
  message(FATAL_ERROR "execution plan JSON not found: ${JSON_OUT}")
endif()

# ---------------------------------------------------------------------------
# Assertion helpers
# ---------------------------------------------------------------------------

macro(assert_contains _file _needle)
  file(READ "${_file}" _contents)
  string(FIND "${_contents}" "${_needle}" _pos)
  if(_pos EQUAL -1)
    message(FATAL_ERROR
      "expected '${_needle}' in ${_file}\n"
      "actual contents:\n${_contents}")
  endif()
endmacro()

macro(assert_not_contains _file _needle)
  file(READ "${_file}" _contents)
  string(FIND "${_contents}" "${_needle}" _pos)
  if(NOT _pos EQUAL -1)
    message(FATAL_ERROR
      "found forbidden field '${_needle}' in ${_file} — compiler plans must not contain measured runtime fields")
  endif()
endmacro()

# ---------------------------------------------------------------------------
# Assertions: schema identity
# ---------------------------------------------------------------------------
assert_contains("${JSON_OUT}" "\"schema\": \"execution_plan\"")
assert_contains("${JSON_OUT}" "\"schema_version\": \"2.0.0\"")

# ---------------------------------------------------------------------------
# Assertions: forced-AWQ profile reflected in provenance
# ---------------------------------------------------------------------------
assert_contains("${JSON_OUT}" "\"hardware_profile_ref\": \"nvidia-gtx1650-maxq-awq-forced-experimental\"")

# ---------------------------------------------------------------------------
# Assertions: global quantization decision carries the forced AWQ override
# ---------------------------------------------------------------------------
assert_contains("${JSON_OUT}" "\"strategy\": \"weight_only_int4\"")
assert_contains("${JSON_OUT}" "\"algorithm\": \"awq\"")
assert_contains("${JSON_OUT}" "\"quantized_model_artifact_ref\": \"artifacts/qwen_awq\"")
assert_contains("${JSON_OUT}" "\"truth_boundary\": \"experimental_forced_quant_not_native_int4_support_on_gtx1650\"")

# ---------------------------------------------------------------------------
# Assertions: Qwen model identity
# ---------------------------------------------------------------------------
assert_contains("${JSON_OUT}" "\"model_id\": \"qwen2.5-0.5b\"")

# ---------------------------------------------------------------------------
# Assertions: plan structure produced by the serving pipeline
# ---------------------------------------------------------------------------
assert_contains("${JSON_OUT}" "\"function_plans\"")
assert_contains("${JSON_OUT}" "\"global_decisions\"")
assert_contains("${JSON_OUT}" "\"kv_cache_layout\"")

# ---------------------------------------------------------------------------
# Assertions: no measured runtime fields — a forced global quantization
# override does not relax this invariant.
# ---------------------------------------------------------------------------
assert_not_contains("${JSON_OUT}" "\"measured_latency_ms\"")
assert_not_contains("${JSON_OUT}" "\"actual_latency_ms\"")
assert_not_contains("${JSON_OUT}" "\"measured_speedup\"")
assert_not_contains("${JSON_OUT}" "\"performance_claim\"")
assert_not_contains("${JSON_OUT}" "\"evidence_type\"")

message(STATUS "QwenAwqForcedCompileTest: PASS")
message(STATUS "  output: ${JSON_OUT}")
message(STATUS "  schema: execution_plan / 2.0.0 / strategy=weight_only_int4 / algorithm=awq / no measured fields")
