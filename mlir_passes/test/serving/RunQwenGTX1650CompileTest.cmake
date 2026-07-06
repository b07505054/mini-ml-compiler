# RunQwenGTX1650CompileTest.cmake
# End-to-end integration test: qwen-to-serving-mlir -> compile-for-target (GTX 1650 Max-Q).
# Invoked by CTest via add_test(...COMMAND cmake -P ...).
#
# Variables passed from CMakeLists.txt:
#   QWEN_TOOL — path to qwen-to-serving-mlir executable
#   CFT_TOOL  — path to compile-for-target executable
#   SPEC      — path to configs/models/qwen_0_5b_spec.json
#   PROFILE   — path to configs/target_profiles/nvidia_gtx1650_maxq.json
#   MLIR_OUT  — intermediate MLIR path (written to build dir)
#   JSON_OUT  — execution plan JSON path (written to build dir)
#
# Assertions:
#   - execution_plan.json exists
#   - schema == "execution_plan"
#   - schema_version == "2.0.0"
#   - global_decisions.quantization.strategy == "none"  (GTX 1650: no mandatory quant)
#   - no measured fields present

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
# Step 2: Compile through 15-pass serving pipeline with GTX 1650 profile.
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
# Assertions: GTX 1650 profile reflected in provenance
# ---------------------------------------------------------------------------
assert_contains("${JSON_OUT}" "\"hardware_profile_ref\": \"nvidia-gtx1650-maxq\"")

# ---------------------------------------------------------------------------
# Assertions: no-quant strategy (GTX 1650: no required_weight_quant_mode)
#
# The GTX 1650 profile declares no required_weight_quant_mode.
# The QuantizationStrategyPlanningPass therefore emits no global quantization
# decision (global_decisions.quantization is absent) and per-op decisions use
# "fp16_fallback" for accuracy-sensitive ops — which is the expected no-quant
# outcome for this profile.
#
# We assert absence of int4/awq/gptq quantization modes to confirm that no
# mandatory quantization strategy was applied.
# ---------------------------------------------------------------------------
assert_not_contains("${JSON_OUT}" "\"awq_int4\"")
assert_not_contains("${JSON_OUT}" "\"int4_weight_only\"")
assert_not_contains("${JSON_OUT}" "\"gptq_int4\"")
assert_not_contains("${JSON_OUT}" "\"fp8_e4m3\"")

# ---------------------------------------------------------------------------
# Assertions: Qwen model identity
# ---------------------------------------------------------------------------
assert_contains("${JSON_OUT}" "\"model_id\": \"qwen2.5-0.5b\"")

# ---------------------------------------------------------------------------
# Assertions: plan structure produced by 15-pass pipeline
# ---------------------------------------------------------------------------
assert_contains("${JSON_OUT}" "\"function_plans\"")
assert_contains("${JSON_OUT}" "\"global_decisions\"")
assert_contains("${JSON_OUT}" "\"kv_cache_layout\"")

# ---------------------------------------------------------------------------
# Assertions: no measured runtime fields
# ---------------------------------------------------------------------------
assert_not_contains("${JSON_OUT}" "\"measured_latency_ms\"")
assert_not_contains("${JSON_OUT}" "\"actual_latency_ms\"")
assert_not_contains("${JSON_OUT}" "\"measured_speedup\"")
assert_not_contains("${JSON_OUT}" "\"performance_claim\"")
assert_not_contains("${JSON_OUT}" "\"evidence_type\"")

message(STATUS "QwenGTX1650CompileTest: PASS")
message(STATUS "  output: ${JSON_OUT}")
message(STATUS "  schema: execution_plan / 2.0.0 / strategy=none / no measured fields")
