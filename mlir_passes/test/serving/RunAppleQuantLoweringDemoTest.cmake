# RunAppleQuantLoweringDemoTest.cmake
# CTest driver for Commit L: Apple/CoreML quantization-lowering end-to-end demo.
#
# Verifies that compile-for-target produces a complete serving execution plan
# from the apple_quant_lowering_demo fixture showing the full decision chain:
#   profile -> representation -> layout -> boundary -> quant strategy
#   -> kernel availability -> lowering decision -> JSON
#
# Variables passed in from CMakeLists.txt:
#   TOOL     -- path to compile-for-target executable
#   PROFILE  -- integration_bundle/apple_quant_lowering_demo/target_profile.json
#   MLIR     -- integration_bundle/apple_quant_lowering_demo/input.mlir
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
      "actual contents (first 4096 chars):\n${_contents}")
  endif()
endmacro()

# Test 1: compile-for-target produces a well-formed execution plan.
assert_contains("${OUT}" "\"artifact_type\": \"serving_execution_plan\"")
assert_contains("${OUT}" "\"schema_version\": \"1.0.0\"")

# Test 2: JSON includes per_op_quantization_decisions array.
assert_contains("${OUT}" "\"per_op_quantization_decisions\"")

# Test 3: JSON includes per_op_lowering_decisions array.
assert_contains("${OUT}" "\"per_op_lowering_decisions\"")

# Test 4: hir.matmul and hir.conv2d inputs are runtime tensors (from llm.attention_prefill).
# WeightClassificationPlanningPass sets weight.constant_satisfied=false for both.
# QuantizationStrategyPlanningPass reads this -> fp16_fallback, fallback_reason=weight_not_constant.
assert_contains("${OUT}" "\"fallback_reason\": \"weight_not_constant\"")

# Test 5: At least one op has strategy=fp16_fallback (hir.softmax — accuracy-sensitive).
assert_contains("${OUT}" "\"strategy\": \"fp16_fallback\"")

# Test 6: At least one op has lowering_decision=direct_lower (hir.softmax — coreml kernel).
assert_contains("${OUT}" "\"lowering_decision\": \"direct_lower\"")

# Test 7: At least one op has lowering_decision=fallback_backend
#          (hir.matmul / hir.conv2d — requiresConstantWeight but inputs are runtime tensors).
assert_contains("${OUT}" "\"lowering_decision\": \"fallback_backend\"")

# Test 8: truth_boundary fields present for both decision arrays.
assert_contains("${OUT}" "lowering_decision_static_not_backend_codegen_verified")
assert_contains("${OUT}" "quantization_strategy_static_not_accuracy_calibrated")

# Backend: coreml should be the primary backend (not constraint_conflict).
assert_contains("${OUT}" "\"primary_backend\": \"coreml\"")
assert_contains("${OUT}" "\"decision_source\": \"target_preferred\"")

message(STATUS "AppleQuantLoweringDemoTest: PASS")
