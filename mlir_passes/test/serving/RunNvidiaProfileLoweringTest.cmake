# RunNvidiaProfileLoweringTest.cmake
# Invoked by CTest via add_test(...COMMAND cmake -P ...).
# Variables passed in from CMakeLists.txt:
#   TOOL       — path to compile-for-target executable
#   PROFILE    — path to nvidia_datacenter_gpu.json
#   MLIR       — path to tiny_gpt_serving.mlir
#   OUT        — canonical JSON output path (written to build dir)
#   DUMP_MLIR  — annotated MLIR output path (written to build dir)
#
# Tests:
#   2. backendCapabilities lower to target.backend_capabilities.* attrs in annotated MLIR.
#   3. Unknown cost fields are absent from the annotated MLIR.

get_filename_component(_out_dir "${OUT}" DIRECTORY)
file(MAKE_DIRECTORY "${_out_dir}")

execute_process(
  COMMAND "${TOOL}"
    --device-profile "${PROFILE}"
    --mlir           "${MLIR}"
    --out            "${OUT}"
    --dump-annotated-mlir "${DUMP_MLIR}"
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

if(NOT EXISTS "${DUMP_MLIR}")
  message(FATAL_ERROR "annotated MLIR not found: ${DUMP_MLIR}")
endif()

macro(assert_contains _file _needle)
  file(READ "${_file}" _contents)
  string(FIND "${_contents}" "${_needle}" _pos)
  if(_pos EQUAL -1)
    message(FATAL_ERROR
      "expected to find '${_needle}' in ${_file}")
  endif()
endmacro()

macro(assert_not_contains _file _needle)
  file(READ "${_file}" _contents)
  string(FIND "${_contents}" "${_needle}" _pos)
  if(NOT _pos EQUAL -1)
    message(FATAL_ERROR
      "unexpected '${_needle}' found in ${_file}")
  endif()
endmacro()

# Test 2: backendCapabilities (Level 1) lower to target.backend_capabilities.* attrs.
assert_contains("${DUMP_MLIR}" "target.backend_capabilities.tensor_core.backend_api")
assert_contains("${DUMP_MLIR}" "target.backend_capability_names")
assert_contains("${DUMP_MLIR}" "target.backend_capabilities.cuda.backend_api")
assert_contains("${DUMP_MLIR}" "nvidia-datacenter-gpu")

# Test — Level 2 constraint attrs are lowered (tensor_core backend).
assert_contains("${DUMP_MLIR}" "target.backend_capabilities.tensor_core.required_k_alignment")
assert_contains("${DUMP_MLIR}" "target.backend_capabilities.tensor_core.required_n_alignment")
assert_contains("${DUMP_MLIR}" "target.backend_capabilities.tensor_core.allowed_quant_granularity")
assert_contains("${DUMP_MLIR}" "target.backend_capabilities.tensor_core.requires_static_shape")
assert_contains("${DUMP_MLIR}" "target.backend_capabilities.tensor_core.requires_constant_weight")
assert_contains("${DUMP_MLIR}" "target.backend_capabilities.tensor_core.fallback_backend")

# Test — Level 3 preference attrs are lowered (tensor_core backend).
assert_contains("${DUMP_MLIR}" "target.backend_capabilities.tensor_core.acceptable_activation_layouts")
assert_contains("${DUMP_MLIR}" "target.backend_capabilities.tensor_core.preferred_dtypes")

# Test — Level 4 runtime model attrs are lowered (tensor_core backend).
assert_contains("${DUMP_MLIR}" "target.backend_capabilities.tensor_core.runtime_model_memory_hierarchy")
assert_contains("${DUMP_MLIR}" "target.backend_capabilities.tensor_core.runtime_model_truth_boundary")

# Test — Level 5 cost model placeholder attrs are lowered (tensor_core backend).
assert_contains("${DUMP_MLIR}" "target.backend_capabilities.tensor_core.cost_model_kind")
assert_contains("${DUMP_MLIR}" "target.backend_capabilities.tensor_core.cost_model_truth_boundary")

# Test 6: unknown cost fields are absent from the lowered MLIR (absent != 0.0).
assert_not_contains("${DUMP_MLIR}" "layout_transform_cost_ms")
assert_not_contains("${DUMP_MLIR}" "cast_cost_ms")
assert_not_contains("${DUMP_MLIR}" "quantize_cost_ms")
assert_not_contains("${DUMP_MLIR}" "dequantize_cost_ms")
assert_not_contains("${DUMP_MLIR}" "requantize_cost_ms")
assert_not_contains("${DUMP_MLIR}" "backend_transfer_cost_ms")

message(STATUS "NvidiaProfileLoweringTest: PASS")
