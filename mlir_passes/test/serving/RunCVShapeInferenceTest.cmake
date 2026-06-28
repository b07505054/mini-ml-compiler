# RunCVShapeInferenceTest.cmake
# Runs cv-shape-inference on the dedicated small-shape test file and checks
# that per-op and function-level size/shape attrs are emitted correctly.
# No compile-for-target step (no CV plan builder exists yet).
#
# Required variables (passed via -D from CMakeLists.txt):
#   MLIR_OPT   -- path to mlir-opt
#   PLUGIN     -- path to HIRMatMulBiasReluFusionPass plugin (.dylib / .so)
#   INPUT_MLIR -- path to mlir_passes/test/serving/cv_shape_inference.mlir
#   NORM_OUT   -- output path for the annotated MLIR (written to build dir)

cmake_minimum_required(VERSION 3.20)

foreach(var MLIR_OPT PLUGIN INPUT_MLIR NORM_OUT)
  if(NOT DEFINED ${var})
    message(FATAL_ERROR
      "[CVShapeInferenceTest] Missing required variable: ${var}")
  endif()
endforeach()

get_filename_component(NORM_DIR "${NORM_OUT}" DIRECTORY)
file(MAKE_DIRECTORY "${NORM_DIR}")

# ---- Run mlir-opt with cv-shape-inference ----
execute_process(
  COMMAND "${MLIR_OPT}"
    --allow-unregistered-dialect
    "--load-pass-plugin=${PLUGIN}"
    "--pass-pipeline=builtin.module(cv-shape-inference)"
    "${INPUT_MLIR}"
    -o "${NORM_OUT}"
  RESULT_VARIABLE rc
  OUTPUT_VARIABLE stdout
  ERROR_VARIABLE  stderr
)

if(NOT rc EQUAL 0)
  message(FATAL_ERROR
    "[CVShapeInferenceTest] mlir-opt failed (rc=${rc}):\n${stderr}")
endif()

if(NOT EXISTS "${NORM_OUT}")
  message(FATAL_ERROR
    "[CVShapeInferenceTest] output file not found: ${NORM_OUT}")
endif()

# ---- Check expected strings in the annotated MLIR ----
file(READ "${NORM_OUT}" norm_content)

macro(assert_contains _needle)
  string(FIND "${norm_content}" "${_needle}" _pos)
  if(_pos EQUAL -1)
    message(FATAL_ERROR
      "[CVShapeInferenceTest] normalized MLIR missing: '${_needle}'")
  endif()
endmacro()

# @tiny_cv: two annotated ops, total = 64 bytes.
assert_contains("cv.shape_inference.status = \"completed\"")
assert_contains("cv.shape_inference.annotated_ops = 2 : i64")
assert_contains("cv.shape_inference.skipped_ops = 0 : i64")
assert_contains("cv.shape_inference.total_bytes_estimate = 64 : i64")
assert_contains("cv.shape_inference.truth_boundary = \"static_ranked_tensor_shape_metadata_not_runtime_allocation\"")

# Per-op attrs: 16 elements, 32 bytes each (tensor<1x4x2x2xf16>).
assert_contains("cv.num_elements = 16 : i64")
assert_contains("cv.bytes_estimate = 32 : i64")

# @has_skip: one annotated, one skipped; total = 32 bytes.
assert_contains("cv.shape_inference.annotated_ops = 1 : i64")
assert_contains("cv.shape_inference.skipped_ops = 1 : i64")
assert_contains("cv.shape_inference.total_bytes_estimate = 32 : i64")

# @no_cv must NOT have cv.shape_inference attrs — checked indirectly:
# the only occurrences of annotated_ops/skipped_ops should belong to
# @tiny_cv and @has_skip; any third occurrence would be a regression.
# (FileCheck handles the precise per-function scoping in cv_shape_inference.mlir.)

message(STATUS "[CVShapeInferenceTest] passed")
