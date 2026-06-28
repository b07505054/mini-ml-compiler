# RunCVFrontendNormalizationTest.cmake
# Runs cv-frontend-normalization on mlir/cv_raw_yoloseg.mlir and checks
# that func.func attrs are emitted correctly.
# No compile-for-target step (no CV plan builder exists yet).
#
# Required variables (passed via -D from CMakeLists.txt):
#   MLIR_OPT   -- path to mlir-opt
#   PLUGIN     -- path to HIRMatMulBiasReluFusionPass plugin (.dylib / .so)
#   RAW_MLIR   -- path to mlir/cv_raw_yoloseg.mlir
#   NORM_OUT   -- output path for the normalized MLIR (written to build dir)

cmake_minimum_required(VERSION 3.20)

foreach(var MLIR_OPT PLUGIN RAW_MLIR NORM_OUT)
  if(NOT DEFINED ${var})
    message(FATAL_ERROR
      "[CVFrontendNormalizationTest] Missing required variable: ${var}")
  endif()
endforeach()

get_filename_component(NORM_DIR "${NORM_OUT}" DIRECTORY)
file(MAKE_DIRECTORY "${NORM_DIR}")

# ---- Run mlir-opt with cv-frontend-normalization ----
execute_process(
  COMMAND "${MLIR_OPT}"
    --allow-unregistered-dialect
    "--load-pass-plugin=${PLUGIN}"
    "--pass-pipeline=builtin.module(cv-frontend-normalization)"
    "${RAW_MLIR}"
    -o "${NORM_OUT}"
  RESULT_VARIABLE rc
  OUTPUT_VARIABLE stdout
  ERROR_VARIABLE  stderr
)

if(NOT rc EQUAL 0)
  message(FATAL_ERROR
    "[CVFrontendNormalizationTest] mlir-opt failed (rc=${rc}):\n${stderr}")
endif()

if(NOT EXISTS "${NORM_OUT}")
  message(FATAL_ERROR
    "[CVFrontendNormalizationTest] output file not found: ${NORM_OUT}")
endif()

# ---- Check that the normalized MLIR contains expected func.func attrs ----
file(READ "${NORM_OUT}" norm_content)

macro(assert_contains _needle)
  string(FIND "${norm_content}" "${_needle}" _pos)
  if(_pos EQUAL -1)
    message(FATAL_ERROR
      "[CVFrontendNormalizationTest] normalized MLIR missing: '${_needle}'")
  endif()
endmacro()

macro(assert_absent _needle)
  string(FIND "${norm_content}" "${_needle}" _pos)
  if(NOT _pos EQUAL -1)
    message(FATAL_ERROR
      "[CVFrontendNormalizationTest] normalized MLIR should not contain: '${_needle}'")
  endif()
endmacro()

# --- Presence checks: @yoloseg_inference should be annotated ---
assert_contains("cv.frontend.kind = \"raw_pseudo_cv_mlir\"")
assert_contains("cv.frontend.model_family = \"yoloseg\"")
assert_contains("cv.frontend.normalization_status = \"detected_not_rewritten\"")
assert_contains("cv.frontend.truth_boundary = \"raw_pseudo_cv_mlir_not_full_onnx_importer\"")
assert_contains("cv.frontend.conv2d_count = 5 : i64")
assert_contains("cv.frontend.batch_norm_count = 1 : i64")
assert_contains("cv.frontend.silu_count = 4 : i64")
assert_contains("cv.frontend.upsample_count = 1 : i64")
assert_contains("cv.frontend.concat_count = 1 : i64")
assert_contains("cv.frontend.detect_head_count = 1 : i64")
assert_contains("cv.frontend.prototype_head_count = 1 : i64")

# --- Body preservation check: ops must still be present after detect-and-annotate ---
assert_contains("\"cv.conv2d\"")
assert_contains("\"cv.detect_head\"")
assert_contains("\"cv.prototype_head\"")

message(STATUS "[CVFrontendNormalizationTest] passed")
