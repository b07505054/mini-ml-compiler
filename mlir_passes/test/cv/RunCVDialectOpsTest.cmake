# RunCVDialectOpsTest.cmake
# Variables:
#   MLIR_OPT -- path to mlir-opt
#   PLUGIN   -- path to dialect plugin
#   INPUT    -- path to cv_dialect_ops.mlir

foreach(var MLIR_OPT PLUGIN INPUT)
  if(NOT DEFINED ${var})
    message(FATAL_ERROR "[CVDialectOpsTest] Missing required variable: ${var}")
  endif()
endforeach()

execute_process(
  COMMAND "${MLIR_OPT}"
    "--load-dialect-plugin=${PLUGIN}"
    "${INPUT}"
  RESULT_VARIABLE rc
  OUTPUT_VARIABLE output
  ERROR_VARIABLE stderr
)

if(NOT rc EQUAL 0)
  message(FATAL_ERROR
    "[CVDialectOpsTest] mlir-opt failed (rc=${rc}):\n${stderr}")
endif()

macro(assert_contains needle)
  string(FIND "${output}" "${needle}" pos)
  if(pos EQUAL -1)
    message(FATAL_ERROR
      "[CVDialectOpsTest] output missing '${needle}'\n${output}")
  endif()
endmacro()

assert_contains("cv.conv2d")
assert_contains("cv.batch_norm")
assert_contains("cv.silu")
assert_contains("cv.concat")
assert_contains("cv.upsample")
assert_contains("cv.detect_head")
assert_contains("cv.prototype_head")

message(STATUS "[CVDialectOpsTest] passed")
