# RunCVMemoryPlanningTest.cmake
# Primary: runs cv-memory-planning on the dedicated small-shape test file and
#   checks that buffer assignment, lifetime, and reuse attrs are emitted.
# Secondary smoke: runs cv-shape-inference,cv-memory-planning on
#   cv_raw_yoloseg.mlir and checks only that status = "completed" appears.
#
# Required variables (passed via -D from CMakeLists.txt):
#   MLIR_OPT     -- path to mlir-opt
#   PLUGIN       -- path to HIRMatMulBiasReluFusionPass plugin (.dylib / .so)
#   INPUT_MLIR   -- path to mlir_passes/test/serving/cv_memory_planning.mlir
#   PLAN_OUT     -- output path for annotated MLIR (primary run)
#   YOLOSEG_MLIR -- path to mlir/cv_raw_yoloseg.mlir (smoke run)
#   SMOKE_OUT    -- output path for smoke-run MLIR

cmake_minimum_required(VERSION 3.20)

foreach(var MLIR_OPT PLUGIN INPUT_MLIR PLAN_OUT YOLOSEG_MLIR SMOKE_OUT)
  if(NOT DEFINED ${var})
    message(FATAL_ERROR
      "[CVMemoryPlanningTest] Missing required variable: ${var}")
  endif()
endforeach()

get_filename_component(PLAN_DIR "${PLAN_OUT}" DIRECTORY)
file(MAKE_DIRECTORY "${PLAN_DIR}")

# ---- Primary run: cv-memory-planning alone on test file ----
execute_process(
  COMMAND "${MLIR_OPT}"
    --allow-unregistered-dialect
    "--load-pass-plugin=${PLUGIN}"
    "--pass-pipeline=builtin.module(cv-memory-planning)"
    "${INPUT_MLIR}"
    -o "${PLAN_OUT}"
  RESULT_VARIABLE rc
  OUTPUT_VARIABLE stdout
  ERROR_VARIABLE  stderr
)

if(NOT rc EQUAL 0)
  message(FATAL_ERROR
    "[CVMemoryPlanningTest] mlir-opt failed (rc=${rc}):\n${stderr}")
endif()

if(NOT EXISTS "${PLAN_OUT}")
  message(FATAL_ERROR
    "[CVMemoryPlanningTest] output file not found: ${PLAN_OUT}")
endif()

file(READ "${PLAN_OUT}" plan_content)

macro(assert_contains _needle)
  string(FIND "${plan_content}" "${_needle}" _pos)
  if(_pos EQUAL -1)
    message(FATAL_ERROR
      "[CVMemoryPlanningTest] primary output missing: '${_needle}'")
  endif()
endmacro()

# @chain: 3 planned, 0 skipped, 2 buffers, 1 reused, 64 bytes total and peak.
assert_contains("cv.memory_plan.buffer_count = 2 : i64")
assert_contains("cv.memory_plan.peak_memory_bytes = 64 : i64")
assert_contains("cv.memory_plan.planned_ops = 3 : i64")
assert_contains("cv.memory_plan.reused_buffer_count = 1 : i64")
assert_contains("cv.memory_plan.skipped_ops = 0 : i64")
assert_contains("cv.memory_plan.status = \"completed\"")
assert_contains("cv.memory_plan.total_allocated_bytes = 64 : i64")
assert_contains("cv.memory_plan.truth_boundary = \"static_compiler_memory_plan_not_runtime_allocation\"")

# Per-op attrs: buffer_id, buffer_offset, and lifetime present.
assert_contains("cv.buffer_id = 0 : i64")
assert_contains("cv.buffer_offset = 0 : i64")
assert_contains("cv.lifetime_begin = 0 : i64")
assert_contains("cv.lifetime_end = 1 : i64")
assert_contains("cv.reuse_group = 0 : i64")

# @branch: A's lifetime_end = 2 (fan-out extension).
assert_contains("cv.lifetime_end = 2 : i64")

# @missing_bytes: planned_ops=1, skipped_ops=1.
assert_contains("cv.memory_plan.planned_ops = 1 : i64")
assert_contains("cv.memory_plan.skipped_ops = 1 : i64")
assert_contains("cv.memory_plan.total_allocated_bytes = 32 : i64")

# ---- Secondary smoke run: cv-shape-inference,cv-memory-planning on yoloseg ----
execute_process(
  COMMAND "${MLIR_OPT}"
    --allow-unregistered-dialect
    "--load-pass-plugin=${PLUGIN}"
    "--pass-pipeline=builtin.module(cv-shape-inference,cv-memory-planning)"
    "${YOLOSEG_MLIR}"
    -o "${SMOKE_OUT}"
  RESULT_VARIABLE smoke_rc
  OUTPUT_VARIABLE smoke_stdout
  ERROR_VARIABLE  smoke_stderr
)

if(NOT smoke_rc EQUAL 0)
  message(FATAL_ERROR
    "[CVMemoryPlanningTest] smoke run failed (rc=${smoke_rc}):\n${smoke_stderr}")
endif()

file(READ "${SMOKE_OUT}" smoke_content)

string(FIND "${smoke_content}"
  "cv.memory_plan.status = \"completed\"" smoke_pos)
if(smoke_pos EQUAL -1)
  message(FATAL_ERROR
    "[CVMemoryPlanningTest] smoke run missing cv.memory_plan.status = \"completed\"")
endif()

message(STATUS "[CVMemoryPlanningTest] passed")
