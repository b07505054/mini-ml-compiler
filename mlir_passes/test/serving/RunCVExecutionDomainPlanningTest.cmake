# RunCVExecutionDomainPlanningTest.cmake
# Primary: runs cv-execution-domain-planning on the dedicated test file and
#   checks that execution domain classification attrs are emitted correctly.
# Secondary smoke: runs the full 4-pass CV pipeline on cv_raw_yoloseg.mlir
#   and checks that cv.execution_domain_plan.status = "completed" appears.
#
# Required variables (passed via -D from CMakeLists.txt):
#   MLIR_OPT     -- path to mlir-opt
#   PLUGIN       -- path to HIRMatMulBiasReluFusionPass plugin (.dylib / .so)
#   INPUT_MLIR   -- path to mlir_passes/test/serving/cv_execution_domain_planning.mlir
#   PLAN_OUT     -- output path for primary run
#   YOLOSEG_MLIR -- path to mlir/cv_raw_yoloseg.mlir (smoke run)
#   SMOKE_OUT    -- output path for smoke-run MLIR

cmake_minimum_required(VERSION 3.20)

foreach(var MLIR_OPT PLUGIN INPUT_MLIR PLAN_OUT YOLOSEG_MLIR SMOKE_OUT)
  if(NOT DEFINED ${var})
    message(FATAL_ERROR
      "[CVExecutionDomainPlanningTest] Missing required variable: ${var}")
  endif()
endforeach()

get_filename_component(PLAN_DIR "${PLAN_OUT}" DIRECTORY)
file(MAKE_DIRECTORY "${PLAN_DIR}")

# ---- Primary run: cv-execution-domain-planning alone on test file ----
execute_process(
  COMMAND "${MLIR_OPT}"
    --allow-unregistered-dialect
    "--load-pass-plugin=${PLUGIN}"
    "--pass-pipeline=builtin.module(cv-execution-domain-planning)"
    "${INPUT_MLIR}"
    -o "${PLAN_OUT}"
  RESULT_VARIABLE rc
  OUTPUT_VARIABLE stdout
  ERROR_VARIABLE  stderr
)

if(NOT rc EQUAL 0)
  message(FATAL_ERROR
    "[CVExecutionDomainPlanningTest] mlir-opt failed (rc=${rc}):\n${stderr}")
endif()

if(NOT EXISTS "${PLAN_OUT}")
  message(FATAL_ERROR
    "[CVExecutionDomainPlanningTest] output file not found: ${PLAN_OUT}")
endif()

file(READ "${PLAN_OUT}" plan_content)

macro(assert_contains _needle)
  string(FIND "${plan_content}" "${_needle}" _pos)
  if(_pos EQUAL -1)
    message(FATAL_ERROR
      "[CVExecutionDomainPlanningTest] primary output missing: '${_needle}'")
  endif()
endmacro()

# @default_cv_graph: 4 accelerated, 2 host, 1 fallback, 7 planned.
assert_contains("cv.execution_domain_plan.accelerated_ops = 4 : i64")
assert_contains("cv.execution_domain_plan.fallback_ops = 1 : i64")
assert_contains("cv.execution_domain_plan.host_ops = 2 : i64")
assert_contains("cv.execution_domain_plan.planned_ops = 7 : i64")
assert_contains("cv.execution_domain_plan.status = \"completed\"")
assert_contains("cv.execution_domain_plan.truth_boundary = \"static_execution_domain_classification_not_target_mapping\"")

# Per-op domain values (all three must appear).
assert_contains("cv.execution_domain = \"accelerated\"")
assert_contains("cv.execution_domain = \"host\"")
assert_contains("cv.execution_domain = \"fallback\"")

# Per-op reason strings.
assert_contains("cv.execution_domain_reason = \"accelerated_tensor_policy\"")
assert_contains("cv.execution_domain_reason = \"host_postprocess_policy\"")
assert_contains("cv.execution_domain_reason = \"fallback_unknown_cv_op\"")

# Per-op truth boundary.
assert_contains("cv.execution_domain.truth_boundary = \"static_execution_domain_classification_not_target_mapping\"")

# ---- Secondary smoke: full 4-pass CV pipeline on cv_raw_yoloseg.mlir ----
execute_process(
  COMMAND "${MLIR_OPT}"
    --allow-unregistered-dialect
    "--load-pass-plugin=${PLUGIN}"
    "--pass-pipeline=builtin.module(cv-frontend-normalization,cv-shape-inference,cv-memory-planning,cv-execution-domain-planning)"
    "${YOLOSEG_MLIR}"
    -o "${SMOKE_OUT}"
  RESULT_VARIABLE smoke_rc
  OUTPUT_VARIABLE smoke_stdout
  ERROR_VARIABLE  smoke_stderr
)

if(NOT smoke_rc EQUAL 0)
  message(FATAL_ERROR
    "[CVExecutionDomainPlanningTest] smoke run failed (rc=${smoke_rc}):\n${smoke_stderr}")
endif()

file(READ "${SMOKE_OUT}" smoke_content)

string(FIND "${smoke_content}"
  "cv.execution_domain_plan.status = \"completed\"" smoke_pos)
if(smoke_pos EQUAL -1)
  message(FATAL_ERROR
    "[CVExecutionDomainPlanningTest] smoke run missing cv.execution_domain_plan.status = \"completed\"")
endif()

message(STATUS "[CVExecutionDomainPlanningTest] passed")
