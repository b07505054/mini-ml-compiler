# RunCVExecutionPlanArtifactToolTest.cmake
# Invoked by CTest via add_test(... COMMAND cmake -P ...).
# Variables:
#   TOOL       path to emit-cv-execution-plan
#   INPUT_MLIR path to CV MLIR input
#   OUT        output JSON path in the build tree

execute_process(
  COMMAND "${TOOL}"
    --input  "${INPUT_MLIR}"
    --output "${OUT}"
  RESULT_VARIABLE _rc
  OUTPUT_VARIABLE _stdout
  ERROR_VARIABLE  _stderr
)

if(NOT _rc EQUAL 0)
  message(FATAL_ERROR
    "emit-cv-execution-plan exited with code ${_rc}\n"
    "stdout:\n${_stdout}\n"
    "stderr:\n${_stderr}")
endif()

if(NOT EXISTS "${OUT}")
  message(FATAL_ERROR "expected JSON output was not created: ${OUT}")
endif()

file(READ "${OUT}" _json)

macro(assert_contains _needle)
  string(FIND "${_json}" "${_needle}" _pos)
  if(_pos EQUAL -1)
    message(FATAL_ERROR "expected JSON to contain '${_needle}'")
  endif()
endmacro()

macro(assert_not_contains _needle)
  string(FIND "${_json}" "${_needle}" _pos)
  if(NOT _pos EQUAL -1)
    message(FATAL_ERROR "JSON contained forbidden string '${_needle}'")
  endif()
endmacro()

assert_contains("\"artifact_type\": \"cv_execution_plan\"")
assert_contains("\"model_name\": \"yoloseg\"")
assert_contains("\"graph_plans\"")
assert_contains("\"execution_domain\"")
assert_contains("\"memory_summary\"")

assert_not_contains("metal")
assert_not_contains("cpu")
assert_not_contains("cuda")
assert_not_contains("coreml")
assert_not_contains("ane")

message(STATUS "CVExecutionPlanArtifactToolTest: PASS")
