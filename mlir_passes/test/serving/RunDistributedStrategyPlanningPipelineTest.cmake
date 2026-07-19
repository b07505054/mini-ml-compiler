# RunDistributedStrategyPlanningPipelineTest.cmake
#
# D2/D6 end-to-end integration test, real production pipeline:
#   qwen-onnx-to-serving-mlir (real per-layer Qwen graph)
#   -> compile-for-target (real pipeline incl. DistributedStrategyPlanningPass)
#   -> execution_plan.json (schema_version 2.0.0)
#
# Proves, against REAL freshly-generated artifacts (not stale fixtures):
#   1. Pass registration + execution in the real Qwen pipeline.
#   2. TP1 backward compatibility: the non-opt-in profile's exported plan
#      carries no "distributed" key at all, regardless of model/calibration.
#   3. D6 real profitability selection: the opt-in RTX4090 profile (real
#      D5-calibrated coefficients) + the 7B model/workload profile produces
#      a genuinely profitability-selected TP2 -- world_size=2,
#      tensor_parallel_size=2, an all_reduce collective referencing the
#      real llm.o_proj operator -- because TP2 really does predict higher
#      throughput for this model/workload, not because opt_in alone forced
#      it (see CMakeLists.txt's rationale comment for why this moved from
#      the 0.5B model to the 7B model).
#   4. Candidate/legality/profitability/selection evidence serialization
#      via the optional --distributed-evidence-report.
#
# Required variables (passed via -D):
#   ONNX_TOOL         -- path to qwen-onnx-to-serving-mlir
#   CFT_TOOL          -- path to compile-for-target
#   FACTS             -- path to configs/models/qwen_7b_onnx_graph_facts.json
#   PROFILE_TP1       -- path to a target profile JSON without distributedStrategyOptIn
#   PROFILE_TP2       -- path to a target profile JSON with distributedStrategyOptIn=true
#                        and a real distributedProfitability calibration block
#   MODEL_PROFILE     -- path to a --model-profile JSON (real weightFootprintMb)
#   WORKLOAD_PROFILE  -- path to a --workload-profile JSON (real workload shape)
#   RAW_OUT       -- output path for raw (pre-normalization) MLIR
#   JSON_TP1_OUT  -- output path for the TP1 execution_plan.json
#   JSON_TP2_OUT  -- output path for the TP2 execution_plan.json
#   EVIDENCE_OUT  -- output path for the TP2 distributed evidence report

cmake_minimum_required(VERSION 3.20)

foreach(var ONNX_TOOL CFT_TOOL FACTS PROFILE_TP1 PROFILE_TP2 MODEL_PROFILE WORKLOAD_PROFILE
            RAW_OUT JSON_TP1_OUT JSON_TP2_OUT EVIDENCE_OUT)
  if(NOT DEFINED ${var})
    message(FATAL_ERROR "Missing required variable: ${var}")
  endif()
endforeach()

get_filename_component(RAW_DIR "${RAW_OUT}" DIRECTORY)
file(MAKE_DIRECTORY "${RAW_DIR}")

execute_process(
  COMMAND "${ONNX_TOOL}" --graph-facts "${FACTS}" --out "${RAW_OUT}"
  RESULT_VARIABLE rc OUTPUT_VARIABLE stdout ERROR_VARIABLE stderr
)
if(NOT rc EQUAL 0)
  message(FATAL_ERROR
    "[DistributedStrategyPlanningPipelineTest] qwen-onnx-to-serving-mlir failed (rc=${rc}):\n${stderr}")
endif()

get_filename_component(JSON_TP1_DIR "${JSON_TP1_OUT}" DIRECTORY)
file(MAKE_DIRECTORY "${JSON_TP1_DIR}")
execute_process(
  COMMAND "${CFT_TOOL}" "--device-profile=${PROFILE_TP1}" "--mlir=${RAW_OUT}" "--out=${JSON_TP1_OUT}"
  RESULT_VARIABLE rc OUTPUT_VARIABLE stdout ERROR_VARIABLE stderr
)
if(NOT rc EQUAL 0)
  message(FATAL_ERROR
    "[DistributedStrategyPlanningPipelineTest] compile-for-target (TP1 profile) failed (rc=${rc}):\n${stderr}")
endif()

get_filename_component(JSON_TP2_DIR "${JSON_TP2_OUT}" DIRECTORY)
file(MAKE_DIRECTORY "${JSON_TP2_DIR}")
execute_process(
  COMMAND "${CFT_TOOL}" "--device-profile=${PROFILE_TP2}" "--model-profile=${MODEL_PROFILE}"
    "--workload-profile=${WORKLOAD_PROFILE}" "--mlir=${RAW_OUT}" "--out=${JSON_TP2_OUT}"
    "--distributed-evidence-report=${EVIDENCE_OUT}"
  RESULT_VARIABLE rc OUTPUT_VARIABLE stdout ERROR_VARIABLE stderr
)
if(NOT rc EQUAL 0)
  message(FATAL_ERROR
    "[DistributedStrategyPlanningPipelineTest] compile-for-target (TP2 opt-in profile) failed (rc=${rc}):\n${stderr}")
endif()

# --- TP1 plan: no distributed key at all (backward compatibility) ---
file(READ "${JSON_TP1_OUT}" tp1_content)
string(FIND "${tp1_content}" "\"distributed\"" _pos)
if(NOT _pos EQUAL -1)
  message(FATAL_ERROR
    "[DistributedStrategyPlanningPipelineTest] TP1 (non-opt-in) plan must not contain a 'distributed' key")
endif()

# --- TP2 plan: real distributed block ---
file(READ "${JSON_TP2_OUT}" tp2_content)
foreach(_needle "\"distributed\"" "\"world_size\": 2" "\"tensor_parallel_size\": 2"
                 "llm.o_proj" "\"kind\": \"all_reduce\"" "qwen_prefill")
  string(FIND "${tp2_content}" "${_needle}" _pos)
  if(_pos EQUAL -1)
    message(FATAL_ERROR
      "[DistributedStrategyPlanningPipelineTest] TP2 plan missing expected content: '${_needle}'")
  endif()
endforeach()

# --- Evidence report: both candidates, real D6 profitability-based selection ---
file(READ "${EVIDENCE_OUT}" evidence_content)
foreach(_needle "\"tp1\"" "\"tp2\"" "\"selected_candidate_id\": \"tp2\""
                 "\"selection_reason\": \"profitable_tp2_selected_predicted_throughput_higher\""
                 "\"policy_id\": \"d6_profitability_selector_v1\""
                 "\"predicted_throughput_tokens_per_s\"")
  string(FIND "${evidence_content}" "${_needle}" _pos)
  if(_pos EQUAL -1)
    message(FATAL_ERROR
      "[DistributedStrategyPlanningPipelineTest] evidence report missing expected content: '${_needle}'")
  endif()
endforeach()

message(STATUS "[DistributedStrategyPlanningPipelineTest] passed")
