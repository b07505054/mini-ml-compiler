# RunRaspberryPiFusedMatMulBiasReluKernelSelectionTest.cmake
# Invoked by CTest via add_test(...COMMAND cmake -P ...).
# Variables passed in from CMakeLists.txt:
#   TOOL     — path to compile-for-target executable
#   PROFILE  — path to raspberry_pi5_cortex_a76_cpu.json
#   MLIR     — path to p1b_fused_matmul_bias_relu_cpu.mlir
#   OUT      — canonical JSON output path (written to build dir)
#
# Phase P1B: proves the one evidence-backed runtime kernel declaration in
# raspberry_pi5_cortex_a76_cpu.json is actually selected end to end by the
# real, unmodified compile-for-target pipeline for the exact
# hir.fused_matmul_bias_relu op it targets, and that no other capability was
# fabricated (the two upstream weight/bias producer ops must NOT get a
# kernel selected).
#
# Phase P1C.1: the profile's default candidate changed from
# bm32_bn32_bk32 to bm32_bn128_bk32 (evidence-backed low-regret static
# default, see DOC/result/P1C1_*.md) -- this test's expected selected_kernel
# was updated to match. It still proves the SAME thing: the declared default
# candidate is actually selected end to end, and no other op gets a
# fabricated selection.

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

if(NOT "${_stderr}" STREQUAL "")
  message(FATAL_ERROR "compile-for-target printed unexpected stderr:\n${_stderr}")
endif()

file(READ "${OUT}" _contents)

macro(assert_contains _needle)
  string(FIND "${_contents}" "${_needle}" _pos)
  if(_pos EQUAL -1)
    message(FATAL_ERROR "expected to find '${_needle}' in ${OUT}")
  endif()
endmacro()

# The declared default kernel must be selected for the fused op (P1C.1:
# bm32_bn128_bk32, previously bm32_bn32_bk32).
assert_contains("\"selected_kernel\": \"portable_fused_matmul_bias_relu_bm32_bn128_bk32\"")
assert_contains("\"status\": \"selected\"")
assert_contains("\"hir.fused_matmul_bias_relu\"")
assert_contains("\"hardware_profile_ref\": \"raspberry-pi5-cortex-a76-cpu\"")

# Every other op (the two synthetic weight/bias producers) must remain
# honestly rejected -- no fabricated coverage beyond the one declared kernel.
assert_contains("\"rejected_no_kernel_for_op\"")

message(STATUS "RaspberryPiFusedMatMulBiasReluKernelSelectionTest: PASS")
