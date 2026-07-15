if(NOT DEFINED TOOL OR NOT DEFINED PROFILE OR NOT DEFINED MLIR OR
   NOT DEFINED BAD_MLIR OR NOT DEFINED OUT OR NOT DEFINED DUMP_MLIR)
  message(FATAL_ERROR "TOOL PROFILE MLIR BAD_MLIR OUT and DUMP_MLIR are required")
endif()

execute_process(
  COMMAND "${TOOL}" --device-profile "${PROFILE}" --mlir "${MLIR}"
          --out "${OUT}" --dump-annotated-mlir "${DUMP_MLIR}"
  RESULT_VARIABLE rc
  OUTPUT_VARIABLE stdout
  ERROR_VARIABLE stderr
)
if(NOT rc EQUAL 0)
  message(FATAL_ERROR "compile-for-target failed unexpectedly\nstdout=${stdout}\nstderr=${stderr}")
endif()

file(READ "${DUMP_MLIR}" mlir_text)
foreach(required
    "hir.quantize"
    "hir.load_quantized_weight"
    "hir.qmatmul"
    "hir.dequantize"
    "hir.portable_cpu_int8_fused_matmul_bias_relu"
    "tensor<64x64xi8>"
    "tensor<64x64xi32>"
    "packed_b_transposed_nxk"
    "slice3d_portable_cpu_int8_kernel_contract")
  string(FIND "${mlir_text}" "${required}" pos)
  if(pos EQUAL -1)
    message(FATAL_ERROR "annotated MLIR missing required text: ${required}\n${mlir_text}")
  endif()
endforeach()

file(READ "${OUT}" json_text)
foreach(required
    "\"execution_stages\""
    "\"quantize_activation\""
    "\"load_packed_weight\""
    "\"execute_int8_kernel\""
    "\"return_fp32_output\""
    "\"kernel_id\": \"portable_fused_matmul_bias_relu_int8_symmetric_packed_b\""
    "\"binary_sha256\": \"70f5adb276edefb9e9f0d8fabf807acc7304605b58abbeecc0874235e64c5ed3\""
    "\"packed_weight_artifact_id\": \"slice3b-packed-9327d80c62bf1965\"")
  string(FIND "${json_text}" "${required}" pos)
  if(pos EQUAL -1)
    message(FATAL_ERROR "execution plan missing required text: ${required}\n${json_text}")
  endif()
endforeach()

execute_process(
  COMMAND "${TOOL}" --device-profile "${PROFILE}" --mlir "${BAD_MLIR}"
          --out "${OUT}.bad.json"
  RESULT_VARIABLE bad_rc
  OUTPUT_VARIABLE bad_stdout
  ERROR_VARIABLE bad_stderr
)
if(bad_rc EQUAL 0)
  message(FATAL_ERROR "missing packed artifact input unexpectedly compiled")
endif()
string(FIND "${bad_stderr}" "quant.packed_weight_artifact_ref" err_pos)
if(err_pos EQUAL -1)
  message(FATAL_ERROR "missing artifact failure did not name quant.packed_weight_artifact_ref\nstdout=${bad_stdout}\nstderr=${bad_stderr}")
endif()
