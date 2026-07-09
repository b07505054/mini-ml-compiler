foreach(var MLIR_OPT FILECHECK INPUT OUTPUT)
  if(NOT DEFINED ${var})
    message(FATAL_ERROR "[GenericMLIRPrototypeTest] Missing required variable: ${var}")
  endif()
endforeach()

execute_process(
  COMMAND "${MLIR_OPT}" "${INPUT}" -o "${OUTPUT}"
  RESULT_VARIABLE mlir_rc
  ERROR_VARIABLE mlir_stderr
)
if(NOT mlir_rc EQUAL 0)
  message(FATAL_ERROR
    "[GenericMLIRPrototypeTest] mlir-opt failed (rc=${mlir_rc}):\n${mlir_stderr}")
endif()

execute_process(
  COMMAND "${FILECHECK}" "${INPUT}" "--input-file=${OUTPUT}"
  RESULT_VARIABLE check_rc
  OUTPUT_VARIABLE check_stdout
  ERROR_VARIABLE check_stderr
)
if(NOT check_rc EQUAL 0)
  message(FATAL_ERROR
    "[GenericMLIRPrototypeTest] FileCheck failed (rc=${check_rc}):\n"
    "${check_stdout}${check_stderr}")
endif()
