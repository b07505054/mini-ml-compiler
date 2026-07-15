module attributes {
  model.id = "slice3d_int8_missing_packed_artifact_test",
  target.backend = "cpu"
} {
  func.func @main(
      %a: tensor<64x64xf32>,
      %b: tensor<64x64xf32>,
      %bias: tensor<64xf32>) -> tensor<64x64xf32>
      attributes {cv.semantic_annotation.status = "completed"} {
    %0 = hir.fused_matmul_bias_relu %a, %b, %bias {
      fusion.candidate = "matmul_bias_relu",
      kernel.selection = "selected",
      lowering.source = "linalg.matmul_add_relu",
      quant.strategy = "int8_static_symmetric",
      quant.scheme = "int8_static_symmetric",
      quant.selected_candidate_id = "fused_matmul_bias_relu:quant=int8_static_symmetric:shape=64x64x64",
      quant.selected_complete_candidate_id = "slice3c:portable_cpu:int8_static_symmetric:packed_b_transposed_nxk:cortex_a76_dotprod",
      quant.activation_dtype = "int8",
      quant.weight_dtype = "int8",
      quant.accumulation_dtype = "int32",
      quant.output_dtype = "fp32",
      quant.granularity = "per_tensor",
      quant.activation_granularity = "per_tensor",
      quant.weight_granularity = "per_tensor",
      quant.requires_calibration = true,
      quant.calibration_available = true,
      quant.activation_scale = 0.007871949237011344 : f64,
      quant.weight_scale = 0.007872035035652262 : f64,
      quant.activation_zero_point = 0 : i64,
      quant.weight_zero_point = 0 : i64,
      quant.required_kernel_capability = "quant_kernel.int8_static_symmetric.packed_b_transposed",
      quant.kernel_requires_packed_weight = true,
      quant.kernel_id = "portable_fused_matmul_bias_relu_int8_symmetric_packed_b",
      quant.calibration_artifact_ref = "results/slice3b_packed_weight_host/calibration_64x64x64.json",
      quant.calibration_artifact_id = "slice3a-f14f567ebc6c60bc",
      quant.calibration_artifact_sha256 = "6ec81ec7673a089edb12d61f3c372618e2196dbcdc082b7f5d4d1c48775c86de",
      quant.packed_layout = "packed_b_transposed_nxk",
      quant.packing_scheme = "b_transposed_nxk_contiguous",
      quant.source_weight_sha256 = "87744dba563f3226651e6405ae86d2f56b76bb15bfa805b84bd03c6e89828475",
      quant.workload_id = "slice3a_fused_matmul_bias_relu_64x64x64_seed100",
      quant.codegen_target_id = "cortex_a76_dotprod",
      quant.binary_sha256 = "70f5adb276edefb9e9f0d8fabf807acc7304605b58abbeecc0874235e64c5ed3"
    } : (tensor<64x64xf32>, tensor<64x64xf32>, tensor<64xf32>) -> tensor<64x64xf32>
    return %0 : tensor<64x64xf32>
  }
}
