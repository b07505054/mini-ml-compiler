// FileCheck tests for KernelSelectionPass (kernel_selection_contract_v1).
//
// Run:
//   mlir-opt \
//     --split-input-file \
//     --allow-unregistered-dialect \
//     --load-pass-plugin="$PLUGIN" \
//     --pass-pipeline='builtin.module(tile-planning-pipeline,kernel-selection-pipeline)' \
//     kernel_selection.mlir | \
//   FileCheck kernel_selection.mlir --input-file=- --split-input-file
//
// The registry mirrors reality: the only handwritten runtime kernel with a
// dispatch-validated path in this repo is Metal RMSNorm f32. Synthetic
// descriptors used to exercise rejection paths are labeled source =
// "fixture". A selection is a compiler/runtime contract — never a claim of
// runtime execution or measured performance.

// -----

// Test 1: RMSNorm selected when the descriptor fully matches (op name,
// backend, fp32 dtype, quant none, static shapes) — and MatMul on the same
// function is rejected because no descriptor exists for it: coverage is
// never inferred from the library layer.
//
// CHECK-LABEL: func.func @rmsnorm_selected_matmul_rejected
// CHECK: "llm.rmsnorm"
// CHECK-SAME: kernel_selection.contract_version = "kernel_selection_contract_v1"
// CHECK-SAME: kernel_selection.selected_id = "metal_rmsnorm_f32_v1"
// CHECK-SAME: kernel_selection.source = "handwritten_runtime"
// CHECK-SAME: kernel_selection.status = "selected"
// CHECK-SAME: kernel_selection.truth_boundary = "handwritten_kernel_source_in_repo_dispatch_validated_not_benchmarked"
// CHECK: "compute.matmul"
// CHECK-SAME: kernel_selection.status = "rejected_no_kernel_for_op"

module attributes {
  target.runtime_kernels = [{
    backend = "metal",
    kernel_id = "metal_rmsnorm_f32_v1",
    op_name = "rmsnorm",
    requires_local_memory_bytes = 0 : i64,
    requires_static_shape = true,
    source = "handwritten_runtime",
    supported_dtypes = ["fp32"],
    supported_layouts = [],
    supported_quant_modes = ["none"],
    supported_tile_shapes = [],
    truth_boundary = "handwritten_kernel_source_in_repo_dispatch_validated_not_benchmarked"
  }]
} {
  func.func @rmsnorm_selected_matmul_rejected(%a: tensor<16x768xf32>)
      -> tensor<16x768xf32>
      attributes {representation.source_backend = "metal"} {
    %0 = "llm.rmsnorm"(%a) : (tensor<16x768xf32>) -> tensor<16x768xf32>
    %1 = "compute.matmul"(%0) : (tensor<16x768xf32>) -> tensor<16x768xf32>
    return %1 : tensor<16x768xf32>
  }
}

// -----

// Test 2: dtype mismatch is a rejection with a per-descriptor reason —
// the kernel is fp32-only, the op runs fp16.
//
// CHECK-LABEL: func.func @dtype_mismatch_rejected
// CHECK: "llm.rmsnorm"
// CHECK-SAME: kernel_selection.rejection_reasons = ["metal_rmsnorm_f32_v1:dtype_unsupported"]
// CHECK-SAME: kernel_selection.status = "rejected_dtype_unsupported"

module attributes {
  target.runtime_kernels = [{
    backend = "metal",
    kernel_id = "metal_rmsnorm_f32_v1",
    op_name = "rmsnorm",
    requires_local_memory_bytes = 0 : i64,
    requires_static_shape = true,
    source = "handwritten_runtime",
    supported_dtypes = ["fp32"],
    supported_layouts = [],
    supported_quant_modes = ["none"],
    supported_tile_shapes = [],
    truth_boundary = "handwritten_kernel_source_in_repo_dispatch_validated_not_benchmarked"
  }]
} {
  func.func @dtype_mismatch_rejected(%a: tensor<16x768xf16>)
      -> tensor<16x768xf16>
      attributes {representation.source_backend = "metal"} {
    %0 = "llm.rmsnorm"(%a) : (tensor<16x768xf16>) -> tensor<16x768xf16>
    return %0 : tensor<16x768xf16>
  }
}

// -----

// Test 3: layout mismatch is a rejection when both the descriptor and the
// op's layout decision state a layout (fixture descriptor: NHWC-only).
//
// CHECK-LABEL: func.func @layout_mismatch_rejected
// CHECK: "compute.conv"
// CHECK-SAME: kernel_selection.rejection_reasons = ["fixture_cpu_conv_nhwc:layout_unsupported"]
// CHECK-SAME: kernel_selection.status = "rejected_layout_unsupported"

module attributes {
  target.runtime_kernels = [{
    backend = "cpu",
    kernel_id = "fixture_cpu_conv_nhwc",
    op_name = "conv",
    requires_local_memory_bytes = 0 : i64,
    requires_static_shape = true,
    source = "fixture",
    supported_dtypes = ["fp32"],
    supported_layouts = ["NHWC"],
    supported_quant_modes = [],
    supported_tile_shapes = [],
    truth_boundary = "test_fixture_descriptor_not_a_real_kernel"
  }]
} {
  func.func @layout_mismatch_rejected(%a: tensor<1x3x8x8xf32>)
      -> tensor<1x3x8x8xf32>
      attributes {representation.source_backend = "cpu"} {
    %0 = "compute.conv"(%a) {
      layout.effective_layout = "NCHW"
    } : (tensor<1x3x8x8xf32>) -> tensor<1x3x8x8xf32>
    return %0 : tensor<1x3x8x8xf32>
  }
}

// -----

// Test 4: a tile-constrained fixture kernel. With local memory declared,
// tile planning selects [128,128,32], which the 64x64x32-only kernel
// rejects — tile constraints are matched against the real tile plan, not
// assumed.
//
// CHECK-LABEL: func.func @tile_shape_mismatch_rejected
// CHECK: "compute.matmul"
// CHECK-SAME: kernel_selection.rejection_reasons = ["fixture_cpu_matmul_64x64x32:tile_shape_unsupported"]
// CHECK-SAME: kernel_selection.status = "rejected_tile_shape_unsupported"

module attributes {
  target.static_cost_profile.local_memory_bytes = 65536 : i64,
  target.runtime_kernels = [{
    backend = "cpu",
    kernel_id = "fixture_cpu_matmul_64x64x32",
    op_name = "matmul",
    requires_local_memory_bytes = 0 : i64,
    requires_static_shape = true,
    source = "fixture",
    supported_dtypes = ["fp16"],
    supported_layouts = [],
    supported_quant_modes = [],
    supported_tile_shapes = ["64x64x32"],
    truth_boundary = "test_fixture_descriptor_not_a_real_kernel"
  }]
} {
  func.func @tile_shape_mismatch_rejected(%a: tensor<256x512xf16>)
      -> tensor<256x1024xf16>
      attributes {representation.source_backend = "cpu"} {
    %0 = "compute.matmul"(%a) : (tensor<256x512xf16>) -> tensor<256x1024xf16>
    return %0 : tensor<256x1024xf16>
  }
}

// -----

// Test 5: the same tile-constrained kernel with NO tile plan available
// (memory hierarchy not declared -> tile planning deferred) defers kernel
// selection — missing information is a deferral, not a rejection.
//
// CHECK-LABEL: func.func @missing_tile_plan_defers
// CHECK: "compute.matmul"
// CHECK-SAME: kernel_selection.rejection_reasons = ["fixture_cpu_matmul_64x64x32:missing_tile_plan"]
// CHECK-SAME: kernel_selection.status = "deferred_missing_tile_plan"

module attributes {
  target.runtime_kernels = [{
    backend = "cpu",
    kernel_id = "fixture_cpu_matmul_64x64x32",
    op_name = "matmul",
    requires_local_memory_bytes = 0 : i64,
    requires_static_shape = true,
    source = "fixture",
    supported_dtypes = ["fp16"],
    supported_layouts = [],
    supported_quant_modes = [],
    supported_tile_shapes = ["64x64x32"],
    truth_boundary = "test_fixture_descriptor_not_a_real_kernel"
  }]
} {
  func.func @missing_tile_plan_defers(%a: tensor<256x512xf16>)
      -> tensor<256x1024xf16>
      attributes {representation.source_backend = "cpu"} {
    %0 = "compute.matmul"(%a) : (tensor<256x512xf16>) -> tensor<256x1024xf16>
    return %0 : tensor<256x1024xf16>
  }
}

// -----

// Test 6: dynamic shapes defer selection for a kernel that requires static
// shapes — the static compiler cannot verify the contract, so it does not
// pretend to.
//
// CHECK-LABEL: func.func @dynamic_shape_defers
// CHECK: "llm.rmsnorm"
// CHECK-SAME: kernel_selection.rejection_reasons = ["metal_rmsnorm_f32_v1:dynamic_shape"]
// CHECK-SAME: kernel_selection.status = "deferred_dynamic_shape"

module attributes {
  target.runtime_kernels = [{
    backend = "metal",
    kernel_id = "metal_rmsnorm_f32_v1",
    op_name = "rmsnorm",
    requires_local_memory_bytes = 0 : i64,
    requires_static_shape = true,
    source = "handwritten_runtime",
    supported_dtypes = ["fp32"],
    supported_layouts = [],
    supported_quant_modes = ["none"],
    supported_tile_shapes = [],
    truth_boundary = "handwritten_kernel_source_in_repo_dispatch_validated_not_benchmarked"
  }]
} {
  func.func @dynamic_shape_defers(%a: tensor<?x768xf32>)
      -> tensor<?x768xf32>
      attributes {representation.source_backend = "metal"} {
    %0 = "llm.rmsnorm"(%a) : (tensor<?x768xf32>) -> tensor<?x768xf32>
    return %0 : tensor<?x768xf32>
  }
}

// -----

// Test 7: no runtime kernel registry declared -> every op records an
// explicit deferral. Missing declarations are never silent.
//
// CHECK-LABEL: func.func @no_registry_defers
// CHECK: "llm.rmsnorm"
// CHECK-SAME: kernel_selection.contract_version = "kernel_selection_contract_v1"
// CHECK-SAME: kernel_selection.status = "deferred_no_kernel_library_declared"
// CHECK-SAME: kernel_selection.truth_boundary = "kernel_selection_static_descriptor_match_not_runtime_execution"

module {
  func.func @no_registry_defers(%a: tensor<16x768xf32>)
      -> tensor<16x768xf32>
      attributes {representation.source_backend = "metal"} {
    %0 = "llm.rmsnorm"(%a) : (tensor<16x768xf32>) -> tensor<16x768xf32>
    return %0 : tensor<16x768xf32>
  }
}
