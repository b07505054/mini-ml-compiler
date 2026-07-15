// FileCheck tests for TilePlanningPass (tile_planning_v1) and its
// conservative integration with shape_cost_model_v2.
//
// Run:
//   mlir-opt \
//     --split-input-file \
//     --allow-unregistered-dialect \
//     --load-pass-plugin="$PLUGIN" \
//     --pass-pipeline='builtin.module(tile-planning-pipeline,candidate-evaluation-pipeline)' \
//     tile_planning.mlir | \
//   FileCheck tile_planning.mlir --input-file=- --split-input-file
//
// All values are static feasibility/capacity computations from declared
// profile numbers — not measured performance, not DMA execution, not
// codegen, and no claim the backend kernel uses this tiling.

// -----

// Test 1: 256x512 -> 256x1024 fp16 matmul against 64 KB local memory.
// First menu tile (128,128,32) fits: A 8192 + B 8192 + C 32768 = 49152 B.
// Double-buffered staging 2x(A+B) + C = 65536 B fits exactly; the profile
// declares async-copy support so the fact is actionable. Reuse-limited
// traffic: A x ceil(1024/128) + B x ceil(256/128) + C
// = 2097152 + 2097152 + 524288 = 4718592 bytes.
// The elementwise op gets no tile attrs — matmul-like only in V1.
//
// CHECK-LABEL: func.func @tile_fits_local_memory
// CHECK: "compute.matmul"
// CHECK-SAME: tile.plan.double_buffer_fits = true
// CHECK-SAME: tile.plan.estimated_global_traffic_bytes = 4718592
// CHECK-SAME: tile.plan.local_memory_bytes = 49152
// CHECK-SAME: tile.plan.rejected_tile_count = 0
// CHECK-SAME: tile.plan.shape = [128, 128, 32]
// CHECK-SAME: tile.plan.staging_capability = "async_copy_declared"
// CHECK-SAME: tile.plan.status = "planned"
// CHECK-SAME: tile.plan.truth_boundary = "tile_planning_static_local_memory_model_not_measured_not_codegen"
// CHECK-SAME: memory_placement.compute_unit = "synthetic_accelerator"
// CHECK-SAME: memory_placement.selected_memory_space = "local_sram"
// CHECK-SAME: memory_placement.status = "selected"
// CHECK-SAME: memory_placement.single_buffer_bytes = 49152
// CHECK-SAME: memory_placement.additional_double_buffer_bytes = 16384
// CHECK-SAME: memory_placement.total_required_local_memory_bytes = 65536
// CHECK-SAME: memory_placement.transfer_operations
// CHECK-SAME: transfer_input_to_local
// CHECK-SAME: transfer_weight_to_local
// CHECK-SAME: transfer_output_to_host
// CHECK-SAME: compute_complete
// CHECK: "compute.relu"
// CHECK-NOT: tile.plan

module attributes {
  target.static_cost_profile.local_memory_bytes = 65536 : i64,
  target.static_cost_profile.supports_async_copy = true
} {
  func.func @tile_fits_local_memory(%a: tensor<256x512xf16>)
      -> tensor<256x1024xf16> {
    %0 = "compute.matmul"(%a) : (tensor<256x512xf16>) -> tensor<256x1024xf16>
    %1 = "compute.relu"(%0) : (tensor<256x1024xf16>) -> tensor<256x1024xf16>
    return %1 : tensor<256x1024xf16>
  }
}

// -----

// Test 2: same matmul against 1 KB local memory — every menu tile is
// rejected (smallest footprint 1536 B at (16,16,16) fp16). No tile is
// invented; the rejection and its reason are recorded.
//
// CHECK-LABEL: func.func @tile_rejected_local_memory_too_small
// CHECK: "compute.matmul"
// CHECK-SAME: tile.plan.rejected_tile_count = 4
// CHECK-SAME: tile.plan.rejection_reason = "smallest_tile_footprint_1536_bytes_exceeds_local_memory_1024_bytes"
// CHECK-SAME: tile.plan.status = "no_feasible_tile"
// CHECK-NOT: tile.plan.shape

module attributes {
  target.static_cost_profile.local_memory_bytes = 1024 : i64
} {
  func.func @tile_rejected_local_memory_too_small(%a: tensor<256x512xf16>)
      -> tensor<256x1024xf16> {
    %0 = "compute.matmul"(%a) : (tensor<256x512xf16>) -> tensor<256x1024xf16>
    return %0 : tensor<256x1024xf16>
  }
}

// -----

// Test 3: dynamic dims defer tile planning with an explicit status —
// never an estimated tile.
//
// CHECK-LABEL: func.func @dynamic_shape_defers_tile_planning
// CHECK: "compute.matmul"
// CHECK-SAME: tile.plan.status = "dynamic_dims_unresolved"
// CHECK-NOT: tile.plan.shape

module attributes {
  target.static_cost_profile.local_memory_bytes = 65536 : i64
} {
  func.func @dynamic_shape_defers_tile_planning(%a: tensor<?x512xf16>)
      -> tensor<?x1024xf16> {
    %0 = "compute.matmul"(%a) : (tensor<?x512xf16>) -> tensor<?x1024xf16>
    return %0 : tensor<?x1024xf16>
  }
}

// -----

// Test 4: MemoryHierarchyProfile is optional declared metadata. Without a
// declared local memory capacity, feasibility is DEFERRED and recorded —
// no capacity is invented, no tile is planned, and the reason is explicit
// so the exported plan stays valid and explains itself. The elementwise op
// still gets no attrs (op-kind gate precedes the memory gate).
//
// CHECK-LABEL: func.func @missing_memory_hierarchy_defers
// CHECK: "compute.matmul"
// CHECK-SAME: tile.plan.deferred_reason = "local_memory_bytes_not_declared_in_target_profile"
// CHECK-SAME: tile.plan.status = "deferred_missing_memory_hierarchy"
// CHECK-SAME: tile.plan.truth_boundary = "tile_planning_static_local_memory_model_not_measured_not_codegen"
// CHECK-SAME: memory_placement.compute_unit = "cpu"
// CHECK-SAME: memory_placement.selected_memory_space = "cpu_visible_host_memory"
// CHECK-SAME: memory_placement.status = "selected"
// CHECK-SAME: memory_placement.transfer_operations = []
// CHECK: "compute.relu"
// CHECK-NOT: tile.plan

module {
  func.func @missing_memory_hierarchy_defers(%a: tensor<256x512xf16>)
      -> tensor<256x1024xf16> {
    %0 = "compute.matmul"(%a) : (tensor<256x512xf16>) -> tensor<256x1024xf16>
    %1 = "compute.relu"(%0) : (tensor<256x1024xf16>) -> tensor<256x1024xf16>
    return %1 : tensor<256x1024xf16>
  }
}

// -----

// Test 5: existing weight-only quantization metadata shrinks the B tile.
// Local memory 40960 B rejects (128,128,32) for both ops; both settle on
// (64,64,32), but int8 weights (quant.weight_dtype) shrink the footprint
// from 16384 B (fp16 weights) to 14336 B.
//
// The module declares local memory but no async-copy/DMA information, so
// staging capability is honestly "unknown_not_declared" — unknown is not
// the same fact as unavailable.
//
// CHECK-LABEL: func.func @quant_weight_shrinks_tile_footprint
// CHECK: "compute.matmul"
// CHECK-SAME: tile.plan.local_memory_bytes = 16384
// CHECK-SAME: tile.plan.rejected_tile_count = 1
// CHECK-SAME: tile.plan.shape = [64, 64, 32]
// CHECK-SAME: tile.plan.staging_capability = "unknown_not_declared"
// CHECK: "compute.matmul"
// CHECK-SAME: tile.plan.local_memory_bytes = 14336
// CHECK-SAME: tile.plan.shape = [64, 64, 32]

module attributes {
  target.static_cost_profile.local_memory_bytes = 40960 : i64
} {
  func.func @quant_weight_shrinks_tile_footprint(%a: tensor<256x512xf16>)
      -> (tensor<256x1024xf16>, tensor<256x1024xf16>) {
    %0 = "compute.matmul"(%a) : (tensor<256x512xf16>) -> tensor<256x1024xf16>
    %1 = "compute.matmul"(%a) {
      quant.weight_dtype = "int8"
    } : (tensor<256x512xf16>) -> tensor<256x1024xf16>
    return %0, %1 : tensor<256x1024xf16>, tensor<256x1024xf16>
  }
}

// -----

// Test 6: cost-model integration is annotation-only. With bandwidth
// declared, the planned tile's reuse-limited traffic (4718592 B) becomes
// compiler.shape_profile.estimated_tiled_memory_cost_nanos = 47186,
// reported ALONGSIDE shape_cost_model_v2's ideal each-byte-once bytes
// (1835008) — and the V0 ranking score is untouched.
//
// The module declares supports_dma = false: staging capability is
// "declared_unavailable" — declared-and-false, distinct from not-declared.
//
// CHECK-LABEL: func.func @tile_traffic_annotated_ranking_unchanged
// CHECK: "compute.matmul"
// CHECK-SAME: evaluation.penalty_score = 0
// CHECK-SAME: evaluation.shape_cost.total_memory_bytes_estimate = 1835008
// CHECK-SAME: compiler.shape_profile.estimated_tiled_memory_cost_nanos = 47186
// CHECK-SAME: tile.plan.estimated_global_traffic_bytes = 4718592
// CHECK-SAME: tile.plan.staging_capability = "declared_unavailable"
// CHECK-SAME: tile.plan.status = "planned"

module attributes {
  target.static_cost_profile.local_memory_bytes = 65536 : i64,
  target.static_cost_profile.peak_flops_fp16 = 1.0e12 : f64,
  target.static_cost_profile.memory_bandwidth_bytes_per_sec = 1.0e11 : f64,
  target.static_cost_profile.supports_dma = false
} {
  func.func @tile_traffic_annotated_ranking_unchanged(%a: tensor<256x512xf16>)
      -> tensor<256x1024xf16> {
    %0 = "compute.matmul"(%a) {
      compiler.candidates = [{
        candidate_type = "direct_lower",
        required_boundary_ops = [],
        source_op = "matmul"
      }]
    } : (tensor<256x512xf16>) -> tensor<256x1024xf16>
    return %0 : tensor<256x1024xf16>
  }
}
