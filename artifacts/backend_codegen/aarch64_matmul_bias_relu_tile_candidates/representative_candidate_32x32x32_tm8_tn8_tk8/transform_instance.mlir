// tile_vectorize_matmul_bias_relu.template.mlir
//
// Parameterized Transform-dialect template for the tiled-vectorized AArch64
// backend-codegen variant. The M/N/K tile-size placeholder tokens below are
// substituted with concrete integers by
// mlir_passes/tools/generate_tiled_transform.sh before use -- this file is
// never fed to mlir-opt directly. See that script and
// mlir_passes/tools/compile_hir_matmul_bias_relu_aarch64.sh --variant
// tiled-vectorized --tile-m/--tile-n/--tile-k for how a concrete instance is
// produced and consumed at compile time (written to a scratch location, not
// committed -- see the tile-candidate slice's README for why seven
// near-identical committed Transform files were deliberately avoided).
//
// Structure (unchanged from the fixed TM=4/TN=8/TK=8 version this replaces):
//   1. tile_using_for the bias+relu linalg.generic (the final consumer) by
//      [M-tile, N-tile] -- outer M/N scf.for loops.
//   2. fuse_into_containing_op, twice: the linalg.matmul producer, then the
//      linalg.fill zero-init producer, into that loop nest.
//   3. A second tile_using_for on the now-fused matmul, tile size
//      [0, 0, K-tile] -- inner K-reduction scf.for loop.
//   4. vectorize_children_and_apply_patterns on the whole tiled func.func.
//
// __hoist_main is unused by the current compile pipeline (kept from the
// original file for parity/history) -- retained here unmodified.
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %fill = transform.structured.match ops{["linalg.fill"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    %matmul = transform.structured.match ops{["linalg.matmul"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    %generic = transform.structured.match ops{["linalg.generic"]} in %arg0 : (!transform.any_op) -> !transform.any_op

    %tiled_generic, %loop_m, %loop_n = transform.structured.tile_using_for %generic tile_sizes [8, 8]
        : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)

    %fused_matmul, %containing_1 = transform.structured.fuse_into_containing_op %matmul into %loop_n
        : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)

    %fused_fill, %containing_2 = transform.structured.fuse_into_containing_op %fill into %containing_1
        : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)

    %tiled_k_matmul, %loop_k = transform.structured.tile_using_for %fused_matmul tile_sizes [0, 0, 8]
        : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

    %func = transform.structured.match ops{["func.func"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func {
      transform.apply_patterns.linalg.tiling_canonicalization
      transform.apply_patterns.scf.for_loop_canonicalization
    } : !transform.any_op

    %vectorized = transform.structured.vectorize_children_and_apply_patterns %func
        : (!transform.any_op) -> !transform.any_op

    transform.yield
  }

  transform.named_sequence @__hoist_main(%arg0: !transform.any_op {transform.readonly}) {
    %func = transform.structured.match ops{["func.func"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    %hoisted = transform.structured.hoist_redundant_vector_transfers %func
        : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}
