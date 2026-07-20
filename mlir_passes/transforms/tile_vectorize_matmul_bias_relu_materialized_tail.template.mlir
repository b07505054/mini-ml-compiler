// Bounded tiled vector schedule with tile-local materialized tail padding.
//
// Unlike the whole-shape padding path, padding is applied only to the current
// BMxBK, BKxBN and BMxBN tile.  The copy-back operation writes only the
// original dynamic tile, so edge stores remain in bounds and no output crop is
// needed.  This is a bounded scalar-cleanup schedule, not a masked schedule:
// tensor.pad initialization/copy-back lower to small loops after bufferization.
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(
      %arg0: !transform.any_op {transform.readonly}) {
    %fill = transform.structured.match ops{["linalg.fill"]} in %arg0
        : (!transform.any_op) -> !transform.any_op
    %matmul = transform.structured.match ops{["linalg.matmul"]} in %arg0
        : (!transform.any_op) -> !transform.any_op
    %generic = transform.structured.match ops{["linalg.generic"]} in %arg0
        : (!transform.any_op) -> !transform.any_op

    %tiled_generic, %loop_m, %loop_n =
        transform.structured.tile_using_for %generic
            tile_sizes [TILE_M, TILE_N]
        : (!transform.any_op) ->
          (!transform.any_op, !transform.any_op, !transform.any_op)
    %fused_matmul, %containing_1 =
        transform.structured.fuse_into_containing_op %matmul into %loop_n
        : (!transform.any_op, !transform.any_op) ->
          (!transform.any_op, !transform.any_op)
    %fused_fill, %containing_2 =
        transform.structured.fuse_into_containing_op %fill into %containing_1
        : (!transform.any_op, !transform.any_op) ->
          (!transform.any_op, !transform.any_op)
    %tiled_k_matmul, %loop_k =
        transform.structured.tile_using_for %fused_matmul
            tile_sizes [0, 0, TILE_K]
        : (!transform.any_op) ->
          (!transform.any_op, !transform.any_op)

    %padded_matmul, %matmul_pads, %matmul_copy =
        transform.structured.pad %tiled_k_matmul
            pad_to_multiple_of [TILE_M, TILE_N, TILE_K]
            {padding_dimensions = [0, 1, 2]}
        : (!transform.any_op) ->
          (!transform.any_op, !transform.any_op, !transform.any_op)
    %padded_generic, %generic_pads, %generic_copy =
        transform.structured.pad %tiled_generic
            pad_to_multiple_of [TILE_M, TILE_N]
            {padding_dimensions = [0, 1]}
        : (!transform.any_op) ->
          (!transform.any_op, !transform.any_op, !transform.any_op)

    %func = transform.structured.match ops{["func.func"]} in %arg0
        : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func {
      transform.apply_patterns.linalg.tiling_canonicalization
      transform.apply_patterns.scf.for_loop_canonicalization
    } : !transform.any_op
    %vectorized =
        transform.structured.vectorize_children_and_apply_patterns %func
        : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}
