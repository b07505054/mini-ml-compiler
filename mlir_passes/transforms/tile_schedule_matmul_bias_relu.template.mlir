// tile_schedule_matmul_bias_relu.template.mlir
//
// Parameterized Transform-dialect template for the "tiled-scheduled"
// AArch64 backend-codegen variant -- Stage 8/9/10 of the machine-scheduling
// analysis slice. Identical to tile_vectorize_matmul_bias_relu.template.mlir
// (the "tiled-vectorized" baseline this extends) with exactly ONE addition:
// a stock `transform.loop.unroll` applied to the K-reduction scf.for loop
// (Option A from the task brief: "K-loop unroll and interleave"), by a
// caller-supplied factor (SCHEDULE_UNROLL_K placeholder, substituted the
// same way M/N/K-tile placeholders are -- see
// mlir_passes/tools/generate_tiled_transform.sh). Factor 1 is a
// byte-for-byte no-op (verified: `transform.loop.unroll %loop { factor = 1
// }` degenerates to the identity, since "up to the given number of loop
// body copies per iteration" per the op's own docs never applies more than
// the requested factor and 1 copy is the un-unrolled loop itself).
//
// Why this transform and not Option B/C (see the artifact README's
// "Transformation Design Gate" section for the full evidence-based
// rationale): LLVM's default machine scheduler was measured (this slice)
// to ALREADY eliminate all real allocator spills and interleave the 16
// independent accumulator chains at 32x32x32/tile-8x8x8 with a median
// same-accumulator distance of 18 instructions (vs. FMLA's Cortex-A76
// llvm-mca-reported 10-cycle latency) -- i.e. LLVM is not starved for
// independent work to schedule. The lowest-risk, most evidence-motivated
// experiment is therefore to give the scheduler MORE material to work
// with per static loop body (halving the K-loop's dynamic trip count,
// which is the SAME mechanism the prior tile-selection slice already
// proved reduces per-iteration loop overhead) rather than hand-restructure
// the accumulator dependency graph itself (Option C), which would require
// custom IR construction beyond stock Transform-dialect combinators and
// was explicitly deprioritized as higher-risk for this slice's time
// budget -- see the Design Gate section for the full evaluation of all
// three options.
//
// Structure (all four original stages unchanged, ONE new stage 3.5):
//   1. tile_using_for the bias+relu linalg.generic (the final consumer) by
//      [M-tile, N-tile] -- outer M/N scf.for loops.
//   2. fuse_into_containing_op, twice: the linalg.matmul producer, then the
//      linalg.fill zero-init producer, into that loop nest.
//   3. A second tile_using_for on the now-fused matmul, tile size
//      [0, 0, K-tile] -- inner K-reduction scf.for loop.
//   3.5 [NEW] transform.loop.unroll the K-reduction loop by
//       schedule-unroll-k-factor copies per iteration.
//   4. vectorize_children_and_apply_patterns on the whole tiled func.func.
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %fill = transform.structured.match ops{["linalg.fill"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    %matmul = transform.structured.match ops{["linalg.matmul"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    %generic = transform.structured.match ops{["linalg.generic"]} in %arg0 : (!transform.any_op) -> !transform.any_op

    %tiled_generic, %loop_m, %loop_n = transform.structured.tile_using_for %generic tile_sizes [TILE_M, TILE_N]
        : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)

    %fused_matmul, %containing_1 = transform.structured.fuse_into_containing_op %matmul into %loop_n
        : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)

    %fused_fill, %containing_2 = transform.structured.fuse_into_containing_op %fill into %containing_1
        : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)

    %tiled_k_matmul, %loop_k = transform.structured.tile_using_for %fused_matmul tile_sizes [0, 0, TILE_K]
        : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

    transform.loop.unroll %loop_k { factor = SCHEDULE_UNROLL_K } : !transform.any_op

    %func = transform.structured.match ops{["func.func"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func {
      transform.apply_patterns.linalg.tiling_canonicalization
      transform.apply_patterns.scf.for_loop_canonicalization
    } : !transform.any_op

    %vectorized = transform.structured.vectorize_children_and_apply_patterns %func
        : (!transform.any_op) -> !transform.any_op

    transform.yield
  }
}
