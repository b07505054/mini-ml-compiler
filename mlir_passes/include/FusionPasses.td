#ifndef HIR_FUSION_PASSES
#define HIR_FUSION_PASSES

include "mlir/Pass/PassBase.td"

def HIRCanonicalization
    : Pass<"hir-canonicalize", "mlir::func::FuncOp"> {
  let summary = "Canonicalize tensor-level HIR-friendly MLIR patterns";
  let description = [{
    Performs small canonical rewrites before fusion detection:

      linalg.map addf(x, 0.0) -> x
      linalg.map relu(linalg.map relu(x)) -> linalg.map relu(x)

    The pass is intentionally narrow and deterministic so downstream fusion
    passes can consume cleaner MLIR without relying only on descriptive
    artifact contracts.
  }];
  let constructor = "mlir::hir::createHIRCanonicalizationPass()";
}

def MatMulBiasReluFusion
    : Pass<"matmul-bias-relu-fusion", "mlir::func::FuncOp"> {
  let summary = "Detect MatMul + bias add + ReLU fusion candidates";
  let description = [{
    Detects the tensor-level MLIR pattern:

      linalg.matmul
        -> linalg.map arith.addf
        -> linalg.map arith.maximumf with zero

    The first version annotates linalg.matmul with:

      fusion.candidate = "matmul_bias_relu"

    This is intentionally a detect-and-annotate compiler pass before adding
    a custom fused op rewrite.
  }];
  let constructor = "mlir::hir::createMatMulBiasReluFusionPass()";
}

#endif // HIR_FUSION_PASSES
