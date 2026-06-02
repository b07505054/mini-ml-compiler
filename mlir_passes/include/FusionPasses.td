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

def RMSNormKernelSelection
    : Pass<"rmsnorm-kernel-selection", "mlir::func::FuncOp"> {
  let summary = "Annotate RMSNorm ops for runtime-aware kernel selection";
  let description = [{
    Detects LLM RMSNorm ops and annotates them as compiler-visible fusion and
    lowering candidates:

      "llm.rmsnorm" -> hir.fused_rmsnorm

    The pass does not claim a kernel is faster by itself. Downstream lowering
    consumes runtime benchmark evidence to select a custom CUDA kernel or a
    PyTorch/runtime fallback.
  }];
  let constructor = "mlir::hir::createRMSNormKernelSelectionPass()";
}

def HIRFusionLowering
    : Pass<"hir-fusion-lowering", "mlir::func::FuncOp"> {
  let summary = "Lower compiler fusion candidates to HIR generic MLIR ops";
  let description = [{
    Lowers annotated fusion candidates into HIR-stage MLIR ops before JSON
    export:

      linalg.matmul + add + relu -> "hir.fused_matmul_bias_relu"
      "llm.rmsnorm"              -> "hir.fused_rmsnorm"

    These are emitted as generic MLIR ops so the compiler pipeline can carry a
    real lowering stage even before a full custom HIR dialect is introduced.
  }];
  let constructor = "mlir::hir::createHIRFusionLoweringPass()";
}

#endif // HIR_FUSION_PASSES
