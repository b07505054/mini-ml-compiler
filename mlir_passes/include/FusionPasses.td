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

def HIRFusedOpVerifier
    : Pass<"hir-verify-fused-ops", "mlir::func::FuncOp"> {
  let summary = "Verify invariants on generic HIR fused ops";
  let description = [{
    Checks the generic HIR ops emitted by the fusion lowering pipeline before
    they are exported to runtime-facing artifacts. This gives the pass pipeline
    an explicit verifier stage even before the generic HIR ops are promoted to
    a fully registered dialect.
  }];
  let constructor = "mlir::hir::createHIRFusedOpVerifierPass()";
}

def HIRRMSNormToLinalg
    : Pass<"hir-rmsnorm-to-linalg", "mlir::func::FuncOp"> {
  let summary = "Lower hir.fused_rmsnorm to executable Linalg/Math tensor IR";
  let description = [{
    Lowers the runtime-facing HIR RMSNorm op into MLIR-native tensor code:

      hir.fused_rmsnorm
        -> linalg.generic row sum-of-squares reduction
        -> linalg.generic elementwise x * math.rsqrt(mean + epsilon)

    This is the first codegen-oriented backend lowering stage after HIR. The
    output is designed to be consumed by MLIR bufferization and LLVM lowering
    pipelines rather than by the JSON runtime bridge.
  }];
  let constructor = "mlir::hir::createHIRRMSNormToLinalgPass()";
}

def HIRMatMulBiasReluToLinalg
    : Pass<"hir-matmul-bias-relu-to-linalg", "mlir::func::FuncOp"> {
  let summary = "Lower hir.fused_matmul_bias_relu to executable Linalg tensor IR";
  let description = [{
    Lowers the runtime-facing HIR MatMul-Bias-ReLU op into MLIR-native tensor
    code:

      hir.fused_matmul_bias_relu
        -> linalg.matmul
        -> linalg.generic arith.addf + arith.maximumf

    This gives the fused HIR op a native backend lowering path that can be
    consumed by bufferization, loop conversion, and LLVM lowering.
  }];
  let constructor = "mlir::hir::createHIRMatMulBiasReluToLinalgPass()";
}

def StableHLOCompatibleRMSNormImport
    : Pass<"stablehlo-compatible-rmsnorm-import", "mlir::func::FuncOp"> {
  let summary = "Import StableHLO-compatible RMSNorm decomposition into HIR";
  let description = [{
    Recognizes the standard frontend decomposition shape for RMSNorm after it
    has been represented in Linalg/Arith/Math form:

      row sum-of-squares reduction
      x * math.rsqrt(sum / hidden + epsilon)

    The pass rewrites the final normalize operation to `hir.fused_rmsnorm`.
    It is intentionally narrow and CI-friendly until native StableHLO tooling
    is installed.
  }];
  let constructor = "mlir::hir::createStableHLOCompatibleRMSNormImportPass()";
}

#endif // HIR_FUSION_PASSES
