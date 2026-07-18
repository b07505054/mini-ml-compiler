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

def ServingPhaseAnalysis
    : Pass<"serving-phase-analysis", "mlir::func::FuncOp"> {
  let summary = "Annotate serving functions with colocated vs pd_split policy decisions";
  let description = [{
    Walks llm.attention_prefill and llm.attention_decode ops within a func.func,
    reads serving.phase, kv_cache.role, serving.prompt_tokens, and
    serving.output_tokens attributes, computes formula-synthetic colocated vs
    pd_split serving costs, and annotates the enclosing func.func with:

      serving.policy            -- "colocated" or "pd_split"
      serving.colocated_total_ms
      serving.pd_split_total_ms
      serving.decision_margin_ms
      serving.decision_margin_pct
      serving.confidence        -- "low", "medium", or "high"
      serving.cost_source       = "formula_synthetic"
      serving.truth_boundary    = "estimated_cost_not_measured_latency"

    Truth boundary: cost values are formula estimates, not measured latency.
    This pass does not execute CUDA Graphs, vLLM, or real KV transfers.

    Future split: this pass combines phase detection, cost computation, and
    policy decision. It will eventually be split into ServingPhaseAnalysisPass,
    DistributedServingCostAnalysisPass, and ServingDecisionPass when multi-source
    cost merging is required.
  }];
  let constructor = "mlir::hir::createServingPhaseAnalysisPass()";
}

def KVLayoutPlanning
    : Pass<"kv-layout-planning", "mlir::func::FuncOp"> {
  let summary = "Annotate serving functions with KV cache layout policy and byte estimate";
  let description = [{
    Walks llm.attention_prefill and llm.attention_decode ops within a func.func,
    reads kv_cache.role, serving.prompt_tokens, serving.output_tokens, and
    module attrs llm.num_layers and llm.hidden_size, then annotates the
    enclosing func.func with:

      kv.layout            -- "paged" if kv_cache.role == "producer", else "contiguous"
      kv.byte_estimate_mb  -- formula: num_layers * 2 * hidden_size * 2 * seq_tokens / 1MB
      kv.truth_boundary    = "static_formula_estimate_not_measured_memory"

    Truth boundary: layout is a compiler policy decision matching the vLLM
    paged attention model; byte estimate is a static formula estimate and has
    not been measured against actual KV cache memory usage.

    Intended to run after ServingPhaseAnalysisPass in serving-optimization-pipeline.
  }];
  let constructor = "mlir::hir::createKVLayoutPlanningPass()";
}

def ReplayEligibility
    : Pass<"replay-eligibility", "mlir::func::FuncOp"> {
  let summary = "Annotate serving functions with CUDA graph replay eligibility metadata";
  let description = [{
    Inspects the function body for attention op type (prefill or decode) and
    checks whether any tensor type in the function has dynamic dimensions.
    Annotates the enclosing func.func with:

      replay.eligible          -- true if function is a replay candidate
      replay.cuda_graph_bucket -- "decode_static" if eligible, else ""
      replay.truth_boundary    = "static_shape_replay_eligibility_not_cuda_graph_capture"

    Eligibility rules (applied in order):
      - llm.attention_prefill present           -> not eligible (variable-length)
      - llm.attention_decode + dynamic tensor   -> not eligible
      - llm.attention_decode + all-static shape -> eligible, bucket "decode_static"
      - no attention op                         -> pass skips the function

    Truth boundary: this pass emits compiler-side eligibility metadata only.
    It does not capture CUDA Graphs, measure capture latency, or validate
    runtime replay correctness. The runtime decides whether to actually
    capture and replay based on this annotation.

    Intended to run after KVLayoutPlanningPass in serving-optimization-pipeline.
  }];
  let constructor = "mlir::hir::createReplayEligibilityPass()";
}

def ExecutionProviderPlanning
    : Pass<"execution-provider-planning", "mlir::func::FuncOp"> {
  let summary = "Produce an execution provider plan for serving functions based on target constraints";
  let description = [{
    Consumes target.* module attributes (produced by a TargetLowering step
    outside compiler core) and prior-pass func.func attributes (kv.layout,
    replay.eligible) to emit a compiler-produced execution provider plan:

      execution_provider.primary          -- primary backend: "coreml", "cuda", etc.
      execution_provider.fallback_chain   -- ordered fallback providers (ArrayAttr)
      execution_provider.decision_source  -- closed-set tag explaining the decision
      execution_provider.required_precision -- first supported precision or "fp16"
      execution_provider.required_kv_layout -- copy of kv.layout or "unknown"
      execution_provider.requires_replay  -- copy of replay.eligible or false
      execution_provider.truth_boundary   -- always "compiler_execution_provider_plan_not_runtime_dispatch"

    Decision logic (strict priority order):
      1. candidate list = target.allowed_backends; absent/empty -> ["cpu"], default_policy
      2. if kv.layout == "paged": filter candidates to paged-compatible set (cuda, vllm);
         empty result -> constraint_conflict, primary = "cpu"
      3. if preferred in filtered candidates -> target_preferred
         if preferred present but filtered out -> kv_layout_constraint
         else -> allowed_backends_default (or keep default_policy)
      4. fallback_chain = remaining filtered candidates in order, excluding primary

    Truth boundary: the pass emits a compiler-derived static plan. Runtime is
    responsible for dynamic dispatch, availability checks, and thermal overrides.

    Intended to run after ReplayEligibilityPass in serving-optimization-pipeline.
  }];
  let constructor = "mlir::hir::createExecutionProviderPlanningPass()";
}

def RepresentationPlanning
    : Pass<"representation-planning", "mlir::func::FuncOp"> {
  let summary = "Planning-only: annotate func with effective dtype and layout from backend capabilities";
  let description = [{
    Consumes execution_provider.primary (from ExecutionProviderPlanningPass),
    quantization.plan_dtype (module-level), and
    target.backend_capabilities.{backend}.* module attrs to emit
    planning-only representation metadata on each func.func:

      representation.effective_dtype             -- dtype to use for this func
      representation.effective_dtype_source      -- "target_profile", "fallback_first_supported", etc.
      representation.preferred_activation_layout -- preferred_activation_layouts[0] or "unknown"
      representation.preferred_weight_layout     -- preferred_weight_layouts[0] or "unknown"
      representation.source_backend              -- backend name from execution_provider.primary
      representation.truth_boundary              -- always "representation_plan_static_not_validated_against_kernel_performance"

    When the plan dtype is not supported by the backend, also emits:
      representation.conflict_reason             -- "backend_lacks_plan_dtype" or "backend_capability_missing"
      representation.conflict_fallback_dtype     -- the dtype chosen instead (dtype-conflict path only)

    Decision rules (in priority order):
      1. execution_provider.primary absent -> skip this func (no annotation).
      2. Backend not in target.backend_capability_names ->
           effective_dtype = quantization.plan_dtype (or "fp16"), layouts = "unknown",
           conflict_reason = "backend_capability_missing".
      3. plan_dtype in supported_dtypes ->
           effective_dtype = plan_dtype, source = "target_profile".
      4. plan_dtype not in supported_dtypes ->
           effective_dtype = supported_dtypes[0], source = "fallback_first_supported",
           conflict_reason = "backend_lacks_plan_dtype".

    This pass does NOT insert cast, dequant, layout_transform, or any ops.
    It is annotation-only and must run after ExecutionProviderPlanningPass.

    Truth boundary: representation_plan_static_not_validated_against_kernel_performance.

    Intended to run after ExecutionProviderPlanningPass in serving-optimization-pipeline.
  }];
  let constructor = "mlir::hir::createRepresentationPlanningPass()";
}

def LayoutPlanning
    : Pass<"layout-planning", "mlir::func::FuncOp"> {
  let summary = "Planning-only: annotate ops with layout assignment from backend capabilities";
  let description = [{
    Reads representation.preferred_activation_layout and representation.source_backend
    from func.func (set by RepresentationPlanningPass), and
    target.backend_capabilities.{backend}.layout_agnostic_ops from the module,
    to emit planning-only layout metadata on each non-terminator op in the
    function body entry block.

    Emitted per non-terminator op:
      layout.required_input_layout  -- layout this op requires its activation inputs to be in
      layout.effective_layout       -- layout of this op's activation output
      layout.layout_source          -- "backend_preference" or "layout_agnostic_propagation"
      layout.transform_required     -- true only at the first layout-changing boundary
      layout.truth_boundary         -- always "layout_planning_static_cost_model_not_measured_kernel_performance"

    Layout propagation rules (applied in entry-block op order):
      1. Start with layout.initial_layout from func (default "NCHW").
      2. For each non-terminator op:
         - Extract short op name (everything after the last '.').
         - If short name is in layout_agnostic_ops:
             effective_layout = current_layout, transform_required = false,
             layout_source = "layout_agnostic_propagation"
         - Otherwise:
             effective_layout = preferred_activation_layout, transform_required = (current_layout != effective_layout),
             layout_source = "backend_preference"
         - current_layout = effective_layout after each op.

    Also reads representation.preferred_weight_layout (available for future extension;
    weight layout planning is out of scope for this pass).

    This pass does NOT insert hir.layout_transform ops or modify the graph.
    It is annotation-only and must run after RepresentationPlanningPass.
    Only the entry block is processed; multi-block functions are not yet supported.

    Truth boundary: layout_planning_static_cost_model_not_measured_kernel_performance.

    Intended to run after RepresentationPlanningPass in serving-optimization-pipeline.
  }];
  let constructor = "mlir::hir::createLayoutPlanningPass()";
}

def WeightClassificationPlanning
    : Pass<"weight-classification-planning", "mlir::func::FuncOp"> {
  let summary = "Classify weight operands as constant or runtime before quantization";
  let description = [{
    Walks every quantizable op (matmul, conv, gemm, linear families) in the entry
    block and classifies its weight operand (RHS / operand 1) via def-use analysis.

    Classification values:
      constant_weight    - defined by arith.constant, ConstantLike op, or
                           an op/op-attr explicitly declaring weight.is_constant=true
      runtime_activation - defined by a function argument, a compute op (hir.*, llm.*,
                           cv.*), or any arith.* op that is not a constant
      unknown            - defining op not recognized as constant or runtime;
                           conservatively treated as non-constant

    Rules:
      1. Function-level representation.weights_are_constant=true overrides all → constant_weight.
      2. Quantizable op has weight.is_constant=true attr → constant_weight.
      3. Weight operand is a BlockArgument (func arg) → runtime_activation.
      4. Defining op has weight.is_constant=true attr → constant_weight.
      5. Defining op is arith.constant or has ConstantLike trait → constant_weight.
      6. Defining op name starts with hir., llm., cv., arith. (non-const), func. → runtime_activation.
      7. Otherwise → unknown (conservative: unknown ≠ constant).

    Emitted on each quantizable op:
      weight.classification          = "constant_weight" | "runtime_activation" | "unknown"
      weight.constant_required       = true
      weight.constant_satisfied      = true iff classification == constant_weight
      weight.classification_reason   = short reason string
      weight.truth_boundary          = "weight_classification_static_def_use_analysis_not_runtime_validated"

    This pass does NOT modify the graph, insert quantize/dequantize ops,
    or change any existing attributes.

    Truth boundary: weight_classification_static_def_use_analysis_not_runtime_validated.
    Must run after BoundaryPlanningPass and before QuantizationStrategyPlanningPass.
  }];
  let constructor = "mlir::hir::createWeightClassificationPlanningPass()";
  let dependentDialects = ["mlir::func::FuncDialect"];
}

def QuantizationStrategyPlanning
    : Pass<"quantization-strategy-planning", "mlir::func::FuncOp"> {
  let summary = "Per-op quantization strategy annotation for the HIR serving pipeline";
  let description = [{
    Reads target.backend_capabilities.{backend}.*, representation.*, boundary.*,
    quantization.plan_dtype (module-level), and op type/shape/constant-weight
    information. For each quantizable or accuracy-sensitive op in the entry block,
    emits planning-only quant.* strategy metadata.

    Quantizable ops: matmul, conv, linear, dense, gemm, dot, fc families.
    Accuracy-sensitive ops: softmax, norm, rmsnorm, layernorm, batchnorm,
    groupnorm, embedding — always fall back to fp16 with accuracy_risk="medium".

    Strategy selection (first matching rule wins):
      1. Accuracy-sensitive op         -> fp16_fallback, fallback_reason=accuracy_sensitive_op
      2. weights_are_constant=false    -> fp16_fallback, fallback_reason=requires_constant_weight
      3. Backend supports int8 weight  -> weight_only_int8
           weight_dtype=int8, activation_dtype=fp16 (Rule 1: prefer weight-only)
           accumulation_dtype from backend accumulation_dtypes or fp32 (Rule 4)
      4. Backend lacks all int8        -> fp16_fallback, fallback_reason=backend_lacks_activation_int8

    Rule 6: op result type is quantized and func effective_dtype is float
            -> requires_dequant_boundary=true.
    Rule 7: requires_requant_boundary not yet detectable statically -> always false.

    Emitted per quantizable/accuracy-sensitive op:
      quant.strategy, quant.weight_dtype, quant.activation_dtype,
      quant.accumulation_dtype, quant.output_dtype, quant.granularity,
      quant.activation_quant_mode, quant.weight_quant_mode,
      quant.requires_dequant_boundary, quant.requires_requant_boundary,
      quant.backend, quant.decision_reason, quant.fallback_reason,
      quant.accuracy_risk, quant.truth_boundary

    This pass does NOT insert hir.quantize/dequantize/requantize ops.
    No calibration data is used or generated.
    Absent calibration -> accuracy_risk="unknown" for quantized, "medium" for
    accuracy-sensitive, "low" for fp16 fallback.

    Truth boundary: quantization_strategy_static_not_accuracy_calibrated.
    Must run after RepresentationPlanningPass and BoundaryPlanningPass.
  }];
  let constructor = "mlir::hir::createQuantizationStrategyPlanningPass()";
  let dependentDialects = ["mlir::func::FuncDialect"];
}

def KernelAvailabilityPlanning
    : Pass<"kernel-availability-planning", "mlir::func::FuncOp"> {
  let summary = "Per-op kernel availability annotation from the KernelLibraryCapability schema";
  let description = [{
    Reads target.kernel_libraries.{backend} (ArrayAttr of DictionaryAttr) from
    the module, and quant.*, layout.*, representation.* attrs from the func and
    ops. For each non-terminator op in the entry block, matches against the
    declared kernel library and emits:

      kernel.backend             -- backend used for matching
      kernel.decision_reason     -- match outcome tag
      kernel.exists              -- true if an exact kernel was found
      kernel.fallback_backend    -- declared fallback backend (may be empty)
      kernel.library             -- kernel_library name (empty if no exact match)
      kernel.lowering_status     -- lowerable | rewrite_candidate | fallback_required | unsupported
      kernel.name                -- kernel_name (empty if no exact match)
      kernel.required_rewrite    -- rewrite pattern name (empty if not rewrite_candidate)
      kernel.truth_boundary      -- from matched entry or kTruth constant

    Matching rules (first match wins):
      1. Exact match: op_type + dtype + layout + quant_mode all match (or list empty)
         -> lowerable.
      2. Op type found, no exact match, rewrite_patterns non-empty
         -> rewrite_candidate.
      3. Op type found, no exact match, fallback_backend declared
         -> fallback_required.
      4. Op type found, no exact match, no rewrite or fallback
         -> unsupported.
      5. Op type not found in kernel library at all
         -> unsupported.

    This pass does NOT rewrite IR, lower code, or insert any ops.
    Ops without a kernel entry are annotated as unsupported.
    Ops in modules with no target.kernel_libraries.{backend} are skipped.

    Truth boundary: kernel_availability_static_library_model_not_runtime_verified.
    Must run after RepresentationPlanningPass, LayoutPlanningPass, and
    QuantizationStrategyPlanningPass.
  }];
  let constructor = "mlir::hir::createKernelAvailabilityPlanningPass()";
  let dependentDialects = ["mlir::func::FuncDialect"];
}

def LoweringDecisionPlanning
    : Pass<"lowering-decision-planning", "mlir::func::FuncOp"> {
  let summary = "Per-op final lowering decision from kernel.* and boundary.* attrs";
  let description = [{
    Consumes kernel.* attrs (from KernelAvailabilityPlanningPass) and
    boundary.* attrs (from BoundaryPlanningPass) to produce the final
    per-op lowering decision. Reads but does not modify kernel.* or boundary.*
    attrs. Does not rewrite IR, insert ops, or lower code.

    Inputs per op:
      kernel.lowering_status  -- lowerable | rewrite_candidate | fallback_required | unsupported
      kernel.required_rewrite -- rewrite pattern name (if rewrite_candidate)
      kernel.fallback_backend -- declared fallback backend (if fallback_required)
      kernel.library          -- kernel library name
      kernel.name             -- kernel name
      kernel.backend          -- primary backend
      boundary.dequant_required         -- true when dequant boundary is needed
      boundary.cast_required            -- true when precision cast is needed
      boundary.layout_transform_required -- true when layout transform is needed

    Decision rules (first matching rule wins):
      1. kernel.lowering_status = lowerable
         -> lowering.decision = direct_lower
      2. kernel.lowering_status = rewrite_candidate
         -> lowering.decision = rewrite_then_lower
      3. kernel.lowering_status = fallback_required AND boundary.dequant_required = true
         -> lowering.decision = dequant_then_lower
      4. kernel.lowering_status = fallback_required AND kernel.fallback_backend non-empty
         -> lowering.decision = fallback_backend
      5. No match (unsupported or absent kernel status)
         -> lowering.decision = unsupported

    Emitted per non-terminator op (only when kernel.lowering_status is present):
      lowering.decision             -- direct_lower | rewrite_then_lower | dequant_then_lower
                                       | fallback_backend | unsupported
      lowering.reason               -- human-readable decision tag
      lowering.required_rewrite     -- rewrite pattern (rewrite_then_lower only, else "")
      lowering.requires_cast        -- from boundary.cast_required
      lowering.requires_dequant     -- from boundary.dequant_required
      lowering.requires_layout_transform -- from boundary.layout_transform_required
      lowering.target_backend       -- backend selected for lowering
      lowering.target_kernel        -- kernel name (direct/rewrite) or "" (fallback/unsupported)
      lowering.target_kernel_library -- kernel library (direct/rewrite) or ""
      lowering.truth_boundary       -- lowering_decision_static_not_backend_codegen_verified

    Ops without kernel.lowering_status are skipped (KernelAvailabilityPlanningPass did not run).
    Truth boundary: lowering_decision_static_not_backend_codegen_verified.
    Must run after KernelAvailabilityPlanningPass and BoundaryPlanningPass.
  }];
  let constructor = "mlir::hir::createLoweringDecisionPlanningPass()";
  let dependentDialects = ["mlir::func::FuncDialect"];
}

def QuantizedBoundaryRefinement
    : Pass<"quantized-boundary-refinement", "mlir::func::FuncOp"> {
  let summary = "Refine per-op weight dequant boundary requirements after lowering decisions are known";
  let description = [{
    Runs after LoweringDecisionPlanningPass. For each non-terminator op, reads
    quant.strategy and lowering.decision to determine whether the quantized
    weight dtype (e.g. int8 from weight_only_int8) is compatible with the
    effective execution backend. When the op falls back to a backend that does
    not support the quantized weight format, the pass marks the op as requiring
    a weight dequantization boundary.

    This pass does NOT insert hir.dequantize ops or modify the graph.

    Inputs per op (from upstream passes):
      quant.strategy          -- from QuantizationStrategyPlanningPass
      quant.weight_dtype      -- from QuantizationStrategyPlanningPass
      lowering.decision       -- from LoweringDecisionPlanningPass
      lowering.target_backend -- from LoweringDecisionPlanningPass
      kernel.fallback_backend -- from KernelAvailabilityPlanningPass (fallback)

    Outputs per non-terminator op:
      boundary.fallback_backend_supports_weight_dtype
      boundary.quantized_weight_dtype
      boundary.weight_dequant_reason
      boundary.weight_dequant_required
      boundary.weight_dtype_mismatch
      boundary.weight_truth_boundary

    Decision rules:
      1. quant.strategy != weight_only_int8
         -> weight_dequant_required=false, reason=not_weight_quantized
      2. quant.strategy == weight_only_int8, lowering.decision absent
         -> weight_dequant_required=false, reason=target_backend_unknown
      3. quant.strategy == weight_only_int8, lowering.decision=direct_lower
         -> weight_dequant_required=false,
            reason=kernel_exact_match_backend_supports_weight_quantization
      4. quant.strategy == weight_only_int8, fallback/rewrite/unsupported,
         effective backend supported_quant_modes lacks weight_only/static_int8
         -> weight_dequant_required=true, weight_dtype_mismatch=true,
            reason=fallback_backend_lacks_quantized_weight_support
      5. As rule 4 but backend supports weight_only or static_int8
         -> weight_dequant_required=false, reason=backend_supports_weight_quantization

    Effective backend determination: lowering.target_backend → kernel.fallback_backend
    → representation.source_backend. Absent → target_backend_unknown.

    Supported_quant_modes semantics: consistent with QuantizationStrategyPlanningPass.
    Empty list = backend has no declared quant support.

    Truth boundary: quantized_boundary_refinement_static_not_materialized.
    Must run after LoweringDecisionPlanningPass and QuantizationStrategyPlanningPass.
  }];
  let constructor = "mlir::hir::createQuantizedBoundaryRefinementPass()";
  let dependentDialects = ["mlir::func::FuncDialect"];
}

def AlternativeLoweringPlanning
    : Pass<"alternative-lowering-planning", "mlir::func::FuncOp"> {
  let summary = "Generate alternative legal lowering candidates when exact kernel lowering is unavailable";
  let description = [{
    Runs after QuantizedBoundaryRefinementPass. For each non-terminator op where
    the exact kernel match is absent (kernel.exists=false), generates alternative
    lowering candidates: algebraic decomposition, representation conversion
    (weight_only_int8 → dequant_weight + fp16 op), layout conversion
    (layout_transform + same op), cast conversion, and backend fallback.

    Each candidate is validated: a candidate is valid only if every target op has
    a matching kernel in the library or the required boundary conversion is
    supported by the backend.

    This pass does NOT select a best candidate, emit cost, or materialize IR.

    Outputs per non-terminator op (when kernel.lowering_status is present):
      alternative.available      -- true iff at least one candidate is valid
      alternative.best_candidate -- always "" (no cost model, no selection)
      alternative.candidates     -- ArrayAttr of DictionaryAttr, one per candidate
      alternative.reason         -- summary string
      alternative.truth_boundary -- always kTruth

    Each candidate DictionaryAttr contains:
      alternative_id, alternative_type, source_op, target_ops,
      required_boundary_ops, required_kernel_checks,
      validation_status (valid/invalid_missing_kernel/invalid_missing_boundary_support/unknown),
      validation_failures, risk, source_level, truth_boundary.

    Truth boundary: alternative_lowering_static_not_materialized_not_cost_evaluated.
    Must run after QuantizedBoundaryRefinementPass.
  }];
  let constructor = "mlir::hir::createAlternativeLoweringPlanningPass()";
  let dependentDialects = ["mlir::func::FuncDialect"];
}

def CandidateGeneration
    : Pass<"candidate-generation", "mlir::func::FuncOp"> {
  let summary = "Generate execution candidates from expanded BackendCapability schema";
  let description = [{
    Reads target.backend_capabilities.{backend}.* from the module and
    representation.* / layout.* / boundary.* from the func and ops.
    For each backend, generates dtype variants (preferred → acceptable →
    supported, deduplicated), layout variants (preferred → acceptable,
    deduplicated), and selects primary quant_mode and fusion_pattern.

    Each candidate captures:
      candidate_id, backend, backend_api, dtype, accumulation_dtype,
      activation_layout, weight_layout, quant_mode, fusion_pattern,
      fallback_backend, requires_cast, requires_dequant,
      requires_layout_transform, constraint_status, failed_constraints,
      source_level, truth_boundary.

    Level 2 constraint checks applied (fail → mark, do NOT discard):
      requires_static_shape   — fail when func has dynamic tensor shapes
      requires_constant_weight — fail when representation.weights_are_constant = false
      unsupported_ops          — fail when any op in func body matches
      required_activation_quant_mode / required_weight_quant_mode
      allowed_quant_granularity (recorded; not yet enforced)

    Cost is NOT evaluated. No winner is chosen. No IR is rewritten.

    Output: func.func attribute
      candidates = ArrayAttr of DictionaryAttr, one entry per candidate
      candidates.count = total candidate count
      candidates.truth_boundary = "candidate_generation_static_constraints_not_cost_evaluated"

    Truth boundary: candidate_generation_static_constraints_not_cost_evaluated
  }];
  let constructor = "mlir::hir::createCandidateGenerationPass()";
  let dependentDialects = ["mlir::func::FuncDialect"];
}

// Registration name kept as "candidate-evaluation" for backward compatibility
// with FileCheck tests and pipeline strings. The C++ implementation is
// ServingCostModelPass (ServingCostModelPass.cpp), which applies
// ServingStaticCostModel_v1. The factory createCandidateEvaluationPass() is
// an inline alias for createServingCostModelPass().
def CandidateEvaluation
    : Pass<"candidate-evaluation", "mlir::func::FuncOp"> {
  let summary = "Apply ServingStaticCostModel_v1 to compiler.candidates (upgraded from V0 penalty model)";
  let description = [{
    Applies ServingStaticCostModel_v1 to each compiler.candidates candidate dict
    produced by CandidateGenerationPass. Emits two layers of output per candidate:

    Layer 1 — V0 backward-compat (ranking and FileCheck tests):
      evaluation.status          = evaluated | partially_evaluated | rejected
      evaluation.penalty_score   = V0 penalty formula (used by PlanSelectionPass ranking)
      evaluation.reason          = human-readable dominant penalty source
      evaluation.truth_boundary  = "candidate_evaluation_static_penalty_not_measured_latency"

    Layer 2 — V1 structured cost evidence (new):
      evaluation.cost.compute, .memory, .dequant, .requant, .layout_transform,
      .cast, .backend_switch, .launch_overhead, .kv_cache, .transfer, .unsupported,
      .total, .model_id, .truth_boundary
      where total == exact sum of all component fields.

    V1 component model (StaticCostWeights defaults, declared_profile not measured_latency):
      direct_lower:              compute=0, boundary costs, launch_overhead per boundary op
      algebraic_decomposition:   compute=5, boundary costs
      representation_conversion: memory=3, boundary costs (dequant=3, requant=2, cast=2, layout=2)
      layout_conversion:         layout_transform=4 (standalone)
      cast_conversion:           cast=2 (standalone)
      backend_fallback:          backend_switch=20, launch_overhead=2, transfer=5 if fallback_backend set
      unsupported:               unsupported_penalty=100 (compile-time rejection sentinel)

    Cost weights are tunable via StaticCostWeights in DeploymentTarget (default values above).

    Layer 3 — V2 shape-aware cost evidence (shape_cost_model_v2, see
    ShapeCostModel.h): for supported op kinds with fully static shapes,
    per-candidate
      evaluation.shape_cost.flops_estimate, .input_bytes_estimate,
      .output_bytes_estimate, .weight_bytes_estimate,
      .total_memory_bytes_estimate, .arithmetic_intensity_milli,
      .estimated_{compute,memory,boundary,total}_cost_nanos (only when the
      module declares target.static_cost_profile.* peak numbers),
      .status, .model_version, .truth_boundary
    Op kinds: matmul_like (2*M*K*N flops, inferred K*N weight),
    normalization (4 ops/elem), elementwise (1 op/elem); unknown op kinds and
    dynamic shapes fall back to the V1 model, recorded in the op-level
    compiler.shape_profile.{op_kind, status, ranking_mode, truth_boundary}
    attrs. Bytes are dtype-aware (fp32/fp16/bf16/int8/int4) using candidate
    dtype -> quant.activation_dtype -> result element type ->
    representation.effective_dtype, and quant.weight_dtype for weights.
    Time estimates use the roofline form
    max(flops/peak, bytes/bandwidth) + boundary_bytes/bandwidth in integer
    nanoseconds — static estimates from declared theoretical peaks, never
    measured, never a runtime latency guarantee.

    Ranking: evaluation.penalty_score keeps the V0 heuristic by default.
    Opt-in module attr serving.cost_model.mode = "shape_aware_v2" ranks an
    op's evaluated candidates by estimated_total_cost_nanos instead — only
    when every evaluated candidate of that op has a time estimate (V0 is
    preserved as evaluation.penalty_score_v0). PlanSelection tier rules
    (fallback last resort, unsupported never) are unaffected.

    No winner is selected. No IR is rewritten. No measured latency is claimed.

    truth_boundary (V2 shape cost):
      static_shape_derived_declared_profile_not_measured_not_runtime_validated
    truth_boundary (V1 cost): serving_static_cost_model_v1_not_measured_latency
    truth_boundary (V0 compat): candidate_evaluation_static_penalty_not_measured_latency
  }];
  let constructor = "mlir::hir::createCandidateEvaluationPass()";
  let dependentDialects = ["mlir::func::FuncDialect"];
}

def PlanSelection
    : Pass<"plan-selection", "mlir::func::FuncOp"> {
  let summary = "Select the best evaluated candidate per op using static penalty scores";
  let description = [{
    Reads compiler.evaluated_candidates (per-op) from ServingCostModelPass and
    selects the lowest-penalty candidate with evaluation.status=evaluated.

    Selection rules (applied in order):
      1. Prefer evaluated (non-fallback) candidates by lowest penalty score.
      2. Tiebreak: direct_lower > representation_conversion > cast_conversion >
         layout_conversion > algebraic_decomposition > backend_fallback > unsupported.
      3. backend_fallback is selected only when no evaluated non-fallback candidate exists.
      4. If no evaluated candidates exist, partially_evaluated candidates are considered.
      5. If no viable candidate exists, unsupported is selected with
         reason=no_valid_lowering_path.

    Per-op outputs (flat attrs):
      selected_plan.candidate_id     = "plan_0"
      selected_plan.candidate_type   = the winning candidate type
      selected_plan.backend          = kernel.backend (or "" if absent)
      selected_plan.kernel_library   = kernel.library (or "" if absent)
      selected_plan.kernel_name      = kernel.name (or "" if absent)
      selected_plan.penalty_score    = evaluation.penalty_score of winner (V0 ranking value)
      selected_plan.reason           = selection reason string
      selected_plan.required_boundary_ops = boundary ops from winning candidate
      selected_plan.truth_boundary   = "plan_selection_static_penalty_not_measured_runtime"

    V1 cost evidence promoted from winner (flat attrs for builder collection):
      selected_plan.cost.{compute,memory,dequant,requant,layout_transform,cast,
        backend_switch,launch_overhead,kv_cache,transfer,unsupported,total,
        model_id,truth_boundary}

    Per-op collection attrs:
      compiler.selected_candidates   = [winning candidate dict]
      compiler.selection_rejections  = [all non-selected evaluated candidate dicts]

    compiler.rejected_candidates (from CandidateGenerationPass) is not modified.
    No winner is ranked for IR rewriting. No boundary ops are inserted.
    No IR is rewritten.

    Truth boundary: plan_selection_static_penalty_not_measured_runtime
  }];
  let constructor = "mlir::hir::createPlanSelectionPass()";
  let dependentDialects = ["mlir::func::FuncDialect"];
}

def BoundaryPlanning
    : Pass<"boundary-planning", "mlir::func::FuncOp"> {
  let summary = "Planning-only: annotate ops with boundary materialization requirements from backend capabilities";
  let description = [{
    Reads representation.effective_dtype and representation.source_backend from func.func
    (set by RepresentationPlanningPass), layout.transform_required per-op (set by
    LayoutPlanningPass), op result element dtype from IR types, and
    target.backend_capabilities.{backend}.{supports_cast, supports_dequant_boundary,
    supports_layout_transform} from the module.

    Emitted per non-terminator op:
      boundary.cast_required                -- true if float precision cast is needed and backend supports it
      boundary.dequant_required             -- true if dequantization boundary is needed and backend supports it
      boundary.layout_transform_required    -- true if layout transform is needed and backend supports it
      boundary.materialization_required     -- true if any boundary can be materialized
      boundary.reason                       -- comma-separated summary of decisions
      boundary.truth_boundary               -- always "boundary_planning_only_no_ir_materialization"
      boundary.unsupported_reason           -- (optional) comma-separated list of needed-but-unsupported boundaries

    Decision rules:
      1. Cast: op first result element dtype differs from effective_dtype, both are float
         (f16/f32/bf16), and backend supports_cast.
      2. Layout transform: layout.transform_required == true (from LayoutPlanningPass)
         and backend supports_layout_transform.
      3. Dequant: effective_dtype is quantized (i8/i4) and op first result is float,
         and backend supports_dequant_boundary.
      4. For needed but unsupported boundaries, emit boundary.unsupported_reason and
         leave the _required flag false.

    This pass does NOT insert hir.cast, hir.dequantize, or hir.layout_transform ops.
    It is annotation-only and must run after LayoutPlanningPass.
    Only the entry block is processed.

    Truth boundary: boundary_planning_only_no_ir_materialization.

    Intended to run after LayoutPlanningPass in serving-optimization-pipeline.
  }];
  let constructor = "mlir::hir::createBoundaryPlanningPass()";
}

def BoundaryMaterialization
    : Pass<"boundary-materialization", "mlir::func::FuncOp"> {
  let summary = "Materialize planned cast boundary ops into IR after plan selection";
  let description = [{
    First IR-materializing pass in the serving pipeline. Consumes the
    boundary requirements the planning passes annotated and inserts real
    boundary ops for the smallest safe subset:

      boundary.cast_required = true
        -> inserts hir.cast on the op's first result, converting from the
           op's float element dtype to the func-level
           representation.effective_dtype (the same source
           BoundaryPlanningPass used to decide the requirement). All other
           uses of the result are redirected to the cast; if the cast
           reaches func.return, the function result types are updated.

    Deliberately NOT materialized (deferred, recorded per-op in
    boundary.materialization.deferred and counted at func level):
      boundary.dequant_required / boundary.weight_dequant_required
        -- an honest hir.dequantize needs scale/zero_point metadata that the
           planning pipeline does not produce; fabricating values would
           violate the truth boundary.
      boundary.layout_transform_required
        -- no layout-transform op exists in the hir dialect yet.

    Skipped (never materialized): ops whose selected_plan.candidate_type is
    "unsupported" — a plan with no viable lowering must not have boundary
    ops materialized. Recorded in boundary.materialization.skipped_reason.
    Ops already carrying boundary.cast_materialized = true are left
    untouched (idempotent re-run).

    Diagnostics (no silent no-op on malformed planning attrs):
      - boundary.cast_required = true but the func has no float
        representation.effective_dtype        -> error, pass failure
      - boundary.cast_required = true but the op's first result is not a
        ranked tensor with a float element type -> error, pass failure
      - boundary.cast_required = true but source and target element types
        are equal                              -> error, pass failure
      - boundary.materialization_required = true with no specific
        requirement flag set                   -> warning
      - deferral / unsupported-skip            -> remark

    Every inserted op carries provenance attrs:
      compiler.materialized = true
      materialized.by = "boundary-materialization"
      materialized.from_decision = "boundary.cast_required"
      materialized.of_op = source op name
      cast.from_dtype / cast.to_dtype
      materialized.truth_boundary
        = "compiler_materialized_boundary_op_not_runtime_executed"

    Anchor op attrs: boundary.cast_materialized = true,
    boundary.materialized_ops = ["cast"], boundary.materialization.status.
    Func attrs: boundary.materialization.{materialized_count,
    deferred_count, status, truth_boundary}.

    Truth boundary: this pass performs a compiler IR transformation only.
    It does not execute kernels, claim runtime behavior, or alter planning
    attrs written by earlier passes. Must run after PlanSelectionPass and
    before ExecutionPlanBuilder. Only the entry block is processed.
  }];
  let constructor = "mlir::hir::createBoundaryMaterializationPass()";
  let dependentDialects = ["mlir::func::FuncDialect"];
}

def TilePlanning
    : Pass<"tile-planning", "mlir::func::FuncOp"> {
  let summary = "Static local-memory tile feasibility planning for matmul-like ops (tile_planning_v1)";
  let description = [{
    Memory-hierarchy-aware static planning. For each matmul-like op with
    fully static shapes, selects the largest tile from a fixed conservative
    menu ({128,128,32}, {64,64,32}, {32,32,32}, {16,16,16}, clipped to the
    op's M/N/K) whose A (Mt x Kt) + B (Kt x Nt) + C (Mt x Nt) working set
    fits the declared local memory
    (target.static_cost_profile.local_memory_bytes — SRAM / shared memory /
    scratchpad capacity from the target profile's staticCostProfile block).

    Emitted per matmul-like op:
      tile.plan.status
        = "planned" | "no_feasible_tile"
        | "deferred_missing_memory_hierarchy"
        | "dynamic_dims_unresolved" | "dtype_unresolved" | "no_tensor_result"
      tile.plan.shape                         -- [Mt, Nt, Kt] when planned
      tile.plan.local_memory_bytes            -- selected tile footprint
      tile.plan.rejected_tile_count
      tile.plan.rejection_reason              -- when no tile fits
      tile.plan.deferred_reason               -- when feasibility deferred
      tile.plan.estimated_global_traffic_bytes
        -- reuse-limited traffic: A read ceil(N/Nt) times, B read
           ceil(M/Mt) times, C written once. Reported alongside (never
           replacing) shape_cost_model_v2's ideal each-byte-once bytes.
      tile.plan.double_buffer_fits            -- 2x(A+B) + C fits capacity
      tile.plan.staging_capability
        = "async_copy_declared" | "dma_declared"
        | "declared_unavailable" (declared and false)
        | "unknown_not_declared" (no declaration — unknown, not unavailable)
      tile.plan.truth_boundary
        = "tile_planning_static_local_memory_model_not_measured_not_codegen"

    Dtypes come from existing metadata (quant.activation_dtype /
    quant.weight_dtype / result element type / representation.effective_dtype),
    so weight-only quantization shrinks the B-tile footprint.

    MemoryHierarchyProfile is OPTIONAL declared metadata — not every backend
    exposes local memory or DMA details. Tile feasibility runs only when
    (1) the op kind is supported, (2) local_memory_bytes is declared, and
    (3) shapes are fully static (plus a resolvable dtype). When the memory
    hierarchy is missing, no value is invented: matmul-like ops are stamped
    "deferred_missing_memory_hierarchy" with tile.plan.deferred_reason, the
    plan stays valid, and the deferral is exported. Dynamic shapes are
    deferred with an explicit status, never estimated.

    ServingCostModelPass converts the planned traffic into
    compiler.shape_profile.estimated_tiled_memory_cost_nanos when bandwidth
    is declared — an annotation only; candidate ranking is unchanged.

    Truth boundary: static local-memory capacity model from declared
    profile numbers. Not measured performance, not DMA/async-copy
    execution, not codegen, and no claim that the backend kernel uses this
    tiling. Runs after QuantizationStrategyPlanning (for quant dtypes) and
    before CandidateEvaluation in compile-for-target; also registered
    standalone as tile-planning-pipeline.
  }];
  let constructor = "mlir::hir::createTilePlanningPass()";
  let dependentDialects = ["mlir::func::FuncDialect"];
}

def KernelSelection
    : Pass<"kernel-selection", "mlir::func::FuncOp"> {
  let summary = "Select a concrete runtime kernel descriptor per op (kernel_selection_contract_v1)";
  let description = [{
    Compiler-side kernel selection against CONCRETE runtime kernel
    descriptors — the compiler/runtime kernel contract. Distinct from
    KernelAvailabilityPlanningPass, which models declared third-party
    LIBRARY coverage (cuBLAS/Triton/CoreML public APIs); this pass selects
    a kernel only when a RuntimeKernelDescriptor in the module's
    target.runtime_kernels registry (from the target profile's
    runtimeKernels array) fully matches the planned op:

      op name x backend x dtype x quant mode x layout (when both sides
      state one) x static/dynamic shape x tile plan (when the descriptor
      constrains tile shapes) x declared local memory (when the descriptor
      requires it)

    Emitted per non-terminator op (kernel_selection.* namespace):
      kernel_selection.status
        = "selected"
        | "rejected_no_kernel_for_op" | "rejected_backend_mismatch"
        | "rejected_dtype_unsupported" | "rejected_quant_mode_unsupported"
        | "rejected_layout_unsupported" | "rejected_tile_shape_unsupported"
        | "rejected_local_memory_insufficient"
        | "deferred_dynamic_shape" | "deferred_missing_tile_plan"
        | "deferred_missing_memory_hierarchy"
        | "deferred_no_kernel_library_declared"
      kernel_selection.selected_id       -- when selected
      kernel_selection.source            -- descriptor source, when selected
        ("handwritten_runtime" | "declared_profile" | "measured_runtime"
         | "fixture")
      kernel_selection.rejection_reasons -- per-descriptor "{id}:{reason}"
      kernel_selection.contract_version  = "kernel_selection_contract_v1"
      kernel_selection.truth_boundary

    The registry is expected to be SMALL AND HONEST: only kernels with a
    known dispatchable implementation are declared (today: the handwritten
    Metal RMSNorm f32 kernel). No coverage is inferred; ops without a
    matching descriptor are rejected or deferred with explicit reasons,
    including when no registry is declared at all.

    Truth boundary: static descriptor matching from declared metadata. The
    compiler does not execute, dispatch, or benchmark kernels; a selection
    is a contract handed to the runtime, not an execution claim. Runs after
    TilePlanningPass (to see tile plans) and before candidate evaluation in
    compile-for-target; standalone as kernel-selection-pipeline.
  }];
  let constructor = "mlir::hir::createKernelSelectionPass()";
  let dependentDialects = ["mlir::func::FuncDialect"];
}

def QuantizationCoDesign
    : Pass<"quantization-codesign", "mlir::func::FuncOp"> {
  let summary = "Quantization co-design evidence per op (quantization_codesign_contract_v1)";
  let description = [{
    For each matmul-like constant-weight op, reports SEPARATE facts (never
    collapsed into one string): numeric representation under consideration,
    algorithm declaration status (none is implemented in this repository;
    only the forced-AWQ profile DECLARES one), backend legality (declared
    profile capability), concrete runtime-kernel support (a
    target.runtime_kernels descriptor — backend capability never counts as
    dispatchable), static systems-cost evidence, honest accuracy-evidence
    status, and materialization requirements.

    INERT unless the module sets quant.codesign.policy:
      planning_only              -- evidence only, never selects
      systems_cost_only          -- selects iff estimated benefit > 0
      require_dispatchable_kernel-- additionally requires a runtime kernel
      require_accuracy_evidence  -- defers without calibrated/measured
                                    evidence (always defers today)
    The policy is recorded in every decision. An unknown policy is a pass
    error, not a silent fallback.

    Emitted per matmul-like op (quant_codesign.* namespace; unknown
    metadata — granularity, group size, axis, symmetric/asymmetric, scale,
    zero point — is OMITTED, never defaulted):
      quant_codesign.status
        = "selected" | "planning_only_not_selected"
        | "rejected_weight_not_constant" | "rejected_backend_not_legal"
        | "rejected_no_systems_benefit"
        | "backend_supported_but_not_dispatchable"
        | "deferred_no_runtime_kernel"
        | "deferred_missing_accuracy_evidence"
        | "deferred_missing_cost_estimates"
        | "deferred_missing_weight_classification"
      quant_codesign.policy, .contract_version, .truth_boundary
      quant_codesign.candidate.{representation, weight_dtype,
        activation_dtype, accumulator_dtype (only when declared)}
      quant_codesign.algorithm.{status, name (only when declared)}
      quant_codesign.accuracy_evidence.{status, artifact_ref (only when
        declared)}   -- status = "no_accuracy_evidence" |
                        "algorithm_declared_not_calibrated" (the only
                        truthful values today)
      quant_codesign.backend_legality = "legal" | "not_legal"
      quant_codesign.kernel_support.{status, kernel_id, source}
        -- status = "runtime_dispatchable" |
           "backend_supported_but_not_dispatchable" |
           "no_kernel_registry_declared"
      quant_codesign.scale_source / .zero_point_source
        = "not_available_no_calibration"
      quant_codesign.est.{weight_bytes_before, weight_bytes_after,
        boundary_bytes, total_cost_before_nanos, total_cost_after_nanos,
        systems_benefit_nanos, excluded_terms}
      quant_codesign.materialization.{required, status}
      quant_codesign.rejection_reasons

    Cost-term inclusion boundary (full table in
    docs/QUANTIZATION_CODESIGN.md): the boundary term models the
    MATERIALIZED float dequant intermediate (2x float weight bytes, same
    formula as ShapeCostModel). A dispatchable weight-only kernel removes
    that materialized intermediate only; inline scale/metadata/unpack/
    conversion cost is NOT modeled and is listed in est.excluded_terms
    rather than silently treated as free.

    Invariants: existing quant.strategy semantics are untouched;
    quant_codesign.est.* is never read by CandidateEvaluation or
    PlanSelection (enforced by a ranking-invariance test); nothing is
    estimated for dynamic shapes.

    Truth boundary:
    quantization_codesign_static_planning_no_calibration_no_measured_accuracy
    — static planning evidence from declared profiles and tensor shapes.
    No calibration data, no measured accuracy, and no quantized runtime
    execution exist in this repository.
  }];
  let constructor = "mlir::hir::createQuantizationCoDesignPass()";
  let dependentDialects = ["mlir::func::FuncDialect"];
}

def DistributedStrategyPlanning
    : Pass<"distributed-strategy-planning", "mlir::ModuleOp"> {
  let summary = "D2: generate, evaluate, and select a TP1/TP2 distributed "
                "strategy candidate for one real Qwen operator instance";
  let description = [{
    Module-scoped pass (like QuantizationPlanning, not a per-func pass):
    the distributed strategy is a single whole-model decision, not a
    per-function one.

    Reads real module/operator metadata already present in the pipeline by
    this point: llm.model, llm.num_layers, llm.hidden_size,
    llm.num_attention_heads, llm.num_key_value_heads, and the first
    "llm.o_proj" op found in the module (its serving.layer_index, parent
    function name, and static trailing tensor dimension). Reads the
    explicit D2 opt-in flag distributed.opt_in (set by compile-for-target
    only when the target profile JSON declares
    "distributedStrategyOptIn": true).

    Generates exactly the D1 {TP1, TP2} candidates (reusing
    serving::generateDistributedCandidates), applies D1 structural legality
    plus D2 Qwen-specific legality (operator support, static shape
    availability, hidden-dimension divisibility, distributed capability/
    opt-in availability), computes an explicit inspectable cost estimate
    per candidate (rank-local compute bytes, communication bytes, collective
    count, process-launch overhead calibrated from D1's local IPC
    measurements, unsupported/fallback penalties), and deterministically
    selects one candidate:

      legal TP2 + explicit opt-in profile  -> TP2 may be selected
      illegal TP2                          -> TP1 selected, rejection
                                              reasons recorded
      no distributed capability/opt-in     -> TP1 selected
      internally inconsistent state        -> pass emits an error (fail
                                              closed; module verification
                                              fails, no plan is emitted)

    Emits module-level attributes read by ExecutionPlanBuilder's
    collectDistributedPlan(): distributed.candidates (evidence for every
    candidate, always emitted), distributed.selected_candidate_id,
    distributed.selection_reason, distributed.policy_id, and -- only when
    the selected candidate has world_size > 1 -- distributed.strategy/
    world_size/tensor_parallel_size/pipeline_parallel_size/ranks/
    tensor_shards/collectives/truth_boundary (absent entirely for TP1
    selections, preserving D1's legacy-plan backward compatibility).

    This pass never claims real Qwen tensor-parallel execution, real GPU
    tensor parallelism, NCCL, or measured distributed-serving profitability.
    truth_boundary = "d2_qwen_pipeline_structural_planning_not_measured_gpu_"
    "performance_not_nccl_calibrated".
  }];
  let constructor = "mlir::hir::createDistributedStrategyPlanningPass()";
}

def QuantizationPlanning
    : Pass<"quantization-planning", "mlir::ModuleOp"> {
  let summary = "Compiler-side precision selection for the HIR serving pipeline";
  let description = [{
    Reads model dtype intent (llm.dtype) and target precision support
    (target.supported_precisions) from the MLIR module. Emits a single
    compiler-side precision decision as module-level attributes:

      quantization.plan_dtype    -- chosen precision: "fp16", "int8", etc.
      quantization.plan_source   -- "target_profile" or "default_dtype"
      quantization.truth_boundary
        = "precision_selection_from_target_profile_not_calibrated"

    Decision priority (first matching rule wins):
      1. llm.dtype is in target.supported_precisions -> use llm.dtype,
         plan_source = "target_profile"
      2. target.supported_precisions is non-empty     -> use first element,
         plan_source = "target_profile"
      3. llm.dtype is non-empty (no supported list)   -> use llm.dtype,
         plan_source = "default_dtype"
      4. default                                       -> "fp16",
         plan_source = "default_dtype"

    If plan_dtype == "int8", annotates eligible linalg.matmul ops with:

      quantization.candidate = "int8"
      profile.quantized_path  = "faster"

    This enables the existing shouldUseQuantizedMatMul() path in
    MatMulBiasReluFusionPass without modifying that pass.

    This pass does NOT:
      - insert hir.quantize or hir.dequantize ops
      - generate scale or zero_point values
      - modify the runtime, execution plan schema, or JSON exporter

    Truth boundary: precision selection is derived from target profile
    metadata only. No calibration data, measured quantization speedup,
    or runtime validation is performed.
  }];
  let constructor = "mlir::hir::createQuantizationPlanningPass()";
}

def HIRQuantCanonicalization
    : Pass<"hir-quant-canonicalize", "mlir::func::FuncOp"> {
  let summary = "Eliminate redundant HIR quantize/dequantize boundaries";
  let description = [{
    Performs conservative Q/DQ cleanup when adjacent HIR quantization ops carry
    identical quantization metadata:

      hir.dequantize(hir.quantize(x)) -> x
      hir.quantize(hir.dequantize(x)) -> x

    The rewrite requires matching scale, zero_point, quantized_dtype, and
    quantization.mode attributes. Mismatched metadata leaves the boundary in
    place.
  }];
  let constructor = "mlir::hir::createHIRQuantCanonicalizationPass()";
}

def HIRQuantPropagation
    : Pass<"hir-quant-propagate", "mlir::func::FuncOp"> {
  let summary = "Annotate conservative INT8-capable islands";
  let description = [{
    Forms metadata-only INT8 islands through a small safe op set:

      linalg.matmul with quantization.candidate = "int8"
      ReLU-shaped linalg.map
      tensor reshape/cast ops

    Unsupported ops are left unannotated and therefore break the island. The
    pass does not reorder graph operations, insert Q/DQ ops, or claim runtime
    dispatch.
  }];
  let constructor = "mlir::hir::createHIRQuantPropagationPass()";
}

def HIRINT8OperatorSelection
    : Pass<"hir-int8-operator-selection", "mlir::func::FuncOp"> {
  let summary = "Select INT8 execution metadata using a small capability table";
  let description = [{
    Uses a deterministic compiler capability table keyed by op, precision,
    backend, layout, and simple shape legality. The pass annotates INT8-capable
    ops with selection metadata only; it does not lower to runtime kernels.

    INT8 is selected only when:
      - the op is in a quantized island or explicitly marked as an INT8 candidate
      - the backend capability table supports INT8 for the op
      - layout and shape/alignment constraints are legal
      - profile metadata is absent or says the INT8 path is faster
  }];
  let constructor = "mlir::hir::createHIRINT8OperatorSelectionPass()";
}

def QuantizationMaterialization
    : Pass<"quantization-materialization", "mlir::func::FuncOp"> {
  let summary = "Materialize selected INT8 fused MatMul+Bias+ReLU decisions into explicit HIR Q/DQ";
  let description = [{
    Consumes an already-selected int8_static_symmetric packed-B decision on
    hir.fused_matmul_bias_relu and rewrites the float op into explicit
    hir.quantize, hir.load_quantized_weight, hir.qmatmul, hir.dequantize, and
    hir.fused_qmatmul_bias_relu implementation IR. This pass does not choose
    precision; missing selected artifacts are compilation errors for the
    selected INT8 path.
  }];
  let constructor = "mlir::hir::createQuantizationMaterializationPass()";
  let dependentDialects = ["mlir::hir::HIRDialect"];
}

def QuantizedKernelLowering
    : Pass<"quantized-kernel-lowering", "mlir::func::FuncOp"> {
  let summary = "Lower explicit quantized HIR to the selected portable CPU INT8 kernel contract";
  let description = [{
    Converts materialized hir.fused_qmatmul_bias_relu implementation IR into
    hir.portable_cpu_int8_fused_matmul_bias_relu, preserving the selected
    candidate, packed artifact, calibration artifact, codegen target, binary
    identity, and explicit execution-stage dependency contract. This pass
    consumes existing compiler decisions and never selects a kernel.
  }];
  let constructor = "mlir::hir::createQuantizedKernelLoweringPass()";
  let dependentDialects = ["mlir::hir::HIRDialect"];
}

def LLMFrontendNormalization
    : Pass<"llm-frontend-normalization", "mlir::func::FuncOp"> {
  let summary = "Normalize simplified LLM attention graph patterns to canonical serving-aware IR";
  let description = [{
    Detects simplified pseudo-LLM attention graph patterns using a set of
    recognizable op names:

      llm.q_proj, llm.k_proj, llm.v_proj
      llm.attention_scores, llm.attention_output
      llm.kv_cache_write  (prefill signal — KV producer)
      llm.kv_cache_read   (decode  signal — KV consumer)

    When the attention pattern is detected, the entire function body is
    rewritten to canonical serving-aware IR:

      kv_cache_write present -> llm.attention_prefill, kv_cache.role = "producer"
      kv_cache_read  present -> llm.attention_decode,  kv_cache.role = "consumer"

    The canonical op carries the standard serving attributes:

      kv_cache.role, serving.phase, serving.prompt_tokens, serving.output_tokens
      frontend.source          = "llm_graph_pattern"
      frontend.truth_boundary  = "attention_pattern_simplified_not_full_transformer_lowering"
      frontend.normalized_from = "simplified_llm_attention_graph"

    The function signature and return type are preserved. The rewritten body
    contains exactly two ops: the canonical attention op and a return.

    The enclosing func.func is also annotated with frontend.source and
    frontend.truth_boundary so downstream passes can identify normalized input.

    Output is identical in structure to qwen-to-serving-mlir canonical IR.
    The serving-optimization-pipeline cannot distinguish Path A (qwen-to-serving-mlir)
    from Path B (this pass).

    Truth boundary: input ops are simplified pseudo-LLM ops, not real
    StableHLO/Torch/ONNX graph patterns. No weight import, no operator
    lowering, no shape propagation across projections.
  }];
  let constructor = "mlir::hir::createLLMFrontendNormalizationPass()";
  let dependentDialects = ["mlir::func::FuncDialect"];
}

def AttentionCandidateGeneration
    : Pass<"attention-candidate-generation", "mlir::func::FuncOp"> {
  let summary = "Generate phase-specific FP32 CPU attention candidates";
  let constructor = "mlir::hir::createAttentionCandidateGenerationPass()";
  let dependentDialects = ["mlir::hir::HIRDialect"];
}

def AttentionLegality
    : Pass<"attention-legality", "mlir::func::FuncOp"> {
  let summary = "Validate generated attention candidates independently";
  let constructor = "mlir::hir::createAttentionLegalityPass()";
  let dependentDialects = ["mlir::hir::HIRDialect"];
}

def AttentionSelectionLowering
    : Pass<"attention-selection-lowering", "mlir::func::FuncOp"> {
  let summary = "Select the legal phase implementation and lower to a CPU runtime call";
  let constructor = "mlir::hir::createAttentionSelectionLoweringPass()";
  let dependentDialects = ["mlir::hir::HIRDialect"];
}

def CVFrontendNormalization
    : Pass<"cv-frontend-normalization", "mlir::func::FuncOp"> {
  let summary = "Detect and annotate pseudo-CV YOLOSeg-like graph patterns";
  let description = [{
    Walks a func.func looking for pseudo-CV ops representing a YOLOSeg-style
    architecture. Detect gate: the function must contain at least one cv.conv2d,
    at least one cv.detect_head, and at least one cv.prototype_head. If the gate
    fails, the function is skipped without annotation or error.

    When the gate passes, annotates the enclosing func.func with:

      cv.frontend.kind                 = "raw_pseudo_cv_mlir"
      cv.frontend.model_family         = "yoloseg"
      cv.frontend.normalization_status = "detected_not_rewritten"
      cv.frontend.truth_boundary
        = "raw_pseudo_cv_mlir_not_full_onnx_importer"
      cv.frontend.conv2d_count         = N : i64
      cv.frontend.batch_norm_count     = N : i64
      cv.frontend.silu_count           = N : i64
      cv.frontend.upsample_count       = N : i64
      cv.frontend.concat_count         = N : i64
      cv.frontend.detect_head_count    = N : i64
      cv.frontend.prototype_head_count = N : i64

    The function body is left intact. No op rewrites are performed.
    Downstream CV planning passes consume the func.func attrs emitted here.

    Truth boundary: raw_pseudo_cv_mlir_not_full_onnx_importer
    (inputs are hand-written pseudo-CV ops, not real ONNX-imported YOLOSeg
    nodes; no weights imported; shapes are plausible but not verified against
    an actual YOLOSeg ONNX export).
  }];
  let constructor = "mlir::hir::createCVFrontendNormalizationPass()";
  let dependentDialects = ["mlir::func::FuncDialect"];
}

def CVShapeInference
    : Pass<"cv-shape-inference", "mlir::func::FuncOp"> {
  let summary = "Annotate pseudo-CV ops with static shape-derived size metadata";
  let description = [{
    Walks a func.func for ops whose name starts with "cv.". For each such op
    with exactly one fully-static ranked tensor result, attaches:

      cv.num_elements                  = N : i64
      cv.bytes_estimate                = N : i64  (fp16 assumed, 2 bytes/element)
      cv.shape_inference.truth_boundary
        = "static_ranked_tensor_shape_metadata_not_runtime_allocation"

    After the walk, annotates the enclosing func.func with:

      cv.shape_inference.status                = "completed"
      cv.shape_inference.annotated_ops         = N : i64
      cv.shape_inference.skipped_ops           = N : i64
      cv.shape_inference.total_bytes_estimate  = N : i64
      cv.shape_inference.truth_boundary
        = "static_ranked_tensor_shape_metadata_not_runtime_allocation"

    The pass returns early without annotating the function if it contains no
    cv.* ops at all.

    Skip policy: ops with zero or multiple results, non-ranked result types,
    or any dynamic dimension are counted as skipped. The pass never calls
    signalPassFailure() for skipped ops.

    Dtype assumption: fp16 (2 bytes/element). This pass does not read
    quantization.plan_dtype; dtype coupling is deferred to a later pass.

    Truth boundary: metadata is derived from static tensor type shapes only.
    No runtime allocation, no buffer assignment, no execution plan.
  }];
  let constructor = "mlir::hir::createCVShapeInferencePass()";
  let dependentDialects = ["mlir::func::FuncDialect"];
}

def CVMemoryPlanning
    : Pass<"cv-memory-planning", "mlir::func::FuncOp"> {
  let summary = "Assign compiler buffer slots to pseudo-CV ops using linear lifetime reuse";
  let description = [{
    Walks the entry block of a func.func in order, collecting plannable cv.*
    ops (ops whose name starts with "cv.", with exactly one result and a
    cv.bytes_estimate i64 attr). Computes SSA-based lifetimes and allocates
    compiler buffer slots with a linear first-fit reuse policy.

    Single-block assumption: iterates funcOp.getBody().front() only. Functions
    with control flow or nested regions are not handled in this version.

    Dependency: reads cv.bytes_estimate (set by CVShapeInferencePass). Ops
    without cv.bytes_estimate are counted as skipped_ops and receive no buffer
    assignment.

    Algorithm (two-phase):
      Phase 1 — Lifetime analysis:
        For each plannable op at index i, lifetime_begin = i, lifetime_end = i.
        For each user of the op's result: if the user is also plannable,
        lifetime_end = max(lifetime_end, user_index).

      Phase 2 — Buffer allocation (linear first-fit reuse):
        For each plannable op in block order, call selectReusableBuffer to find
        the first pool entry where entry.freedAt < lifetime_begin and
        entry.sizeBytes >= bytes_estimate. If found, reuse that buffer slot and
        update its freedAt. Otherwise allocate a new buffer at nextOffset.

    Per-op attrs (plannable ops only):
      cv.buffer_id                       = N : i64
      cv.buffer_offset                   = N : i64
      cv.lifetime_begin                  = N : i64
      cv.lifetime_end                    = N : i64
      cv.memory_plan.truth_boundary
        = "static_compiler_memory_plan_not_runtime_allocation"
      cv.memory_plan_reason
        = "linear_lifetime_reuse_from_static_shape_metadata"
      cv.reuse_group                     = N : i64  (= buffer_id; shared by reuse partners)

    Function-level attrs (set when at least one cv.* op is present):
      cv.memory_plan.buffer_count        = N : i64
      cv.memory_plan.peak_memory_bytes   = N : i64
      cv.memory_plan.planned_ops         = N : i64
      cv.memory_plan.reused_buffer_count = N : i64
      cv.memory_plan.skipped_ops         = N : i64
      cv.memory_plan.status              = "completed"
      cv.memory_plan.total_allocated_bytes = N : i64
      cv.memory_plan.truth_boundary
        = "static_compiler_memory_plan_not_runtime_allocation"

    Skip policy: ops without cv.bytes_estimate or with != 1 result are
    skipped. Dead outputs (no plannable cv.* consumer) are still allocated;
    their lifetime_end = lifetime_begin.

    Truth boundary: buffer assignments are static compiler-side metadata
    derived from SSA lifetimes and annotated byte estimates. No runtime
    allocation, no device memory assignment, no execution plan.
  }];
  let constructor = "mlir::hir::createCVMemoryPlanningPass()";
  let dependentDialects = ["mlir::func::FuncDialect"];
}

def CVExecutionDomainPlanning
    : Pass<"cv-execution-domain-planning", "mlir::func::FuncOp"> {
  let summary = "Classify pseudo-CV ops into portable execution domains";
  let description = [{
    Walks a func.func for ops whose name starts with "cv." and classifies
    each into a portable execution domain using op name only:

      accelerated -- tensor-heavy, regular data-parallel computation
                     structurally suitable for accelerator-class execution
      host        -- irregular post-processing or control-flow-like operations
                     structurally suitable for host/CPU execution
      fallback    -- unknown cv.* op; classified conservatively for host fallback

    This pass is target-portable. It does NOT assign Metal, CPU, CUDA,
    CoreML, ANE, or any concrete runtime backend. Execution domain values
    are abstract compiler classifications only.

    Concrete backend binding happens in a later CVExecutionPlanBuilder pass
    that maps (execution_domain + target_profile) to a concrete backend:
      accelerated + apple_profile  -> metal
      accelerated + cuda_profile   -> cuda
      accelerated + mobile_profile -> npu
      host + apple_profile         -> cpu

    Gate: if no cv.* ops are present, returns without setting function attrs.

    Policy (op name -> domain):
      accelerated: cv.conv2d, cv.batch_norm, cv.silu, cv.upsample, cv.concat
      host:        cv.detect_head, cv.prototype_head
      fallback:    all other cv.* ops

    classifyExecutionDomain() is the only place containing this policy.

    Per-op attrs (every cv.* op):
      cv.execution_domain
        = "accelerated" | "host" | "fallback"
      cv.execution_domain.truth_boundary
        = "static_execution_domain_classification_not_target_mapping"
      cv.execution_domain_reason
        = "accelerated_tensor_policy" | "host_postprocess_policy"
        | "fallback_unknown_cv_op"

    Function-level attrs (set when gate passes):
      cv.execution_domain_plan.accelerated_ops = N : i64
      cv.execution_domain_plan.fallback_ops     = N : i64
      cv.execution_domain_plan.host_ops         = N : i64
      cv.execution_domain_plan.planned_ops      = N : i64
      cv.execution_domain_plan.status           = "completed"
      cv.execution_domain_plan.truth_boundary
        = "static_execution_domain_classification_not_target_mapping"

    Truth boundary: domain classification is a compiler-side structural
    judgment. No runtime backend is selected or dispatched by this pass.
  }];
  let constructor = "mlir::hir::createCVExecutionDomainPlanningPass()";
  let dependentDialects = ["mlir::func::FuncDialect"];
}

def CVSemanticAnnotation
    : Pass<"cv-semantic-annotation", "mlir::func::FuncOp"> {
  let summary = "Annotate real upstream YOLO-Seg MLIR with CV semantic attrs";
  let description = [{
    Runs on upstream MLIR, not cv.* operations. The pass recognizes compact
    semantic boundaries and regions from tensor contracts, output topology,
    producer/consumer relationships, operation families, and spatial shapes.

    It annotates existing upstream operations and function results with attrs
    such as:

      cv.model_family
      cv.semantic_role
      cv.region_id
      cv.output_role
      cv.feature_scale
      cv.postprocess_boundary
      cv.recognition_confidence
      cv.recognition_evidence

    The pass does not introduce custom CV operations, select backends, plan
    memory, select kernels, generate an ExecutionPlan, or continue lowering.

    Truth boundary:
      cv_semantic_annotation_only_no_backend_selection_no_memory_plan_no_kernel_selection_no_execution_plan_generation
  }];
  let constructor = "mlir::hir::createCVSemanticAnnotationPass()";
  let dependentDialects = [
    "mlir::func::FuncDialect",
    "mlir::tensor::TensorDialect",
    "mlir::linalg::LinalgDialect",
    "mlir::arith::ArithDialect",
    "mlir::math::MathDialect"
  ];
}

def CVExecutionPlanAttrs
    : Pass<"cv-execution-plan-attrs", "mlir::func::FuncOp"> {
  let summary = "Attach minimum CV planning attrs for canonical ExecutionPlan export";
  let description = [{
    Runs on real upstream MLIR after cv-semantic-annotation and after target.*
    attrs have been attached. The pass does not require or create cv.* numeric
    operations. It annotates the existing function and upstream operations with
    the same generic planning attr families consumed by the canonical
    ExecutionPlanBuilder:

      serving.policy
      execution_provider.*
      representation.*
      layout.*
      quant.*
      kernel.*
      cv.execution_plan.*
      cv.memory.*

    The pass makes only static/configured compiler decisions. Existing generic
    kernel availability, lowering, kernel selection, candidate generation, and
    cost-model passes may refine the seeded attrs when a target profile declares
    concrete capabilities. No runtime execution, measured latency, numerical
    validation, or memory slot assignment is performed.
  }];
  let constructor = "mlir::hir::createCVExecutionPlanAttrsPass()";
  let dependentDialects = [
    "mlir::func::FuncDialect",
    "mlir::tensor::TensorDialect",
    "mlir::linalg::LinalgDialect",
    "mlir::arith::ArithDialect",
    "mlir::math::MathDialect"
  ];
}

#endif // HIR_FUSION_PASSES
