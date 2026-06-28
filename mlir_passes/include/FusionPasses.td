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

#endif // HIR_FUSION_PASSES
