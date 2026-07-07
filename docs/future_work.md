# Future Work

## Completed Recently

CV compiler Phase 1 work has moved out of future work:

- `CVFrontendNormalizationPass`
- `CVShapeInferencePass`
- `CVMemoryPlanningPass`
- `CVExecutionDomainPlanningPass`
- `CVExecutionPlanBuilder`
- `CVExecutionPlanExporter`
- `emit-cv-execution-plan`
- `artifacts/apple_demo/cv_execution_plan.json`
- Registered `cv` dialect
- Seven CV operations
- Variadic `cv.detect_head`
- Registered CV parsing path in CV tool contexts

## Near-Term Handoff Tasks

- Add a short contributor workflow that says which demos/tests to run for common changes.
- Separate generated artifacts from source-owned files, or add clear regeneration commands per artifact family.
- Add unit tests for `Graph`, `MemoryPlanner`, `ExecutionPlanBuilder`, and `CostBasedPlanner`.
- Add CPU kernel correctness tests with small deterministic tensors.
- Add schema checks for runtime JSON artifacts.
- Make output paths configurable instead of relying on `../trace/...`.

## Runtime Improvements

- Extend the backend interface to model device buffers, allocation, synchronization, async execution, and errors.
- Rename or document the generic Metal backend as partial until it executes real graph kernels.
- Keep `MockGPUBackend` available for tests, but prevent it from being confused with measured GPU execution.
- Add explicit unsupported-op handling in the op registry and executor.
- Add runtime validation that each node has the expected number of inputs/outputs before dispatch.
- Add deterministic scheduler tests for backend/provider selection.

## Compiler and IR Improvements

- Define a stronger schema for `Graph`, `Tensor`, `Node`, lowered graph, and execution plan artifacts.
- Add dtype support beyond float32 if the custom runtime is expected to reflect MLIR quantization paths.
- Centralize op metadata such as expected ranks, input counts, broadcasting behavior, and supported backends.
- Add graph rewrite helpers that preserve tensor/node id validity.
- Clarify the boundary between custom graph passes and real MLIR passes.

## MLIR Work

- Keep FileCheck coverage for positive and negative fusion cases.
- Add documentation for required LLVM/MLIR versions known to work.
- Add CI or a script that reports clear skip reasons when MLIR tools are unavailable.
- Consider generating runtime JSON from a formal schema instead of ad hoc parsing.
- Keep HIR verifier tests close to every new attribute or lowering rule.

## CV Compiler Future Work

- ONNX importer.
- More CV operators.
- Dynamic-shape support.
- Backend/kernel mapping.
- PocketChef visualization of the CV execution plan.

## LLM Serving Artifact Work

- Label estimated latency, throughput, memory, and pressure metrics inside generated artifacts.
- Add schema version validation for every JSON artifact.
- Add tests that intentionally break KV cache capacity, memory budget, missing phases, and manifest completeness.
- Separate serving policy contracts from measured runtime traces.
- If real serving execution is added later, add measured TTFT/TPOT/throughput capture with environment metadata.

## Qwen Real Graph Import: Phase 1 Done (Scaffold), Remaining Work

Phase 1 of moving off the ModelSpec-driven frontend is done:

- `mlir_passes/tools/qwen-onnx-to-serving-mlir` emits full per-layer-expanded,
  flat, unrolled serving MLIR (real SSA chaining, `serving.layer_index` /
  `serving.layer_role` stamped per op) from
  `configs/models/qwen_0_5b_onnx_graph_facts.json`, replacing
  `qwen-to-serving-mlir`'s single hand-templated block for the graph-import
  target architecture. `qwen-to-serving-mlir` remains as the legacy/scaffold
  ModelSpec path.
- `LLMFrontendNormalizationPass` was generalized from a whole-function
  erase-and-replace (dummy operand) into a localized per-occurrence rewrite:
  it buckets ops by `serving.layer_index` (an ungrouped bucket for legacy
  single-occurrence functions, preserving prior behavior) and canonicalizes
  each layer's raw attention pattern independently, wiring the canonical
  `llm.attention_prefill`/`decode` op to that layer's real q/k/v_proj values.
- `QuantizationStrategyPlanningPass`/`WeightClassificationPlanningPass` now key
  off an explicit `serving.quantizable = true` attribute stamped by the
  importer (both `qwen-onnx-to-serving-mlir` and the legacy
  `qwen-to-serving-mlir`) on linear/weight-bearing ops, so real per-layer ops
  produce decisions instead of silently vanishing from `per_op_decisions`.
  Earlier in Phase 1 this was done with `"proj"`/`"mlp"` op-name substring
  matching directly in those passes; that leaked frontend-specific naming
  into generic planning passes and has been replaced by the explicit
  attribute. The passes still fall back to a small generic, model-family-
  agnostic name-fragment list (`matmul`, `conv`, `gemm`, etc.) for ops with no
  explicit marker.
- Validated end to end via `QwenOnnxServingPlanExportTest`: a 24-layer plan
  produces 24 distinct `serving.layer_index` values and ~170
  `per_op_decisions` entries per phase (vs. the legacy path's single entry),
  fully verbose — no `layer_range`/`layer_count` compression, no JSON schema
  change (per the Phase 1 decision to prioritize import fidelity first).

Remaining work (not done, do not claim otherwise):

- **Real ONNX parser.** `configs/models/qwen_0_5b_onnx_graph_facts.json` is a
  hand-authored fixture standing in for facts a real ONNX importer would
  extract, not a parse of a real `.onnx` protobuf file. `tools/export_qwen_onnx.py`
  can attempt a real HF Optimum export + real `onnx`-package graph
  introspection when that optional toolchain is installed, but its output is
  a diagnostic report only — it is not consumed by the C++ importer.
- **ONNX-MLIR or equivalent frontend integration.** Building the real bridge
  means either a C++ ONNX protobuf reader or (preferred — see the design
  discussion that produced this milestone) building on an existing
  ONNX-MLIR-family frontend rather than hand-rolling protobuf parsing, so the
  importer's op-name vocabulary is derived from a real graph instead of the
  current fixed known list (`kSupportedOps` in `qwen-onnx-to-serving-mlir`'s
  `main.cpp`).
- **Layer-range compression.** Add `layer_range`/`repeat_count` as an
  additive field on `PerOpDecisionBundle`/the JSON schema, with a fold step
  that only collapses a range when decision content is verified identical
  across layers — never assumed. This is export-time only; the compiler's
  internal IR model stays full expansion regardless.
- **Decode-with-past import**, and revisiting whether `KVLayoutPlanningPass`'s
  `numLayers`-based KV-footprint formula should instead sum real per-op KV
  attrs now that real per-layer ops exist to sum.

## Qwen GTX 1650 vLLM Serving: Phase C (Quantized) — Not Implemented

`artifacts/qwen/execution_plan.json` (fp16, quantization `none`/`fp16_fallback`)
is the compiler-side source artifact for the measured A/B vLLM benchmark
recorded in `heterogeneous-inference-runtime` (`results/qwen_no_quant/`). A
quantized Phase C is not implemented. Minimum remaining work:

- A real AWQ/GPTQ Qwen weight export step (e.g. AutoAWQ / auto-gptq against the
  original `Qwen/Qwen2.5-0.5B-Instruct` checkpoint), producing a local
  quantized model artifact — this repo has no such export tool today.
- A target profile that actually declares int4/AWQ backend support.
  `configs/target_profiles/nvidia_gtx1650_maxq.json` currently declares
  `supportedQuantModes: ["none"]` for both `cuda_triton` and `cuda_cublas`
  backends (Turing, cc 7.5, no native INT4 tensor cores) — it cannot honestly
  produce an AWQ `QuantizationDecision` as-is. Either add a new profile for a
  quant-capable target or add an explicit experimental forced-quant profile
  variant with its own truth-boundary label.
- A `QuantizationStrategyPlanningPass` decision path that selects
  `weight_only_int4`/AWQ when the profile allows it (see the illustrative
  example in `docs/EXECUTION_PLAN_SCHEMA.md`, which is not yet real output).
- A materializer update on the runtime side
  (`deployment/vllm_adapter/config_materializer.py` in
  `heterogeneous-inference-runtime`) to emit `--quantization awq|gptq` and
  point `--model` at the quantized artifact path instead of the HF repo id.
- A repeatability benchmark pass for the quantized path mirroring the existing
  `results/qwen_no_quant/repeatability_summary.md` structure.

## Documentation Work

- Maintain a single "implemented vs simulated" section in every high-level doc that discusses runtime behavior.
- Add regeneration instructions for each PNG and JSON family.
- Add a dependency matrix for CMake, Python, MLIR, Metal, CUDA, IREE, Torch-MLIR, and optional benchmark tools.
- Add a glossary for custom IR, HIR, lowered graph, execution plan, trace, artifact, and integration bundle.
