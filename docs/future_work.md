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

## Qwen Real Graph Import: Phase 1 + Phase 2 Done, Remaining Work

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

Phase 2 (real ONNX protobuf bridge, implemented as a **frontend adapter**,
not a general importer) is done:

- `tools/onnx_graph_to_facts.py` loads a real `.onnx` file with the Python
  `onnx` package and reads real protobuf structure (node op types,
  initializer names/shapes/dtypes — never tensor values). It classifies
  per-layer roles by matching real initializer names against Qwen2's
  specific HuggingFace naming convention, derives
  `num_layers`/`hidden_size`/`intermediate_size`/`vocab_size`/`dtype` from
  real shapes/dtypes, and detects RoPE and lm_head/embedding tying from real
  graph signals. `num_attention_heads`/`num_key_value_heads`/
  `max_position_embeddings` are read from an HF `config.json` next to the
  `.onnx` file or an explicit CLI override, never guessed (they are not
  recoverable from graph structure alone). It emits the same `GraphFacts`
  JSON schema Phase 1 already established, plus additive `provenance` and
  optional `positional_encoding` fields — `qwen-onnx-to-serving-mlir` needed
  no changes to consume its output.
- Any missing per-layer role (e.g. a layer with no `v_proj` initializer) is
  a hard failure (`OnnxGraphToFactsError`), never a silent guess or omitted
  role.
- `tools/validate_onnx_graph_facts.py` validates any `GraphFacts` document.
  For real bridge output it checks `num_layers` against the number of
  distinct parsed layer indices and that every layer has all required
  roles, failing hard on any gap. For the hand-authored fixture (no
  `provenance` field) it validates schema fields only and explicitly
  reports per-layer completeness as skipped, not silently passed.
- `qwen-onnx-to-serving-mlir`'s `main.cpp` gained one small additive read:
  an optional `positional_encoding` GraphFacts field, stamped as a
  function-level `serving.positional_encoding` attr on
  `qwen_prefill`/`qwen_decode` when present. RoPE stays absorbed during
  pattern recognition, not materialized as a distinct op — this only
  records the declared fact. Absent for the hand-authored fixture, which
  emits byte-identical MLIR to before this change.
- The hand-authored fixture is kept, unmodified, as fast/deterministic/
  network-free regression coverage
  (`tests/test_onnx_graph_facts_fixture_regression.py`) — not replaced by
  the bridge. `tests/test_onnx_graph_to_facts.py` covers the bridge itself
  (tiny synthetic ONNX graphs built with `onnx.helper.make_graph`, RoPE
  detection, tied/untied lm_head, hard-failure on an incomplete layer,
  config.json scalar resolution, and clean skip when `onnx` isn't
  installed) plus an end-to-end test through the real
  `qwen-onnx-to-serving-mlir` binary.

Long-term architecture direction (target, not current — see
`docs/architecture.md`'s "Future Architecture" section for the full
diagram): semantic recognition (role classification, RoPE/tied-embedding
detection) should eventually move out of the Python frontend adapter and
into a compiler-side MLIR pass; frontend adapters should eventually become
thin format parsers only (pure parsing and graph traversal, no architecture
knowledge); `GraphFacts` is the current transition layer — it already
isolates planning passes from frontend format details, but today it still
carries semantic role labels a Python adapter assigns, which a future
compiler-side recognition pass would assign instead. The items below are
concrete steps toward that target, not requirements for it to already exist.

Remaining work (not done, do not claim otherwise):

- **ONNX-MLIR or equivalent frontend integration.** `tools/onnx_graph_to_facts.py`
  is a Qwen2-specific pattern matcher over initializer names, not a general
  ONNX frontend. Building a true general importer means either a C++ ONNX
  protobuf reader or (preferred — see the design discussion that produced
  this milestone) building on an existing ONNX-MLIR-family frontend, so the
  importer's op-name vocabulary is derived from arbitrary real graphs
  instead of the current fixed known list (`kSupportedOps` in
  `qwen-onnx-to-serving-mlir`'s `main.cpp`) and the Qwen2-specific naming
  assumptions in `tools/onnx_graph_to_facts.py`.
- **Torch FX adapter.** A second frontend adapter reading a traced/exported
  Torch FX graph directly (bypassing ONNX export entirely) and emitting the
  same `GraphFacts` schema, following the same adapter-seam pattern as the
  ONNX bridge. Not started.
- **StableHLO adapter.** A third frontend adapter over this repo's existing
  StableHLO textual subset tooling (`tools/import_stablehlo_subset.py`,
  `docs/MLIR_COMPILER_PIPELINE_SUMMARY.md`), emitting `GraphFacts` instead
  of going straight to HIR, so the same Qwen serving pipeline can be reached
  from a StableHLO-shaped source. Not started.
- **Layer-range compression.** Add `layer_range`/`repeat_count` as an
  additive field on `PerOpDecisionBundle`/the JSON schema, with a fold step
  that only collapses a range when decision content is verified identical
  across layers — never assumed. This is export-time only; the compiler's
  internal IR model stays full expansion regardless.
- **Decode-with-past import.** Neither frontend adapter handles
  `past_key_values` inputs/outputs or `use_cache_branch` control flow; both
  target a single (prefill-shaped) graph today. Also revisit whether
  `KVLayoutPlanningPass`'s `numLayers`-based KV-footprint formula should
  instead sum real per-op KV attrs now that real per-layer ops exist to sum.

## Qwen GTX 1650 vLLM Serving: Phase C (Quantized) — AWQ Minimal, GPTQ Not Implemented

`artifacts/qwen/execution_plan.json` (fp16, quantization `none`/`fp16_fallback`)
is the compiler-side source artifact for the measured A/B vLLM benchmark
recorded in `heterogeneous-inference-runtime` (`results/qwen_no_quant/`).

A minimal Phase C (AWQ only) is now implemented:

- `tools/export_qwen_awq.py` — real AutoAWQ export edge tool. Fails clearly
  (non-zero exit, install instructions, no fake artifact) when AutoAWQ is
  missing; this repo's development machine (macOS, no CUDA) cannot run the
  actual export — AutoAWQ has no CPU/macOS quantization kernel path. Output:
  `artifacts/qwen_awq/` + `provenance.json` (source model, method, tool
  version, calibration note, `created_at`, `truth_boundary`).
- `configs/target_profiles/nvidia_gtx1650_maxq_awq_forced.json` — an
  experimental forced-quant profile variant, distinct from
  `nvidia_gtx1650_maxq.json`. Its per-op `backendCapabilities` are byte-for-byte
  identical to the no-quant profile (`supportedQuantModes: ["none"]`) — this
  does **not** claim GTX 1650 gained native INT4 Tensor Core support. It adds
  one new top-level `forcedQuantization` block
  (`strategy: weight_only_int4`, `algorithm: awq`,
  `quantizedModelArtifactRef: artifacts/qwen_awq`,
  `truthBoundary: experimental_forced_quant_not_native_int4_support_on_gtx1650`)
  that `compile-for-target` reads and attaches as module attrs only when
  present — the unmodified no-quant profile produces a byte-identical
  `execution_plan.json` to before this change (verified: `diff` against the
  checked-in `artifacts/qwen/execution_plan.json` is empty).
- `tools/run_qwen_awq_compiler_pipeline.sh` — produces
  `artifacts/qwen_awq_plan/execution_plan.json`, whose
  `global_decisions.quantization` now carries `strategy`, `algorithm`,
  `quantized_model_artifact_ref`, and `truth_boundary` (extended
  `QuantizationDecision` in `mlir_passes/include/decision/Decision.h`,
  `ExecutionPlanBuilder.cpp`, `ExecutionPlanExporter.cpp`). Covered by CTest
  `QwenAwqForcedCompileTest`.
- Materializer update on the runtime side
  (`deployment/vllm_adapter/config_materializer.py` +
  `deployment/execution_plan/path_builder.py` in
  `heterogeneous-inference-runtime`): emits `--quantization awq` and points
  `--model`/`--tokenizer` at `quantized_model_artifact_ref` instead of the HF
  repo id, only when that ref is present. No-quant (B) materialization is
  unchanged (regression-tested).
- `scripts/run_qwen_quant_benchmark.sh` (in `heterogeneous-inference-runtime`)
  — A/B/C runner under `results/qwen_quant/`. Materializes all three server
  commands unconditionally; runs the actual servers/benchmarks only when
  vLLM is importable and the AWQ artifact exists locally. On this
  development machine (no vLLM, no AWQ artifact), it stops after
  materialization and still writes `quant_comparison.md` with truth
  boundaries — no measured C results exist yet.

Remaining/not implemented:

- GPTQ export and materialization. Only AWQ is implemented; extending to
  GPTQ means adding an `auto-gptq` export path in a new tool (or extending
  `export_qwen_awq.py`'s dependency check) and a second `algorithm: "gptq"`
  branch — the schema (`QuantizationDecision.algorithm`) already
  accommodates it, no schema change needed.
- No accuracy evaluation of the AWQ artifact (perplexity, task benchmarks).
- No measured C results — this requires actually running
  `tools/export_qwen_awq.py` and `scripts/run_qwen_quant_benchmark.sh` on a
  CUDA-capable Linux host with vLLM installed. Until then, C is
  materialized-only.
- Export-time layer-range compression for the AWQ-forced plan (same gap as
  the no-quant Phase 1 plan — see above).

## Documentation Work

- Maintain a single "implemented vs simulated" section in every high-level doc that discusses runtime behavior.
- Add regeneration instructions for each PNG and JSON family.
- Add a dependency matrix for CMake, Python, MLIR, Metal, CUDA, IREE, Torch-MLIR, and optional benchmark tools.
- Add a glossary for custom IR, HIR, lowered graph, execution plan, trace, artifact, and integration bundle.
