# Future Work

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

## LLM Serving Artifact Work

- Label estimated latency, throughput, memory, and pressure metrics inside generated artifacts.
- Add schema version validation for every JSON artifact.
- Add tests that intentionally break KV cache capacity, memory budget, missing phases, and manifest completeness.
- Separate serving policy contracts from measured runtime traces.
- If real serving execution is added later, add measured TTFT/TPOT/throughput capture with environment metadata.

## Documentation Work

- Maintain a single "implemented vs simulated" section in every high-level doc that discusses runtime behavior.
- Add regeneration instructions for each PNG and JSON family.
- Add a dependency matrix for CMake, Python, MLIR, Metal, CUDA, IREE, Torch-MLIR, and optional benchmark tools.
- Add a glossary for custom IR, HIR, lowered graph, execution plan, trace, artifact, and integration bundle.

