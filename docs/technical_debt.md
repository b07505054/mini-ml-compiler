# Technical Debt

## Weak Spots

- The custom graph IR has no strong validation around tensor id lifetimes, graph mutation safety, or op-specific shape contracts.
- `Tensor` is effectively float32-only in the custom runtime, while docs and MLIR artifacts discuss quantization and accelerator layouts.
- The generic backend interface is too small for real device execution because it does not model device memory, async execution, command queues, or synchronization.
- `MockGPUBackend` is useful for dispatch demos but can be mistaken for real GPU execution.
- Generic `MetalBackend` does not execute graph kernels; separate Metal demo paths should be documented separately from runtime backend support.
- Planner estimates depend on hardcoded backend constants and fallback latency tables.
- Generated artifacts can drift from source code because many files under `trace/`, `artifacts/`, and `integration_bundle/` are checked in.
- Some code paths export to relative paths such as `../trace/...`, which depends on running from expected build directories.
- Error handling is inconsistent across demos, tools, and runtime components.

## Missing or Thin Tests

- The custom C++ graph runtime appears to rely heavily on demos/benchmarks rather than focused unit tests.
- CPU kernels need more correctness tests for shape mismatch, broadcasting assumptions, edge shapes, and numerical tolerance.
- Memory planner needs tests for persistent tensors, zero-sized/empty tensors, overlapping lifetimes, and reuse safety.
- Backend scheduler/provider selection needs deterministic tests that assert placement decisions.
- Cost planner needs tests for fallback behavior, runtime-observed override behavior, transfer costs, and missing cost entries.
- LLM serving artifact validation checks structure but does not prove runtime execution.
- Cross-platform build paths for CUDA, Metal, and MLIR likely need CI matrix coverage.

## Duplicated or Overlapping Logic

- Multiple planning concepts exist: static schedule, execution plan, execution plan v2, provider scheduler, backend scheduler, cost planner, runtime replanner.
- CV and LLM artifact generation have separate schemas and validation logic.
- Runtime-facing metadata appears in C++ structs, JSON traces, MLIR attributes, and Python scripts without a single schema authority.
- Several demos likely construct similar graphs manually.

## Unclear Naming

- `MetalBackend` sounds like a real backend, but the generic path is currently a dispatch/device logging stub.
- `MockGPUBackend` is accurate, but downstream docs should keep repeating that it uses CPU kernels.
- `runtime` is used for both implemented C++ dispatch and simulated planning artifacts.
- `compiler` is used for custom toy graph passes, MLIR passes, and Python artifact lowering.
- `trace` files may be generated artifacts, validation reports, synthetic timelines, or benchmark outputs.

## Future Risks

- Overclaiming simulated behavior as implemented execution is the largest documentation and maintenance risk.
- Stale committed artifacts may cause reviewers or future agents to debug generated JSON instead of source logic.
- Adding more demos without shared test infrastructure will increase maintenance cost.
- Expanding dtype/backend support in one layer without matching other layers will widen the custom IR versus MLIR gap.
- Platform-specific code paths may silently bit-rot on machines without Metal, CUDA, MLIR, IREE, or Torch-MLIR.
- Hardcoded benchmark or planner numbers may be copied into docs as if they were stable measurements.

