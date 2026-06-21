# Design Decisions

## Custom Toy IR Plus Separate MLIR Plugin

The repository uses a small custom C++ graph IR for runtime experiments and a separate real MLIR plugin for compiler-pass demonstrations.

Tradeoff:

- The custom IR is simple, readable, and easy to drive from demo apps.
- The MLIR plugin demonstrates realistic compiler infrastructure.
- The two systems are not a single unified compiler stack; the bridge is primarily JSON/artifact based.

Assumption: this split is intentional for portfolio/demo clarity rather than production integration.

## Simple Vector-Backed Graph Representation

`Graph` stores tensors and nodes in vectors, and nodes refer to tensors by integer ids.

Tradeoff:

- Easy to inspect and serialize.
- Minimal abstraction overhead.
- No robust ownership, aliasing, mutation, or graph rewrite safety guarantees.

Future work should be careful around id stability and mutation passes.

## Float32-First Tensor Model

The custom runtime `Tensor` supports `DType::Float32` only in the observed core IR.

Tradeoff:

- Keeps CPU kernel demos simple.
- Limits realism for quantized, mixed precision, and accelerator-backed execution.

MLIR HIR includes quantization-oriented ops, but that is not reflected as a full dtype system in the toy runtime.

## Backend Interface Over Full Device Runtime

`Backend` exposes a small `execute(Graph&, const Node&)` interface.

Tradeoff:

- Makes CPU, mock GPU, and Metal stubs easy to slot into the executor.
- Does not model real device buffers, command queues, kernel compilation, synchronization, or async errors.

Implemented backend reality:

- CPU executes registered kernels.
- MockGPU uses CPU kernels after logging simulated GPU dispatch.
- Generic Metal backend logs dispatch and Metal device info but does not run generic graph kernels.

## Cost-Based Planning With Estimated Models

`CostBasedPlanner` combines cost report entries, backend constants, transfer estimates, and static fallbacks.

Tradeoff:

- Useful for explaining backend placement decisions.
- Sensitive to placeholder constants and stale cost reports.
- Should not be treated as hardware-calibrated without fresh benchmarks.

Any future metrics from this planner should be labeled estimated unless they come from measured `actual_latency_ms` entries.

## JSON Artifacts as Integration Contracts

Many pipelines emit JSON files for lowered graphs, execution plans, memory plans, scheduler plans, validation manifests, and dashboards.

Tradeoff:

- Great for visualization, validation, and handoff to demos.
- Risk of drift between source code and generated artifacts.
- Schema validation is partial and mostly script-specific.

Future changes should either regenerate artifacts or clearly mark docs as describing source rather than current artifact snapshots.

## Demo-Oriented Executables

The `apps/` directory contains many focused demos and benchmark harnesses instead of one canonical product binary.

Tradeoff:

- Easy to showcase individual compiler/runtime concepts.
- Build and test surface is broad.
- Some demos are platform-specific or depend on external toolchains.

## Real MLIR Pass Infrastructure

`mlir_passes/` uses CMake, TableGen, dialect definitions, pass declarations, and FileCheck tests.

Tradeoff:

- This is closer to production compiler engineering.
- It introduces external LLVM/MLIR ABI/toolchain requirements.
- Failures may be environmental rather than source-level.

## Prefer Conservative Fusion

The MLIR fusion pass includes legality checks around use counts, shapes, dtype, bias shape, target tiling, padding overhead, and verifier metadata.

Tradeoff:

- Avoids claiming unsafe rewrites.
- Leaves unsupported cases unfused.
- Increases the amount of metadata and negative tests needed.

## Serving Pipeline as Planning, Not Serving

The LLM serving path emits contracts for prefill/decode split, KV cache planning, memory pressure, scheduling, and framework-style metrics.

Tradeoff:

- Communicates runtime policy clearly.
- Does not implement a production vLLM/SGLang/Triton/TensorRT serving engine.
- Estimated latency/throughput in candidate plans must be labeled estimated.

