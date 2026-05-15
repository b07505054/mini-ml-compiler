# Heterogeneous ML Compiler and Runtime System

A C++ heterogeneous ML compiler and runtime system that implements graph IR, compiler-style optimizations, backend-aware execution scheduling, subgraph delegation, memory reuse allocation, dynamic-shape execution, quantized inference, and hardware-aware Transformer kernels for edge-oriented machine learning workloads.

This project simulates core components of modern ML infrastructure systems such as Core ML Runtime, TensorRT, ONNX Runtime, MLIR, TVM, and ExecuTorch, focusing on graph compilation, heterogeneous runtime execution, memory optimization, compiler lowering, and hardware-aware inference for on-device and edge environments.

---

## Features

- Graph-based Intermediate Representation (IR) for ML models
- Tensor metadata system (shape, dtype, memory layout)
- Pass-based compiler optimization pipeline
- Shape inference and graph verification
- DAG-based graph lowering and topological scheduling
- Dependency-aware parallel execution engine
- Execution plan abstraction for runtime decoupling
- Operator registry and runtime dispatch system
- Dynamic-shape runtime execution
- INT8 quantized runtime path
- AVX2 SIMD vectorized GEMM kernels
- ARM NEON-aware execution path
- Heterogeneous backend scheduling
- Subgraph-level delegation and partitioning
- Execution-provider capability query system
- CPU fallback execution
- Algebraic simplification compiler pass
- Dead node elimination compiler pass
- MLIR affine and linalg dialect workflow integration
- Graph IR → MLIR emission pipeline
- HuggingFace Transformer ONNX ingestion workflow
- ONNX graph inspection and operator coverage analysis
- Runtime trace export tooling
- FlashAttention-style fused attention kernel
- SIMD tiled attention kernel
- LayerNorm runtime operator onboarding
- Arena allocator and free-list allocator v2
- Best-fit memory reuse policy
- Fragmentation analysis and block coalescing
- Async runtime execution
- Runtime profiling and execution tracing
- CoreML conversion workflow experimentation

---

## Compiler and Runtime Infrastructure

- ONNX → custom MLIR-style IR → Graph IR ingestion pipeline
- Graph lowering into execution plans
- Backend-aware execution scheduler
- Dependency-aware execution engine
- Heterogeneous runtime execution
- Subgraph partitioning across execution backends
- Execution-provider abstraction
- Runtime profiling and execution tracing
- Per-operator latency analysis
- Runtime trace JSON export
- Graph IR → MLIR emission pipeline
- MLIR affine/linalg lowering workflow

---

## Backend System

- Backend abstraction layer
- CPU backend implementation
- Mock GPU backend simulation
- Backend-aware operator dispatch
- Simulated heterogeneous execution
- Backend-level profiling and scheduling
- Subgraph delegation
- CPU fallback execution path

Example delegated execution:

```text
Subgraph 0 -> MockGPU
  matmul1
  matmul2

Subgraph 1 -> CPU
  add
  relu
```

This simulates execution-provider delegation architectures used in systems such as ONNX Runtime, TensorRT delegate execution, CoreML delegation, and mobile inference runtimes.

---

## Compiler Optimizations

- Operator fusion (MatMul + Add + ReLU)
- Tensor lifetime analysis
- Arena-style memory planning
- Memory reuse analysis
- Peak memory reporting
- Dead node elimination
- Algebraic simplification
- Compiler-style graph optimization passes

Example algebraic simplification:

```text
ReLU(ReLU(x)) → ReLU(x)
```

Example dead node elimination:

Before:

```text
live_matmul
live_add
dead_matmul
dead_relu
```

After:

```text
live_matmul
live_add
```

---

## Memory Runtime System

Implemented multiple runtime allocation systems inspired by modern inference runtimes.

### Arena Allocator

- Offset-based tensor memory binding
- Lifetime-aware tensor placement
- Peak memory analysis

### Free-List Allocator v2

- Best-fit allocation policy
- Block splitting
- Block coalescing
- Fragmentation reporting
- Reuse-aware allocation strategy

Example reuse behavior:

```text
After freeing B:
offset=16 size=8 free=true

After allocating D=6:
D offset: 16
```

This simulates memory reuse strategies used in production inference runtimes.

---

## Transformer Runtime Features

- Transformer scaled dot-product attention operator
- FlashAttention-style fused attention kernel
- SIMD tiled attention kernel
- Numerically stable Softmax implementation
- LayerNorm runtime kernel
- Causal attention masking
- KV cache simulation
- Incremental autoregressive decoding simulation
- Attention correctness validation against NumPy reference

### Tiled Attention Kernel

Implemented a tile-aware attention kernel that traverses Key/Value tensors block-by-block to reduce intermediate memory pressure and improve cache locality.

Pipeline:

```text
QKᵀ tiled score computation
→ numerically stable Softmax
→ tiled weighted Value accumulation
```

This simulates hardware-aware Transformer execution strategies commonly used in modern inference systems.

---

## MLIR / Compiler Ecosystem

This project integrates real MLIR compiler workflows using `mlir-opt`.

Implemented MLIR passes:

- Canonicalization
- Common subexpression elimination (CSE)
- Affine loop tiling
- Linalg-to-affine lowering

Example commands:

```bash
mlir-opt generated_graph.mlir --canonicalize --cse
mlir-opt generated_graph.mlir --convert-linalg-to-affine-loops
```

Example generated MLIR:

```mlir
linalg.matmul ins(%A, %B : memref<128x128xf32>, memref<128x128xf32>) outs(%C : memref<128x128xf32>)
```

Generated affine lowering:

```mlir
affine.for %arg3 = 0 to 128 {
  affine.load
  arith.mulf
  affine.store
}
```

This connects the custom compiler pipeline to the MLIR ecosystem and modern compiler infrastructure.

---

## HuggingFace / ONNX Integration

Exported HuggingFace Transformer models from PyTorch to ONNX and implemented ONNX graph inspection and operator onboarding tooling.

Implemented tooling:

- HuggingFace Transformer export workflow
- PyTorch → ONNX export
- ONNX graph inspection
- ONNX operator coverage analysis
- Runtime operator onboarding report generation
- Operator onboarding stub generation

Example graph statistics:

- Nodes: 273
- Initializers: 59
- MatMul: 40
- Add: 52
- Softmax: 5
- LayerNormalization: 11

Example operator coverage:

- Supported nodes: 98
- Partially supported nodes: 80
- Unsupported nodes: 95
- Approx operator coverage: 69.23%

This validates real Transformer model ingestion and runtime operator onboarding workflows.

---

## Parallel Runtime Execution

Implemented a dependency-aware parallel execution engine that identifies ready nodes from the execution graph and launches independent operations asynchronously.

Example graph:

```text
matmul1 ─┐
         ├─ add → relu
matmul2 ─┘
```

Example execution:

```text
[ParallelExecutor] Launching ready node: matmul1 -> MockGPU
[ParallelExecutor] Launching ready node: matmul2 -> MockGPU
[ParallelExecutor] Launching ready node: add -> CPU
[ParallelExecutor] Launching ready node: relu -> CPU
```

This simulates dependency-aware execution engines used in production ML runtimes.

---

## System Architecture

The system follows a compiler-style ML execution pipeline:

```text
PyTorch Model
→ ONNX Export
→ Graph IR
→ Shape Inference
→ Graph Verification
→ Compiler Optimization Passes
→ Memory Planning
→ Graph Partitioning
→ Execution Plan
→ Backend Scheduler
→ Subgraph Delegation
→ Operator Dispatch
→ CPU / MockGPU Backend
→ Kernel Execution
```

The runtime also includes Transformer-oriented execution systems such as causal attention masking, KV cache reuse, tiled attention execution, and incremental decoding simulation.

---

## Benchmarking and Analysis

### AVX2 SIMD MatMul (128×128)

| Kernel | Latency |
|---|---|
| Scalar MatMul | 2.20 ms |
| AVX2 Vectorized MatMul | 0.325 ms |

Speedup: **6.77×**

---

### INT8 Quantized MatMul (128×128)

| Runtime | Latency |
|---|---|
| FP32 Scalar MatMul | 2.43 ms |
| INT8 Quantized MatMul | 1.60 ms |

Speedup: **1.53×**

---

### FlashAttention-style Fused Attention

| Operator | Latency |
|---|---|
| fused_attention | 0.0574 ms |

---

### SIMD Tiled Attention

| Operator | Latency |
|---|---|
| tiled_attention | 0.0116 ms |

---

## Future Work

- Metal backend integration
- CUDA backend integration
- Vectorized Transformer kernels
- Operator autotuning
- Multi-head attention kernels
- MLIR dialect lowering expansion
- Apple MLX benchmarking on Apple Silicon
- CoreML runtime execution on macOS
- Runtime memory fragmentation visualization