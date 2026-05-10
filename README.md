# Heterogeneous ML Compiler and Runtime System

A C++ heterogeneous ML compiler and runtime system that implements graph IR, compiler-style optimizations, backend-aware execution scheduling, dynamic-shape execution, quantized inference, and performance-optimized CPU kernels for edge-oriented machine learning workloads.

This project simulates core components of modern ML infrastructure systems such as Core ML Runtime, TensorRT, MLIR, and TVM, focusing on graph compilation, heterogeneous runtime execution, memory optimization, and hardware-aware inference for on-device and edge environments.

---

## Features

- Graph-based Intermediate Representation (IR) for ML models
- Tensor metadata system (shape, dtype, memory layout)
- Pass-based compiler optimization pipeline
- Shape inference and graph verification
- DAG-based graph lowering and topological scheduling
- Execution plan abstraction for runtime decoupling
- Operator registry and runtime dispatch system
- Dynamic-shape runtime execution
- INT8 quantized runtime path
- AVX2 SIMD vectorized GEMM kernels
- ARM NEON-aware execution path
- Heterogeneous backend scheduling
- Graph partitioning across execution backends
- Algebraic simplification compiler pass
- Dead node elimination compiler pass
- MLIR affine and linalg dialect workflow integration
- HuggingFace Transformer ONNX ingestion workflow
- ONNX graph inspection and operator coverage analysis
- Runtime trace export tooling
- FlashAttention-style fused attention kernel
- LayerNorm runtime operator onboarding
- CoreML conversion workflow experimentation

### Compiler and Runtime Infrastructure

- ONNX → custom MLIR-style IR → Graph IR ingestion pipeline
- Graph lowering into execution plans
- Backend-aware execution scheduler
- Heterogeneous runtime execution
- Graph partitioning across execution backends
- Execution planner and dispatch abstraction
- Runtime profiling and execution tracing
- Per-operator latency analysis
- Runtime trace JSON export

### Backend System

- Backend abstraction layer
- CPU backend implementation
- Mock GPU backend simulation
- Backend-aware operator dispatch
- Simulated heterogeneous execution
- Backend-level profiling and scheduling

### Compiler Optimizations

- Operator fusion (MatMul + Add + ReLU)
- Tensor lifetime analysis
- Arena-style memory planning
- Memory reuse analysis
- Peak memory reporting
- Dead node elimination
- Algebraic simplification
- Compiler-style graph optimization passes

### CPU and SIMD Optimization

- Cache-aware tiled matrix multiplication
- Thread pool based parallel execution
- Persistent worker scheduling
- Task granularity optimization
- AVX2 SIMD vectorized ReLU
- AVX2 SIMD vectorized Add
- AVX2 SIMD vectorized MatMul
- INT8 quantized MatMul
- Hardware-aware kernel optimization

### Transformer Runtime Features

- Transformer scaled dot-product attention operator
- FlashAttention-style fused attention kernel
- Numerically stable Softmax implementation
- LayerNorm runtime kernel
- Causal attention masking
- KV cache simulation
- Incremental autoregressive decoding simulation
- Attention correctness validation against NumPy reference

### MLIR / Compiler Ecosystem

- MLIR affine dialect workflow
- MLIR linalg dialect workflow
- Real MLIR pass pipeline using mlir-opt
- Canonicalization and CSE passes
- Affine loop tiling
- Linalg-to-affine lowering
- Compiler-oriented runtime architecture

### HuggingFace / ONNX Integration

- HuggingFace Transformer model export workflow
- PyTorch → ONNX export
- ONNX graph inspection tooling
- ONNX operator coverage analysis
- Runtime operator onboarding workflow

### Benchmarking and Analysis

- End-to-end runtime benchmarking
- Kernel-level microbenchmarking
- Thread scaling analysis
- GPU-style tiled execution simulation
- Runtime profiling summaries
- Memory optimization analysis

---

## MLIR Compiler Workflow

This project includes real MLIR compiler workflow experiments using `mlir-opt`.

Implemented MLIR passes:

- Canonicalization
- Common subexpression elimination (CSE)
- Affine loop tiling
- Linalg-to-affine lowering

Example commands:

```bash
mlir-opt mlir/matmul_affine.mlir --canonicalize --cse
mlir-opt mlir/matmul_affine.mlir --affine-loop-tile="tile-sizes=32,32,32"
mlir-opt mlir/matmul_linalg.mlir --convert-linalg-to-affine-loops
```

Example affine loop tiling output:

```mlir
affine.for %arg3 = 0 to 128 step 32
affine.for %arg4 = 0 to 128 step 32
affine.for %arg5 = 0 to 128 step 32
```

This demonstrates interaction with real MLIR dialects including `affine` and `linalg`, connecting the custom runtime project to modern compiler infrastructure.

---

## HuggingFace / ONNX Model Integration

Exported a HuggingFace tiny BERT model from PyTorch to ONNX and implemented ONNX graph inspection tooling.

Example graph statistics:

- Nodes: 273
- Initializers: 59
- MatMul: 40
- Add: 52
- Softmax: 5
- LayerNormalization: 11
- Transpose: 25

Example ONNX operator coverage report:

- Total nodes: 273
- Supported nodes: 98
- Partially supported nodes: 80
- Unsupported nodes: 95
- Approx operator coverage: 65.2%

This validates real Transformer model ingestion and provides debugging visibility into exported inference graphs.

Attempted PyTorch-to-CoreML conversion workflows using coremltools and HuggingFace Transformer models, identifying compatibility limitations across Windows environments, Torch versions, and SDPA-based attention implementations.

---

## System Architecture

The system follows a compiler-style pipeline:

```text
PyTorch Model
→ ONNX Export
→ Custom .mlir IR
→ Graph IR
→ Shape Inference
→ Graph Verification
→ Optimization Passes
→ Memory Planning
→ Lowering (DAG Scheduling)
→ Execution Plan
→ Backend Scheduler
→ Operator Dispatch
→ CPU / MockGPU Backend
→ Kernel Execution
```

The runtime also includes Transformer-oriented execution components such as causal attention masking, KV cache reuse, and incremental decoding simulation for autoregressive inference workloads.

This design separates model representation, optimization, scheduling, and execution, enabling flexible and extensible runtime behavior.

---

## Heterogeneous Runtime Architecture

The runtime supports heterogeneous execution through a backend abstraction layer inspired by modern ML runtimes such as TensorRT, TVM, and Core ML Runtime.

Execution flow:

```text
Graph IR
→ Execution Planner
→ Backend Scheduler
→ Backend Dispatcher
→ CPU Backend / Mock GPU Backend
```

Implemented runtime systems include:

- Backend abstraction interface
- CPU backend execution
- Mock GPU backend simulation
- Backend-aware operator scheduling
- Graph partitioning across execution backends
- Runtime dispatch abstraction
- Backend-level profiling

Example heterogeneous scheduling:

```text
MatMul → MockGPU
Add    → CPU
ReLU   → CPU
```

This architecture simulates heterogeneous compute execution commonly used in modern edge inference systems.

---

## Runtime Trace Export

The runtime profiler can export backend-aware execution traces to JSON.

Example:

```json
[
  {
    "op_name": "matmul",
    "backend": "MockGPU",
    "latency_ms": 0.1093
  },
  {
    "op_name": "add",
    "backend": "CPU",
    "latency_ms": 0.0006
  },
  {
    "op_name": "relu",
    "backend": "CPU",
    "latency_ms": 0.0005
  }
]
```

This enables debugging and visualization of heterogeneous runtime execution.

---

## Custom Compiler Passes

The project includes custom graph compiler optimization passes operating directly on the internal Graph IR.

Implemented compiler passes include:

- Operator fusion
- Dead node elimination
- Algebraic simplification
- Tensor lifetime analysis
- Memory reuse optimization

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

These passes simulate real compiler IR transformation workflows used in modern ML compiler systems.

---

## Project Structure

```text
include/
  ir/
  pass/
  runtime/
  kernels/
  analysis/
  utils/

src/
  ir/
  pass/
  runtime/
  kernels/
  analysis/

apps/
  run_demo.cpp
  run_backend_demo.cpp
  run_partition_demo.cpp
  run_scheduled_backend_demo.cpp
  run_dynamic_shape_demo.cpp
  run_dce_demo.cpp
  run_algebraic_simplification_demo.cpp
  run_layernorm_demo.cpp
  run_fused_attention_demo.cpp
  benchmark_matmul.cpp
  benchmark_kernels.cpp
  benchmark_threads.cpp
  benchmark_simd_relu.cpp
  benchmark_simd_add.cpp
  benchmark_avx2_matmul.cpp
  benchmark_int8_matmul.cpp
  benchmark_neon_add.cpp
  run_attention_demo.cpp
  run_causal_attention_demo.cpp
  run_kv_cache_demo.cpp

mlir/
  matmul_affine.mlir
  matmul_linalg.mlir
```

---

## Benchmark Results

### AVX2 SIMD MatMul (128×128)

| Kernel | Latency |
|---|---|
| Scalar MatMul | 2.20 ms |
| AVX2 Vectorized MatMul | 0.325 ms |

Speedup: **6.77×**

Correctness: PASSED

---

### INT8 Quantized MatMul (128×128)

| Runtime | Latency |
|---|---|
| FP32 Scalar MatMul | 2.43 ms |
| INT8 Quantized MatMul | 1.60 ms |

Speedup: **1.53×**

Maximum absolute error:

```text
4.89e-09
```

---

### AVX2 ReLU (16M elements)

| Kernel | Latency |
|---|---|
| Scalar ReLU | 12.92 ms |
| AVX2 Vectorized ReLU | 9.71 ms |

Speedup: **1.33×**

---

### AVX2 Add (16M elements)

| Kernel | Latency |
|---|---|
| Scalar Add | 26.42 ms |
| AVX2 Vectorized Add | 15.66 ms |

Speedup: **1.69×**

---

### FlashAttention-style Fused Attention

| Operator | Latency |
|---|---|
| fused_attention | 0.0574 ms |

Implemented a memory-aware fused attention operator that combines:

- QKᵀ score computation
- Numerically stable Softmax
- Attention-weighted value aggregation

into a single runtime operator to reduce intermediate tensor materialization and memory bandwidth overhead.

---

## Motivation

Modern ML systems on edge devices require:

- Efficient execution under resource constraints
- Compiler-style optimizations for performance
- Modular runtime systems for flexibility
- Fine-grained control over memory and execution
- Hardware-aware inference scheduling
- Heterogeneous compute support

This project explores these challenges by implementing a simplified but realistic ML compiler and runtime system in C++.

---

## Future Work

- Multi-head attention kernels
- Real GPU backend integration (CUDA / Metal)
- Async execution scheduling
- Runtime memory fragmentation analysis
- ONNX graph-level optimization passes
- MLIR dialect lowering
- Vectorized Transformer kernels
- Operator autotuning
- Mixed backend execution optimization
- Apple MLX benchmarking on Apple Silicon
- CoreML runtime execution on macOS
- Metal compute backend
```