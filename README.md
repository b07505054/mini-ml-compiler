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

### Compiler and Runtime Infrastructure

- ONNX → custom MLIR-style IR → Graph IR ingestion pipeline
- Graph lowering into execution plans
- Backend-aware execution scheduler
- Heterogeneous runtime execution
- Graph partitioning across execution backends
- Execution planner and dispatch abstraction
- Runtime profiling and execution tracing
- Per-operator latency analysis

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
- Compiler-style graph optimization passes

### CPU and SIMD Optimization

- Cache-aware tiled matrix multiplication
- Thread pool based parallel execution
- Persistent worker scheduling
- Task granularity optimization
- AVX2 SIMD vectorized ReLU
- AVX2 SIMD vectorized Add
- AVX2 SIMD vectorized MatMul
- Hardware-aware kernel optimization

### Transformer Runtime Features

- Transformer scaled dot-product attention operator
- Numerically stable Softmax implementation
- Causal attention masking
- KV cache simulation
- Incremental autoregressive decoding simulation
- Attention correctness validation against NumPy reference

### MLIR / Compiler Ecosystem

- MLIR-style intermediate representation
- MLIR optimization workflow experimentation
- Affine loop transformation experiments
- Compiler-oriented runtime architecture

### Benchmarking and Analysis

- End-to-end runtime benchmarking
- Kernel-level microbenchmarking
- Thread scaling analysis
- GPU-style tiled execution simulation
- Runtime profiling summaries
- Memory optimization analysis

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
→ Optimization Passes (Fusion)
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

## Project Structure

```text
include/
  ir/              # Graph, Node, Tensor definitions
  pass/            # Pass interface and optimization passes
  runtime/         # Execution plan, executor, backend system
  kernels/         # CPU kernels (MatMul, fused ops)
  analysis/        # Shape inference and graph verification
  utils/           # Benchmark utilities

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
```

---

## Build Instructions

### Requirements

- C++17 compatible compiler
- CMake >= 3.16

### Build

```bash
mkdir build
cd build
cmake ..
cmake --build . --config Release
```

### Run

```bash
./Release/run_demo.exe
./Release/run_backend_demo.exe
./Release/run_partition_demo.exe
./Release/run_scheduled_backend_demo.exe
./Release/run_dynamic_shape_demo.exe
./Release/benchmark_avx2_matmul.exe
./Release/benchmark_int8_matmul.exe
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

### Thread Scaling (512×512)

| Threads | Latency (ms) | Speedup |
|---|---|---|
| 1 | 74.55 | 1.0× |
| 2 | 71.77 | 1.04× |
| 4 | 48.47 | 1.54× |
| 8 | 35.83 | 2.08× |

Observations:

- Scaling improves with larger workloads
- Performance is bounded by scheduling overhead and memory bandwidth
- Persistent worker pools reduce thread creation overhead

---

## Dynamic Shape Runtime

The runtime supports dynamic-shape execution with runtime shape propagation and dynamic tensor allocation.

Execution correctness was validated across multiple batch sizes:

```text
Batch=1 → Output shape [1,2]
Batch=2 → Output shape [2,2]
Batch=4 → Output shape [4,2]
```

Memory planning dynamically adapts to tensor sizes during execution:

```text
Batch=1 → Peak memory: 15 floats
Batch=2 → Peak memory: 24 floats
Batch=4 → Peak memory: 42 floats
```

This simulates real-world inference workloads with variable input sizes and dynamic execution requirements.

---

## Transformer Attention Runtime

The runtime was extended with Transformer-style scaled dot-product attention:

```text
Attention(Q, K, V) = softmax(QKᵀ / √D)V
```

Implemented components include:

- Attention score computation
- Numerically stable Softmax
- Attention-weighted value aggregation
- Causal attention masking
- KV cache reuse
- Incremental autoregressive decoding simulation

Attention correctness was validated against a NumPy reference:

```text
Python reference:
6.224593 7.550813
3.775407 12.449187

C++ runtime:
6.22459 7.55081
3.77541 12.4492
```

KV cache simulation example:

```text
Step 0 output:
10 0 0 0

Step 1 output using KV cache:
3.77541 12.4492 0 0
```

---

## Runtime Profiling Example

```text
=== Runtime Profiling Summary ===
matmul [MockGPU] : 0.0778 ms
add [CPU]        : 0.0006 ms
relu [CPU]       : 0.0005 ms

Total latency: 0.0789 ms
```

The profiling infrastructure supports:

- Per-operator latency tracing
- Backend-aware profiling
- Runtime execution summaries
- Scheduling analysis
- Memory optimization analysis

---

## Memory Planning and Optimization

The runtime includes compiler-style memory planning and tensor lifetime analysis.

Implemented systems include:

- Tensor first/last-use analysis
- Arena-style memory allocation
- Buffer reuse optimization
- Peak memory analysis

Example memory reuse:

```text
output reuses buffer from matmul_out
```

Example memory optimization:

```text
Naive memory: 50 float elements
Planned peak memory: 42 float elements
Saved memory: 8 float elements
```

---

## ARM NEON Runtime Portability

To support mobile and edge-oriented runtime portability, ARM NEON-aware execution paths were added.

Implemented components include:

- ARM NEON vectorized Add kernel
- Compile-time backend selection
- Scalar fallback on non-ARM platforms

```cpp
#ifdef __ARM_NEON
```

This enables runtime portability across ARM-based edge devices commonly used in Qualcomm and mobile inference environments.

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
- FlashAttention-style fused attention kernels
- Real GPU backend integration (CUDA / Metal)
- Async execution scheduling
- Runtime memory fragmentation analysis
- ONNX graph-level optimization passes
- MLIR dialect lowering
- Vectorized Transformer kernels
- Operator autotuning
- Mixed backend execution optimization

---