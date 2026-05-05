# mini-ml-compiler

A C++ mini ML compiler and on-device inference runtime that implements graph IR, compiler-style optimizations, execution planning, and performance-optimized CPU kernels.

This project is designed to simulate core components of modern ML infrastructure systems such as Core ML Runtime, MLIR, and TVM, focusing on graph compilation, runtime execution, and performance optimization for on-device environments.

---

##  Features

- Graph-based Intermediate Representation (IR) for ML models
- Tensor metadata system (shape, dtype, memory layout)
- Pass-based compiler optimization pipeline
- Operator fusion (MatMul + Add + ReLU)
- Shape inference and graph verification
- DAG-based graph lowering and topological scheduling
- Execution plan abstraction for runtime decoupling
- Operator registry and runtime dispatch system
- CPU kernel implementations (baseline, tiled, threaded)
- Memory planning (arena-style allocation)
- End-to-end and kernel-level benchmarking tools

---

##  System Architecture

The system follows a compiler-style pipeline:

`
Graph IR
→ Shape Inference
→ Graph Verification
→ Optimization Passes (Fusion)
→ Memory Planning
→ Lowering (DAG Scheduling)
→ Execution Plan
→ Operator Dispatch
→ CPU Kernel Execution
`

This design separates model representation, optimization, and execution, enabling flexible and extensible runtime behavior.

---

##  Project Structure

```
include/
  ir/              # Graph, Node, Tensor definitions
  pass/            # Pass interface and optimization passes
  runtime/         # Execution plan, executor, operator registry
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
  run_demo.cpp           # End-to-end pipeline demo
  benchmark_matmul.cpp   # End-to-end runtime benchmark
  benchmark_kernels.cpp  # Kernel-level benchmark
  benchmark_threads.cpp  # Thread scaling benchmark
```

---

##  Build Instructions

### Requirements
- C++17 compatible compiler
- CMake >= 3.16

### Build

`bash
mkdir build
cd build
cmake ..
cmake --build . --config Release
`

### Run

`bash
./Release/run_demo.exe
./Release/benchmark_matmul.exe
./Release/benchmark_kernels.exe
./Release/benchmark_threads.exe
`

---

##  Benchmark Results

### Kernel Optimization (256x256)

- Baseline: 21.98 ms
- Optimized (tiled, i-k-j): 8.68 ms
- Speedup: **2.53×**
- Correctness: PASSED

---

### Thread Scaling (512x512)

| Threads | Latency (ms) | Speedup |
|--------|-------------|--------|
| 1      | 74.55       | 1.0×   |
| 2      | 71.77       | 1.04×  |
| 4      | 48.47       | 1.54×  |
| 8      | 35.83       | 2.08×  |

Observations:
- Threading introduces overhead at low parallelism
- Scaling improves with larger workloads
- Performance is bounded by CPU scheduling and memory bandwidth

---

##  Memory Access Optimization Insights

During optimization, multiple tiled loop orderings were evaluated.

Two key variants were tested:

- **i-k-j ordering (optimized)**
- **i-j-k ordering (degraded performance)**

### i-k-j (selected)

`
for i
  for k
    load A[i, k]
    for j
      C[i, j] += A[i, k] * B[k, j]
`

Advantages:
- Sequential access on B (row-major)
- Sequential writes on C
- High reuse of A[i,k]
- Cache-friendly and SIMD-friendly

### i-j-k (rejected)

`
for i
  for j
    for k
      C[i, j] += A[i, k] * B[k, j]
`

Problems:
- B is accessed with large strides (poor locality)
- Cache miss rate increases significantly
- Prefetching and vectorization are less effective

### Result

`
i-k-j tiled: 8.68 ms
i-j-k tiled: 13.96 ms
`

Conclusion:
Loop ordering has a major impact on cache locality and kernel performance.
The i-k-j schedule was selected based on empirical benchmarking.

---

##  Key Techniques

### 1. Operator Fusion
Combines multiple operators into a single kernel to reduce dispatch overhead and improve cache locality.

### 2. Tiled Matrix Multiplication
Improves cache efficiency by blocking memory access patterns.

### 3. DAG Scheduling
Ensures correct execution order using topological sorting based on tensor dependencies.

### 4. Operator Registry
Decouples execution logic from kernel implementations, enabling modular runtime design.

### 5. Shape Inference
Propagates tensor shapes through the graph for validation and memory planning.

### 6. Memory Planning
Uses an arena-style allocation strategy to reduce dynamic memory overhead.

---

##  Motivation

Modern ML systems on edge devices require:

- Efficient execution under resource constraints
- Compiler-style optimizations for performance
- Modular runtime systems for flexibility
- Fine-grained control over memory and execution

This project explores these challenges by implementing a simplified but realistic ML compiler and runtime system in C++.

---

##  Future Work

- ONNX / PyTorch model import
- Thread pool for persistent parallel execution
- SIMD / vectorized kernels
- Memory reuse and lifetime analysis
- Quantization (INT8 / FP16)
- Transformer operator support
- Profiling and latency breakdown tools
- Python API for model loading and execution

---
