@# mini-ml-compiler

@A C++ mini ML compiler and on-device inference runtime that implements graph IR, compiler-style optimizations, execution planning, and performance-optimized CPU kernels.

@This project is designed to simulate core components of modern ML infrastructure systems such as Core ML Runtime, MLIR, and TVM, focusing on graph compilation, runtime execution, and performance optimization for on-device environments.

---

@##  Features

@- Graph-based Intermediate Representation (IR) for ML models
@- Tensor metadata system (shape, dtype, memory layout)
@- Pass-based compiler optimization pipeline
@- Operator fusion (MatMul + Add + ReLU)
@- Shape inference and graph verification
@- DAG-based graph lowering and topological scheduling
@- Execution plan abstraction for runtime decoupling
@- Operator registry and runtime dispatch system
@- CPU kernel implementations (baseline, tiled, threaded)
@- Memory planning (arena-style allocation)
@- End-to-end and kernel-level benchmarking tools
@- PyTorch → ONNX → custom IR model ingestion pipeline
@- Transformer-style scaled dot-product attention operator
@- Attention correctness validation against NumPy reference implementation
---

@##  System Architecture

@The system follows a compiler-style pipeline:

@```

@PyTorch Model
@→ ONNX Export
@→ Custom .mlir IR
@→ Graph IR
@→ Shape Inference
@→ Graph Verification
@→ Optimization Passes (Fusion)
@→ Memory Planning
@→ Lowering (DAG Scheduling)
@→ Execution Plan
@→ Operator Dispatch
@→ CPU Kernel Execution
@The runtime also includes Transformer-oriented execution components such as causal attention masking, KV cache reuse, and incremental decoding simulation for autoregressive inference workloads.
@```


@This design separates model representation, optimization, and execution, enabling flexible and extensible runtime behavior.

---

@##  Project Structure

@```
@include/
@  ir/              # Graph, Node, Tensor definitions
@  pass/            # Pass interface and optimization passes
@  runtime/         # Execution plan, executor, operator registry
@  kernels/         # CPU kernels (MatMul, fused ops)
@  analysis/        # Shape inference and graph verification
@  utils/           # Benchmark utilities

@src/
@  ir/
@  pass/
@  runtime/
@  kernels/
@  analysis/

@apps/
@  run_demo.cpp           # End-to-end pipeline demo
@  benchmark_matmul.cpp   # End-to-end runtime benchmark
@  benchmark_kernels.cpp  # Kernel-level benchmark
@  benchmark_threads.cpp  # Thread scaling benchmark
@  run_attention_demo.cpp          # Transformer attention demo
@  run_causal_attention_demo.cpp   # Causal masked attention
@  run_kv_cache_demo.cpp           # Incremental decoding with KV cache
@```

---

@##  Build Instructions

@### Requirements
@- C++17 compatible compiler
@- CMake >= 3.16

@### Build

@```bash
@mkdir build
@cd build
@cmake ..
@cmake --build . --config Release
@```

@### Run

@```bash
@./Release/run_demo.exe
@./Release/benchmark_matmul.exe
@./Release/benchmark_kernels.exe
@./Release/benchmark_threads.exe
@```

---

@##  Benchmark Results

@### Kernel Optimization (256x256)

@- Baseline: 21.98 ms
@- Optimized (tiled, i-k-j): 8.68 ms
@- Speedup: **2.53×**
@- Correctness: PASSED

---

@### Thread Scaling (512x512)

@| Threads | Latency (ms) | Speedup |
@|--------|-------------|--------|
@| 1      | 74.55       | 1.0×   |
@| 2      | 71.77       | 1.04×  |
@| 4      | 48.47       | 1.54×  |
@| 8      | 35.83       | 2.08×  |

@Observations:
@- Threading introduces overhead at low parallelism
@- Scaling improves with larger workloads
@- Performance is bounded by CPU scheduling and memory bandwidth
@---

@##  GPU-style Execution Simulation

@To better understand GPU execution models, a CPU-based simulator was implemented to mimic:

@- Grid / Block / Thread hierarchy
@- Per-thread computation mapping
@- Tiled execution patterns (simulated shared memory)

@### Naive GPU-style Execution

@Each thread computes one output element:

@- Average latency: 17.83 ms
@- Output correctness: PASSED

@This approach introduces overhead due to sequential CPU execution of all simulated threads.

---

@### Tiled GPU-style Execution (Simulated)

@A tiled execution model was implemented to simulate shared memory behavior:

@- Tile size: 16 × 16
@- Average latency: 178.49 ms
@- Output correctness: PASSED

@---

@##  Thread Pool and Task Scheduling

@To improve parallel execution efficiency, a persistent thread pool was implemented to replace per-run thread creation.

@### Motivation

@Initial implementation used `std::thread` per execution, which introduced significant overhead:

@- Thread creation and destruction cost
@- Poor scalability at low thread counts

@### Persistent Thread Pool

@A worker pool with task queue was introduced:

@- Fixed number of worker threads
@- Task queue with condition variable synchronization
@- Reused threads across multiple executions

@### Performance Observations

@Persistent thread pool reduced overhead compared to per-run thread spawning, but performance depended heavily on task granularity.

@### Task Granularity Sweep

@Different row block sizes were evaluated:

@| Row Block | 8 Workers Latency (ms) | Speedup |
@|----------|----------------------|--------|
@| 32       | 47.63               | 1.52×  |
@| 64       | 49.40               | 1.47×  |
@| 128      | 66.74               | 1.08×  |
@| 256      | 99.82               | 0.73×  |


@---

@## PyTorch / ONNX Model Ingestion

@This project implements a minimal end-to-end model ingestion pipeline:

@```
@PyTorch model
@→ ONNX export
@→ Custom .mlir text IR
@→ C++ Graph IR
@→ Optimization passes
@→ Execution runtime
@```

@A tiny MLP exported from PyTorch was converted into the internal IR and executed through the C++ runtime.

@Runtime correctness was validated against PyTorch reference outputs:

@```
@PyTorch output:
@0 0 0 0.6

@Runtime output:
@0 0 0 0.6
@```

@This demonstrates end-to-end model import, graph lowering, optimization, and runtime execution correctness.

@---

@## Transformer Attention Operator

@The runtime was extended with a Transformer-style scaled dot-product attention operator:

@```
@Attention(Q, K, V) = softmax(QKᵀ / √D)V
@```

@Implemented components include:

@- QKᵀ attention score computation
@- Numerically stable row-wise Softmax
@- Attention-weighted value aggregation

@Attention correctness was validated against a NumPy reference implementation:

@```
@Python reference:
@6.224593 7.550813
@3.775407 12.449187

@C++ runtime:
@6.22459 7.55081
@3.77541 12.4492
@```

@This extends the runtime to support modern Transformer-style workloads and demonstrates understanding of contemporary ML operator execution.

@- Causal attention for autoregressive decoding
@- KV cache simulation for incremental Transformer inference
@- Tensor lifetime analysis and memory reuse


@### Conclusion

@Efficient parallel execution requires balancing:

@- Task granularity (load balancing)
@- Scheduling overhead (queue, locks)
@- Hardware parallelism

@The best configuration was row block size 32 with 8 workers, achieving 47.63 ms latency and 1.52× speedup over the single-thread tiled baseline.

@### Key Insight
@- Small row blocks improve load balancing across workers
@- Large row blocks reduce scheduling overhead but underutilize parallelism
@- Fine-grained scheduling increases queue and synchronization overhead

@Although tiling is a critical optimization on real GPUs, the CPU-based simulation shows degraded performance.

@Reasons:

@- Each simulated thread performs dynamic memory allocation (`std::vector`)
@- No true shared memory or warp-level execution exists
@- All threads are executed sequentially on CPU
@- Memory loading overhead dominates computation

@### Conclusion

@### Engineering Takeaways

@- GPU execution models cannot be directly mapped to CPU without considering hardware differences
@- Shared memory and parallel execution are essential for GPU performance
@- CPU simulation introduces overheads that do not reflect real GPU behavior

@This reinforces the importance of hardware-aware optimization in ML runtime design.

---



@##  Memory Access Optimization Insights

@During optimization, multiple tiled loop orderings were evaluated.

@Two key variants were tested:

@- **i-k-j ordering (optimized)**
@- **i-j-k ordering (degraded performance)**

@### i-k-j (selected)

@```
@for i
@  for k
@    load A[i, k]
@    for j
@      C[i, j] += A[i, k] * B[k, j]
@```

@Advantages:
@- Sequential access on B (row-major)
@- Sequential writes on C
@- High reuse of A[i,k]
@- Cache-friendly and SIMD-friendly

@### i-j-k (rejected)

@```

@for i
@  for j
@    for k
@      C[i, j] += A[i, k] * B[k, j]
@```


@Problems:
@- B is accessed with large strides (poor locality)
@- Cache miss rate increases significantly
@- Prefetching and vectorization are less effective

@### Result

@```

@i-k-j tiled: 8.68 ms
@i-j-k tiled: 13.96 ms
@```


@### Conclusion

@This experiment highlights the gap between conceptual execution models and hardware-aware implementations:

@- GPU execution models rely on true parallelism and fast on-chip shared memory
@- CPU simulation executes threads sequentially and incurs allocation overhead
@- As a result, GPU-style tiling cannot be directly translated to CPU performance

@This reinforces that performance optimization must be co-designed with hardware architecture.
@This experiment was designed to bridge the gap between high-level execution abstractions and hardware-specific optimizations in ML runtime systems.
@Compared to the optimized CPU tiled kernel (8.68 ms), the GPU-style simulation highlights that performance gains depend on real hardware support for parallel execution and memory hierarchy, not just execution abstraction.
---

@##  Key Techniques

@### 1. Operator Fusion
@Combines multiple operators into a single kernel to reduce dispatch overhead and improve cache locality.

@### 2. Tiled Matrix Multiplication
@Improves cache efficiency by blocking memory access patterns.

@### 3. DAG Scheduling
@Ensures correct execution order using topological sorting based on tensor dependencies.

@### 4. Operator Registry
@Decouples execution logic from kernel implementations, enabling modular runtime design.

@### 5. Shape Inference
@Propagates tensor shapes through the graph for validation and memory planning.

@### 6. Memory Planning
@Uses an arena-style allocation strategy to reduce dynamic memory overhead.

@### 7. Tensor Lifetime Analysis and Memory Reuse
@Tracks tensor first/last usage and reuses arena memory for tensors with non-overlapping lifetimes.

@### 8. Causal Attention
@Implements autoregressive masking so tokens only attend to current and previous positions.

@### 9. KV Cache Simulation
@Simulates decoder-style incremental inference by caching previous K/V tensors and reusing them across decode steps.
---

@##  Motivation

@Modern ML systems on edge devices require:

@- Efficient execution under resource constraints
@- Compiler-style optimizations for performance
@- Modular runtime systems for flexibility
@- Fine-grained control over memory and execution

@This project explores these challenges by implementing a simplified but realistic ML compiler and runtime system in C++.

---

@##  Future Work

@- SIMD / vectorized kernels
@- Memory reuse and lifetime analysis
@- Quantization (INT8 / FP16)
@- Multi-head attention
@- KV cache optimization
@- Flash-attention style tiling
@- SIMD/vectorized attention kernels
@- Profiling and latency breakdown tools
@- Python API for model loading and execution
@- Multi-head attention
@- FlashAttention-style kernel optimization
@- SIMD/vectorized Transformer kernels
@- Dynamic shape execution
@- ONNX graph-level optimization passes
@- Real GPU backend integration (CUDA/Metal)

---
