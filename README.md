## CV Graph Compiler and Runtime Infrastructure

Implemented a compiler-runtime simulation pipeline for CV-oriented heterogeneous inference execution, backend-aware scheduling, graph lowering, memory planning, adaptive runtime orchestration, and runtime-feedback-driven execution replanning.

Implemented compiler-runtime infrastructure inspired by:

- TensorRT
- XLA
- TVM
- MLIR-based runtimes
- TensorRT-LLM
- vLLM
- heterogeneous inference-serving systems

### CV Graph Pipeline

Implemented a CV inference graph including:

```text
Conv2D
    →
BatchNorm
    →
ReLU
    →
MaxPool
    →
Flatten
    →
Linear
```

Implemented graph-level compiler optimization passes including:

- ShapeInferencePass
- CanonicalizationPass
- DTypePropagationPass
- FusionCandidatePass
- MemoryPlanningPass
- BackendPlacementPass
- SchedulingPass

Implemented backend-aware graph lowering into:

- LoweredGraph IR
- ExecutionPlan IR
- StaticExecutionSchedule

Generated artifacts:

- [cv_lowered_graph.json](trace/cv_lowered_graph.json)
- [cv_execution_plan_v2.json](trace/cv_execution_plan_v2.json)
- [cv_static_schedule.json](trace/cv_static_schedule.json)

### CV Graph Fusion Analysis

Implemented compiler-side fusion analysis and graph rewrite infrastructure for CV inference optimization.

Implemented fusion rewrite:

```text
Conv2D + BatchNorm + ReLU
    →
FusedConvBatchNormReLU
```

Implemented:

- fusion-candidate analysis
- graph rewrite infrastructure
- fused-op lowering
- fused execution scheduling
- fusion-aware runtime planning

Example compiler rewrite:

```text
rewriting:
Conv2D + BatchNorm + ReLU
    →
FusedConvBatchNormReLU
```

This simulates lightweight fusion infrastructure used in production ML compilers and inference runtimes.

### Tensor Lifetime and Memory Planning

Implemented compiler-side tensor lifetime analysis and runtime memory-reuse planning.

Implemented:

- activation lifetime tracking
- persistent tensor analysis
- buffer reuse planning
- memory-offset assignment
- runtime memory reuse
- peak-memory estimation
- activation reuse analysis

Example memory reuse behavior:

```text
relu_out reuses buffer from conv_out
pool_out reuses buffer from conv_out
flat_out reuses buffer from conv_out
logits reuses buffer from conv_out
```

Example memory-planning result:

```text
Naive memory:
4882250 float elements

Planned peak memory:
3699424 float elements

Saved memory:
1182826 float elements
```

Generated artifacts:

- [cv_memory_plan.json](trace/cv_memory_plan.json)

This simulates lightweight runtime-memory planning infrastructure used in heterogeneous inference runtimes and serving systems.

### Backend Placement and Heterogeneous Scheduling

Implemented heterogeneous backend-placement analysis and runtime execution scheduling.

Implemented backend placement including:

- CPU execution
- Metal execution
- MockGPU execution

Implemented:

- backend-aware lowering
- dependency-aware scheduling
- backend transition analysis
- execution-plan generation
- execution timeline simulation
- heterogeneous runtime orchestration

Example execution schedule:

```text
[0] conv1 | FusedConvBatchNormReLU | backend=Metal
[1] pool1 | MaxPool | backend=CPU
[2] flatten | Flatten | backend=CPU
[3] linear | Linear | backend=Metal
```

Generated artifacts:

- [cv_static_schedule.json](trace/cv_static_schedule.json)
- [cv_runtime_timeline.json](trace/cv_runtime_timeline.json)

### Subgraph Partitioning

Implemented backend-aware subgraph partitioning for heterogeneous inference execution.

Implemented:

- backend-oriented graph partitioning
- execution-region grouping
- backend execution segmentation
- runtime migration-region analysis

Example partitioning result:

```text
subgraph 0 | backend=Metal | ops=conv1
subgraph 1 | backend=CPU | ops=pool1 flatten
subgraph 2 | backend=Metal | ops=linear
```

Generated artifacts:

- [cv_subgraph_partition.json](trace/cv_subgraph_partition.json)

This simulates heterogeneous execution partitioning used in production inference runtimes and compiler-runtime systems.

### Compiler Cost Analysis

Implemented compiler-side cost-analysis infrastructure with runtime-aware scheduling metadata.

Implemented:

- estimated memory-read analysis
- estimated memory-write analysis
- FLOPs estimation
- arithmetic-intensity analysis
- backend-switch overhead estimation
- launch-cost estimation
- fusion-aware execution analysis

Example runtime-aware cost report:

```text
conv1 | FusedConvBatchNormReLU
backend=Metal
read_bytes=603840
write_bytes=3154176
flops=42581376
intensity=11.3308
```

Generated artifacts:

- [cv_cost_report.json](trace/cv_cost_report.json)

This simulates lightweight compiler-side cost modeling and runtime execution analysis used in ML compiler/runtime systems.

## Adaptive Runtime Planning and Orchestration

Implemented adaptive runtime-planning infrastructure for heterogeneous execution scheduling, runtime feedback analysis, backend migration, and dynamic runtime recovery orchestration.

Implemented:

- runtime feedback-driven backend replanning
- heterogeneous execution-plan comparison
- runtime latency-aware backend migration
- runtime overload detection
- adaptive CPU fallback orchestration
- GPU recovery-state management
- runtime state-machine simulation
- runtime orchestration visualization tooling

### Timeline Optimization Simulation

Implemented runtime what-if execution-plan analysis for heterogeneous backend scheduling.

Implemented execution-plan comparisons including:

- current heterogeneous execution plan
- all-Metal execution plan
- Metal-pool optimized execution
- CPU-middle fallback execution

Compared runtime-planning metrics including:

- total execution latency
- backend-switch overhead
- memory pressure estimation
- GPU occupancy proxy
- runtime orchestration efficiency

Example runtime-planning analysis:

```text
Current:
Metal conv
↓ switch
CPU pool
CPU flatten
↓ switch
Metal linear

All-Metal:
Metal conv
Metal pool
Metal flatten
Metal linear
```

Generated artifacts:

![Timeline Optimization](cv_timeline_optimization.png)

This simulates lightweight runtime-planning analysis and heterogeneous execution optimization used in production ML runtimes.

### Cost-Based Backend Planner

Implemented a lightweight cost-based backend planner for heterogeneous runtime execution optimization.

Implemented:

- candidate backend-plan evaluation
- latency-aware plan selection
- backend-switch cost estimation
- GPU occupancy-aware scheduling heuristics
- runtime memory-pressure estimation
- execution-plan ranking
- best-plan selection infrastructure

Implemented runtime-planning candidates including:

- current heterogeneous plan
- all-Metal plan
- Metal-pool-only plan

Example planner output:

```text
current:
latency=1.49 ms
switch_cost=0.04 ms
gpu_occupancy=0.36

all_metal:
latency=0.76 ms
switch_cost=0.00 ms
gpu_occupancy=1.00
BEST
```

Generated artifacts:

![Cost-Based Planner](cv_cost_based_planner.png)

This simulates lightweight cost-based runtime scheduling infrastructure used in modern inference runtimes and compiler-runtime systems.

### Runtime Adaptive Replanning

Implemented runtime-feedback-driven adaptive replanning simulation for heterogeneous inference execution.

Implemented:

- runtime latency monitoring
- backend overload detection
- runtime backend migration
- adaptive CPU fallback orchestration
- runtime-plan replacement
- runtime execution recovery modeling

Example runtime replanning scenario:

```text
Initial Plan:
all_metal
latency=0.76 ms

Runtime Feedback Trigger:
Metal observed 2.84 ms overload

Replanned:
runtime_replanned_cpu_fallback
latency=2.10 ms
```

Generated artifacts:

![Runtime Adaptive Replanning](cv_runtime_replan.png)

This simulates runtime-feedback orchestration and adaptive heterogeneous backend migration systems used in serving runtimes and edge inference systems.

### Adaptive Runtime State Machine

Implemented adaptive runtime state-machine simulation for dynamic backend orchestration and runtime recovery pipelines.

Implemented runtime states including:

- NORMAL
- OVERLOAD_DETECTED
- REPLANNING
- CPU_FALLBACK
- RECOVERY_CHECK
- RESTORE_GPU_PLAN

Implemented runtime transitions including:

- Metal latency-spike detection
- planner invocation
- backend migration
- GPU health probing
- latency normalization recovery

Example runtime orchestration flow:

```text
NORMAL
    →
OVERLOAD_DETECTED
    →
REPLANNING
    →
CPU_FALLBACK
    →
RECOVERY_CHECK
    →
RESTORE_GPU_PLAN
```

Generated artifacts:

![Runtime State Machine](cv_runtime_state_machine.png)

This simulates adaptive runtime orchestration systems used in heterogeneous inference runtimes, edge inference systems, and serving-oriented runtime infrastructures.

## LLM Serving Runtime Infrastructure

Implemented a lightweight serving-oriented Transformer runtime inspired by:

- vLLM
- TensorRT-LLM
- SGLang
- serving-oriented inference runtimes

Implemented:

- prefill/decode execution separation
- continuous batching simulation
- serving-oriented request scheduling
- token-generation orchestration
- runtime serving profiler
- serving-side execution tracing

### KV Cache Infrastructure

Implemented paged KV-cache infrastructure for Transformer serving runtime simulation.

Implemented:

- KV block allocation
- paged KV-cache simulation
- KV block reuse
- token append simulation
- serving-oriented cache lifecycle management
- runtime cache memory tracking

Generated artifacts:

- [kv_cache_trace.json](trace/kv_cache_trace.json)
- [paged_kv_runtime.json](trace/paged_kv_runtime.json)

This simulates lightweight KV-cache memory-management infrastructure used in modern LLM serving runtimes.

### Transformer Attention Runtime

Implemented serving-oriented attention runtime infrastructure.

Implemented:

- fused attention simulation
- tiled attention execution
- causal attention execution
- paged-attention scheduling simulation
- attention runtime orchestration
- backend-aware attention execution

Implemented demos including:

- run_attention_demo
- run_fused_attention_demo
- run_tiled_attention_demo
- run_causal_attention_demo

This simulates lightweight Transformer runtime execution infrastructure used in modern LLM inference systems.

### Serving Runtime Scheduling

Implemented serving-oriented runtime scheduling and orchestration infrastructure.

Implemented:

- request lifecycle management
- serving-side execution scheduling
- runtime batching simulation
- serving-oriented execution orchestration
- runtime timeline analysis
- serving execution profiling

Implemented runtime orchestration including:

```text
LLM Requests
    →
LLMScheduler
    →
ContinuousBatcher
    →
PrefillDecodeExecutor
    →
PagedKVCache
    →
ExecutionPlan
    →
CPU / Metal Backend
    →
ServingProfiler
```

This simulates serving-runtime orchestration used in modern LLM serving systems and inference-serving runtimes.

### MLIR Compiler Pass Pipeline

This project now includes a real MLIR C++ pass plugin under `mlir_passes/`.
The pass detects a tensor-level MatMul + Bias Add + ReLU pattern:

```text
linalg.matmul
  -> linalg.map arith.addf
  -> linalg.map arith.maximumf
```

The pass annotates fusion candidates and assigns fusion metadata:

```mlir
linalg.matmul {
  fusion.candidate = "matmul_bias_relu",
  fusion.group = "matmul_bias_relu_0",
  fusion.role = "producer"
}
```

The MLIR pipeline is connected to runtime-facing artifacts:

```text
trace/mlir_fused_graph.mlir
trace/mlir_lowered_graph.json
trace/mlir_execution_plan.json
```

Run the pipeline and tests:

```bash
cmake --build build-mlir
tools/run_mlir_pass_tests.sh
tools/run_mlir_fusion_pipeline.sh
```

This adds a real MLIR frontend pass stage before the existing custom
LoweredGraph / ExecutionPlan / heterogeneous runtime planning flow.