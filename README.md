## Compiler Runtime Infrastructure

Implemented a compiler-runtime infrastructure for heterogeneous inference execution, graph-level optimization, backend-aware scheduling, graph lowering, runtime tracing, and serving-oriented Transformer inference simulation.

The system simulates lightweight ML compiler/runtime architectures inspired by TensorRT, XLA, TVM, MLIR-based runtimes, vLLM, TensorRT-LLM, and modern inference-serving systems.

### Compiler Runtime Architecture

```text
Graph IR
    →
CanonicalizationPass
    →
ShapeInferencePass
    →
DTypePropagationPass
    →
FusionCandidatePass
    →
MemoryPlanningPass
    →
BackendPlacementPass
    →
SchedulingPass
    →
LoweredGraph IR
    →
ExecutionPlan IR
    →
StaticExecutionSchedule
    →
ScheduleExecutor
    →
Runtime Tracing
    →
Compiler Cost Analysis
    →
Serving Runtime
```

### Implemented Compiler Passes

#### CanonicalizationPass

Implemented graph canonicalization analysis for common inference simplification opportunities.

Implemented canonicalization candidates including:

- ReLU(ReLU(x)) → ReLU(x)
- Add(x, 0) → x
- MatMul(x, Identity) → x

This simulates compiler-side canonicalization infrastructure used in graph optimization systems.

#### ShapeInferencePass

Implemented graph-level tensor shape propagation across operators.

#### DTypePropagationPass

Implemented graph-level dtype propagation infrastructure.

#### FusionCandidatePass

Implemented fusion-candidate analysis and lightweight graph rewrite infrastructure for common inference subgraphs.

Implemented fusion rewrites:

- MatMul + Add → FusedMatMulBias
- MatMul + Add + ReLU → FusedMatMulAddReLU

Implemented fusion-aware execution schedule generation.

This simulates TensorRT-style compiler-side fusion analysis and backend-oriented graph rewrites.

#### MemoryPlanningPass

Implemented tensor lifetime analysis and memory reuse planning.

Implemented:

- activation lifetime tracking
- persistent tensor analysis
- buffer reuse planning
- memory-offset assignment
- peak memory estimation
- activation reuse analysis

Example reuse behavior:

```text
output reuses buffer from matmul_out at offset 16
```

This simulates memory-planning systems used in production compiler-runtime infrastructures.

#### BackendPlacementPass

Implemented backend-aware operator placement analysis for heterogeneous execution.

Implemented backend dispatch candidates including:

- CPU execution
- MockGPU execution
- Metal backend execution

This simulates heterogeneous backend placement systems used in inference runtimes.

#### SchedulingPass

Implemented dependency-aware static execution scheduling.

Implemented:

- topological execution ordering
- backend-aware scheduling
- tensor dependency tracking
- memory-offset propagation
- compiled execution schedule generation

Example execution schedule:

```text
[0] matmul | FusedMatMulAddReLU | backend=Metal | mem_offset=16
```

### LoweredGraph IR

Implemented backend-aware graph lowering from optimized Graph IR into LoweredOp IR.

Implemented lowering paths including:

- MatMul → LoweredMatMul
- Add → LoweredElementwiseAdd
- ReLU → LoweredElementwiseReLU
- FusedMatMulBias → LoweredFusedMatMulBias
- FusedMatMulAddReLU → LoweredFusedMatMulAddReLU
- Attention → LoweredAttention

Implemented lowering metadata propagation including:

- fused-op metadata
- backend placement
- tensor dependency metadata
- memory offsets
- lowered operator typing

Example lowered IR:

```text
[0] matmul
→ LoweredFusedMatMulAddReLU
backend=Metal
mem_offset=16
```

Generated artifacts:

- lowered_graph.json
- lowered_graph.png

This simulates backend-aware graph lowering infrastructure used in production inference compilers and runtime systems.

### ExecutionPlan IR

Implemented ExecutionPlan IR generation from LoweredGraph IR.

Implemented execution-step metadata including:

- step_id
- lowered operator metadata
- backend placement
- dependency steps
- memory offsets
- launch configuration
- tensor dependency propagation

Implemented launch configuration generation including:

```text
threadgroup=256
```

Example execution plan step:

```text
[step 0]
LoweredFusedMatMulAddReLU
backend=Metal
deps=[]
launch=threadgroup=256
```

Generated artifacts:

- execution_plan_v2.json
- execution_plan_v2.png

This simulates executable-plan generation infrastructure used in compiler-runtime systems such as TensorRT, XLA, and MLIR-based runtimes.

### Static Execution Schedule

Implemented execution-schedule export infrastructure including:

- execution order
- backend placement metadata
- tensor dependency metadata
- memory-offset metadata
- schedule visualization tooling

Generated artifacts:

- static_schedule.json
- fusion_bias_schedule.json
- static_schedule_table.png

### ScheduleExecutor

Implemented a compiled-schedule executor that consumes static execution schedules and execution-plan metadata for backend-aware runtime dispatch.

Implemented:

- schedule-driven execution
- backend-aware runtime dispatch
- runtime trace generation
- runtime latency tracking
- execution observability tooling

### Runtime Tracing

Implemented runtime tracing infrastructure including:

- operator-level latency tracing
- backend execution tracing
- runtime event export
- runtime visualization tooling
- schedule-aware runtime tracing

Generated artifacts:

- runtime_trace.json
- scheduled_runtime_trace.json
- runtime_execution_trace.png

### Compiler Cost Analysis

Implemented compiler-side cost analysis with runtime-feedback integration.

The compiler cost system estimates:

- estimated memory read bytes
- estimated memory write bytes
- estimated FLOPs
- arithmetic intensity
- kernel launch overhead
- backend-switch overhead
- fusion-aware execution cost

The runtime system additionally merges measured backend execution latency into compiler-side cost reports.

Example runtime-aware cost report:

```text
matmul | FusedMatMulAddReLU
backend=Metal
read_bytes=48
write_bytes=16
flops=24
actual_latency_ms=31.1692
```

Generated artifacts:

- compiler_cost_report.json
- compiler_cost_report.png

This simulates lightweight cost-model analysis and runtime-feedback integration used in modern ML compiler/runtime systems.

## LLM Serving Runtime

Implemented a mini vLLM-style serving runtime layer on top of the compiler-runtime infrastructure.

The serving runtime simulates modern Transformer inference-serving systems such as vLLM, TensorRT-LLM, and SGLang.

Implemented:

- prefill/decode execution separation
- continuous batching
- paged KV-cache allocation
- KV block reuse
- request lifecycle management
- serving-side throughput profiling
- token-generation tracking
- serving-oriented runtime orchestration

### Serving Runtime Architecture

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
Subgraph Delegation
    →
CPU / MockGPU / Metal Backend
    →
ServingProfiler
```

## Metal Runtime Profiling

Implemented real Metal compute-kernel profiling on Apple Silicon using repeated GPU dispatch benchmarking and latency-distribution analysis.

The profiling pipeline measures real Metal compute-kernel execution instead of command-buffer-only overhead.

Implemented profiling features including:

- real Metal compute-kernel execution
- repeated GPU dispatch benchmarking
- warmup vs steady-state profiling
- p50 / p95 / p99 latency analysis
- runtime latency-distribution export
- Apple Silicon backend profiling
- GPU execution correctness validation

Profiled Metal kernels including:

- vector_add
- Metal backend dispatch infrastructure

Example profiling metrics:

```text
avg_ms: 0.426669
p50_ms: 0.404792
p95_ms: 0.489541
p99_ms: 0.566416
```

Generated profiling artifacts:

- metal_vector_add_profile.json
- metal_vector_add_profile.png

This extends the runtime system from compiler-side static cost estimation into measured GPU runtime profiling with latency-distribution analysis on Apple Silicon Metal backends.