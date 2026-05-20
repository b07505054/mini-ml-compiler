## LLM Serving Runtime

Implemented a mini vLLM-style serving runtime layer on top of the compiler/runtime system.

The serving stack simulates modern Transformer inference serving systems such as vLLM, TensorRT-LLM, SGLang, and production on-device inference runtimes.

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
Execution Plan
    →
Subgraph Delegation
    →
CPU / MockGPU Backend
    →
ServingProfiler
```

### Implemented Serving Components

#### LLMRequest

Implemented request-level serving metadata including:

- request lifecycle state
- prompt tokens
- generated tokens
- KV cache block table
- latency timestamps
- token generation statistics

Request lifecycle:

```text
Waiting
→ Prefill
→ Decode
→ Finished
```

#### LLMScheduler

Implemented a serving-oriented request scheduler supporting:

- prefill/decode phase separation
- request lifecycle orchestration
- decode queue management
- finished-request cleanup
- KV-cache-aware request handling

#### ContinuousBatcher

Implemented a continuous batching system that dynamically constructs decode batches from active requests.

Example batching:

```text
[ContinuousBatcher] Building decode batch
  request 1
  request 2

[ContinuousBatcher] Batch size: 2
```

This simulates continuous batching systems used in modern LLM inference runtimes.

#### PrefillDecodeExecutor

Implemented separated prefill/decode execution paths.

Execution flow:

```text
prefill
→ KV allocation
→ decode execution
→ token generation
→ request completion
```

Example execution:

```text
[Executor] PREFILL request 1

[Executor] DECODE request 1 generated token 100
```

#### Paged KV Cache

Implemented a paged KV-cache manager inspired by vLLM-style memory systems.

Features:

- block-based KV allocation
- request-level block tables
- KV block reuse
- request-aware memory freeing
- serving-side memory tracking

Example reuse behavior:

```text
Request 1 blocks: 0 1

free request 1

Request 3 blocks: 0 1
```

This simulates paged attention memory management used in production Transformer serving runtimes.

#### ServingProfiler

Implemented serving-side runtime profiling including:

- request latency tracking
- generated token counting
- tokens/sec analysis
- average request latency
- serving throughput statistics

Example metrics:

```text
=== Serving Metrics ===

Request 1
  latency_ms: 0.181708
  generated_tokens: 3
  tokens/sec: 16510

Total generated tokens: 5
Average request latency: 0.155958 ms
```

### Serving Runtime Features

- prefill/decode execution separation
- continuous batching
- paged KV-cache allocation
- KV block reuse
- request lifecycle management
- serving-side throughput profiling
- request latency analysis
- token generation tracking
- serving-oriented runtime orchestration

This extends the compiler/runtime system from single-execution graph inference into a serving-oriented Transformer inference runtime.

## Compiler Pass Pipeline

Implemented a compiler-style optimization pipeline transforming Graph IR into backend-aware execution plans.

The compiler pipeline simulates lightweight ML compiler infrastructure inspired by systems such as XLA, TensorRT, TVM, and MLIR-based runtimes.

### Compiler Pipeline Architecture

```text
Graph IR
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
ExecutionPlan
    →
StaticExecutionSchedule
    →
ScheduleExecutor
    →
Runtime Execution
```

### Implemented Compiler Passes

#### ShapeInferencePass

Implemented graph-level shape propagation across operators.

Example:

```text
[ShapeInference] Running shape inference
```

#### DTypePropagationPass

Implemented dtype propagation infrastructure for graph tensors.

Example:

```text
[DTypePropagationPass] Propagating tensor dtypes: default float32
```

#### FusionCandidatePass

Implemented fusion-candidate analysis and graph rewrite infrastructure for common inference subgraphs.

Implemented fusion patterns:

- MatMul + Add → FusedMatMulBias
- MatMul + Add + ReLU → FusedMatMulAddReLU

Example fusion rewrite:

```text
[FusionCandidatePass] Searching fusion candidates
  rewriting: MatMul + Add + ReLU -> FusedMatMulAddReLU
```

Example fused execution schedule:

```text
[0] matmul | FusedMatMulAddReLU | backend=Metal | mem_offset=16
```

This simulates compiler-side fusion analysis and graph rewrite passes used in TensorRT-style inference compilers.

#### MemoryPlanningPass

Implemented tensor lifetime analysis and memory reuse planning.

Features:

- tensor lifetime tracking
- persistent tensor analysis
- activation reuse
- buffer offset assignment
- peak memory estimation

Example:

```text
[MemoryPlanner] Reuse events
  output reuses buffer from matmul_out at offset 16
```

This simulates memory planning systems used in production compiler-runtime infrastructures.

#### BackendPlacementPass

Implemented backend-aware operator placement analysis.

Example:

```text
matmul -> MockGPU/Metal candidate
```

This simulates heterogeneous execution planning across CPU/GPU runtimes.

#### SchedulingPass

Implemented dependency-aware static execution scheduling.

Example:

```text
[SchedulingPass] Building static topological execution schedule
  [0] matmul | FusedMatMulAddReLU
```

### Static Execution Schedule

Implemented compiled execution schedule generation including:

- operator execution order
- backend-aware dispatch metadata
- tensor dependency tracking
- memory-offset tracking
- execution schedule export

Example schedule artifact:

```text
[0] matmul | FusedMatMulAddReLU | backend=Metal | mem_offset=16
```

Implemented JSON schedule export:

```text
static_schedule.json
fusion_bias_schedule.json
```

Implemented schedule visualization tooling:

```text
static_schedule_table.png
```

### ScheduleExecutor

Implemented a compiled-schedule executor that consumes static execution schedules and performs backend-aware runtime dispatch.

Features:

- schedule-driven execution
- backend-aware dispatch
- runtime trace generation
- execution observability
- runtime latency tracking

Example runtime execution:

```text
[ScheduleExecutor] order=0 op=matmul backend=Metal mem_offset=16
[MetalBackend] Executing node: matmul
```

### Runtime Execution Trace

Implemented runtime execution trace infrastructure including:

- runtime event tracing
- operator latency tracking
- backend execution profiling
- runtime trace export
- runtime visualization tooling

Example runtime trace:

```text
matmul | Metal | latency=30.6868 ms
```

Implemented runtime trace export:

```text
runtime_trace.json
scheduled_runtime_trace.json
```

Implemented runtime visualization tooling:

```text
runtime_execution_trace.png
```

### Compiler Runtime Features

- compiler-style pass infrastructure
- graph-level analysis passes
- fusion candidate analysis
- compiler-side graph rewrites
- fused execution schedule generation
- tensor lifetime analysis
- memory reuse planning
- backend-aware placement analysis
- dependency-aware static scheduling
- compiled execution schedule generation
- schedule-driven runtime execution
- runtime execution tracing
- backend-aware heterogeneous dispatch
- execution observability tooling
- lowering to execution plans
- compiler-runtime orchestration

This extends the system from a graph execution runtime into a compiler-runtime infrastructure supporting analysis, optimization, fusion rewrites, scheduling, backend-aware dispatch, runtime tracing, and serving-oriented inference execution.

### Compiler Cost Report

Implemented compiler cost-report analysis for backend-aware scheduling decisions.

The cost report estimates:

- memory read volume
- memory write volume
- kernel launch overhead
- backend transfer cost
- fusion opportunities

Example cost report:

```text
matmul | FusedMatMulAddReLU | backend=Metal | mem_read=8 | mem_write=4 | launch_cost=0.08 | transfer_cost=0.02
fusion: MatMul+Add+ReLU fused
```

Implemented JSON cost report export:

```text
compiler_cost_report.json
```

Implemented cost report visualization tooling:

```text
compiler_cost_report.png
```

This simulates lightweight cost-model analysis used by ML compilers and inference runtimes to guide fusion, backend placement, scheduling, and memory movement decisions.

## Metal Runtime Profiling

Implemented real Metal compute-kernel profiling on Apple Silicon using repeated GPU dispatch benchmarking and latency-distribution analysis.

The profiling pipeline measures real Metal compute execution instead of command-buffer-only overhead.

### Metal Kernel Profiling Flow

```text
Metal Compute Kernel
    →
Repeated GPU Dispatch
    →
Warmup Phase
    →
Steady-State Measurement
    →
Latency Distribution Analysis
    →
JSON Trace Export
    →
Visualization
```

### Implemented Profiling Features

- real Metal compute-kernel execution
- repeated GPU dispatch benchmarking
- warmup vs steady-state measurement
- p50 / p95 / p99 latency analysis
- runtime latency-distribution export
- profiling visualization pipeline
- Apple Silicon backend profiling
- correctness validation for GPU execution

### Profiled Kernel

Implemented Metal vector-add compute-kernel profiling using:

```text
1,048,576 elements
100 measured runs
Apple M5 GPU backend
```

Example profiling metrics:

```text
avg_ms: 0.426669
p50_ms: 0.404792
p95_ms: 0.489541
p99_ms: 0.566416
```

### Profiling Outputs

Generated profiling artifacts including:

- metal_vector_add_profile.json
- metal_vector_add_profile.png

This extends the runtime system from compiler-side static cost estimation into measured GPU runtime profiling with latency-distribution analysis on Apple Silicon Metal backends.