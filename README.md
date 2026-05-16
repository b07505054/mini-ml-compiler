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