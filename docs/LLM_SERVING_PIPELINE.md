# LLM Serving Compiler/Runtime Planning Pipeline

This project contains a small LLM inference serving compiler/runtime planning
pipeline. It is not a vLLM clone and it is not a full production compiler.
Instead, it demonstrates how a compiler/runtime layer can analyze a tiny LLM
graph and emit serving-runtime artifacts for validation and dashboard demos.

## Pipeline

```text
mlir/tiny_gpt_serving.mlir
  -> tools/analyze_llm_serving_mlir.py
  -> trace/llm_serving_compiler_analysis.json
  -> tools/emit_llm_artifacts_from_analysis.py
  -> artifacts/apple_demo/*.json
  -> tools/validate_llm_serving_artifacts.py
  -> trace/llm_artifact_validation_report.json
  -> integration_bundle/apple_demo_artifacts/*.json
```

## One-Command Run

```bash
tools/run_llm_serving_artifact_pipeline.sh
```

Optional overrides:

```bash
MLIR_INPUT=mlir/tiny_gpt_serving.mlir \
CONFIG_INPUT=configs/tiny_gpt_llm_config.json \
ARTIFACT_OUT=artifacts/apple_demo \
BUNDLE_OUT=integration_bundle/apple_demo_artifacts \
tools/run_llm_serving_artifact_pipeline.sh
```

## Input

The input MLIR-style graph is:

```text
mlir/tiny_gpt_serving.mlir
```

It models a tiny GPT-style prefill/decode graph with:

```text
llm.embed
llm.rmsnorm
llm.qkv_projection
llm.attention_prefill
llm.attention_decode
llm.mlp
```

The attention ops carry serving metadata:

```text
kv_cache.role = "producer"
serving.phase = "prefill"

kv_cache.role = "consumer"
serving.phase = "decode"
```

## Compiler Analysis

The analysis pass emits:

```text
trace/llm_serving_compiler_analysis.json
```

It extracts:

```text
model metadata
operator list
prefill/decode partition
KV-cache producer/consumer roles
runtime constraints
```

Example constraints:

```json
{
  "prefill_is_compute_heavy": true,
  "decode_is_token_step": true,
  "decode_requires_kv_cache_read": true,
  "continuous_batching_supported": true
}
```

## Runtime Planning Artifacts

The artifact lowering step emits:

```text
artifacts/apple_demo/llm_graph_ir.json
artifacts/apple_demo/serving_execution_plan.json
artifacts/apple_demo/kv_cache_plan.json
artifacts/apple_demo/memory_plan.json
artifacts/apple_demo/scheduling_plan.json
artifacts/apple_demo/validation_manifest.json
```

These files are the contract between the compiler/runtime planner and the Apple
demo dashboard.

## Artifact Responsibilities

`llm_graph_ir.json`

Describes the tiny LLM graph:

```text
model name
number of layers
hidden size
number of heads
operators
request workload
```

`serving_execution_plan.json`

Describes how the serving runtime should treat prefill and decode:

```text
prefill phase
decode phase
backend preference
runtime contract
```

`kv_cache_plan.json`

Describes KV-cache layout and capacity:

```text
block size
number of blocks
token capacity
KV dtype
bytes per token
bytes per block
allocation strategy
paged attention flags
```

`memory_plan.json`

Describes estimated memory pressure:

```text
prefill peak memory
decode peak memory
KV-cache memory at target concurrency
temporary buffer memory
memory budget status
```

`scheduling_plan.json`

Describes serving scheduler behavior:

```text
continuous batching
prefill queue
decode queue
decode step size
dashboard signals
```

`validation_manifest.json`

Lists expected artifacts and validation checks.

## Validation

Validation emits:

```text
trace/llm_artifact_validation_report.json
```

It checks:

```text
required artifact files exist
prefill/decode phases are present
KV-cache capacity is internally consistent
memory budget is not exceeded
prefill/decode scheduling queues exist
manifest expected outputs are complete
```

A successful report contains:

```json
{
  "summary": {
    "status": "passed"
  }
}
```

## Apple Demo Bundle

The final integration bundle is:

```text
integration_bundle/apple_demo_artifacts/
```

It should contain:

```text
llm_graph_ir.json
serving_execution_plan.json
kv_cache_plan.json
memory_plan.json
scheduling_plan.json
validation_manifest.json
llm_artifact_validation_report.json
```

The Apple demo should read these JSON files directly rather than hardcoding
model or runtime values.

## Positioning

This project should be described as:

```text
An LLM inference serving compiler/runtime planning demo.
```

It is responsible for:

```text
MLIR-style LLM graph input
serving-aware compiler analysis
prefill/decode execution planning
KV-cache layout planning
memory planning
scheduling metadata
artifact validation
Apple demo bundle generation
```

It is not responsible for:

```text
full model inference
tokenization
sampling
OpenAI-compatible chat API
production vLLM serving
distributed inference
```