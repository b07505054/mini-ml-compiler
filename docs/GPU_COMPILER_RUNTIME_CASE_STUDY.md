# GPU Compiler/Runtime Case Study

## Problem

RMSNorm appears in LLM prefill and decode paths. It is simple enough to explain
in an interview, but still representative of real inference-system work:
correctness, launch overhead, memory bandwidth, kernel selection, and fallback
policy all matter.

## Runtime Evidence

The runtime evidence is produced by `heterogeneous-inference-runtime`, not by the
compiler repo:

```text
results/cuda_transformer/rmsnorm_benchmark.json
results/cuda_transformer/rmsnorm_benchmark_report.md
results/cuda_transformer/gpu_pgo_like_rmsnorm_report.json
results/cuda_transformer/gpu_pgo_like_rmsnorm_report.md
```

The benchmark compares:

```text
custom kernel: fused_rmsnorm_cuda
fallback: torch_rmsnorm
dtype: float32
tokens: 1, 16, 128
hidden: 768, 1024, 4096, 8192
```

The correctness test checks the same fixed sweep with explicit tolerances,
finite outputs, and a documented contiguous-input precondition. CUDA-unavailable
machines skip the CUDA test instead of failing CPU-only validation.

The representative measured profile currently records:

```text
shape bucket: 16x4096:f32
custom latency: 0.030196 ms
PyTorch latency: 0.088261 ms
speedup: 2.923x
custom effective bandwidth: 34.726 GB/s
PyTorch effective bandwidth: 11.88 GB/s
arithmetic intensity: 0.25 FLOPs/byte
```

The benchmark report also records environment metadata: GPU name, CUDA version,
NVCC version, PyTorch version, NVIDIA driver version, warmup runs, timed runs,
dtype, and commit hash. Nsight Compute is optional; if `ncu` is missing, the
report records that explicitly instead of blocking the benchmark.

The GPU PGO-like report combines CUDA/Triton/PyTorch candidates by shape bucket
and records the three required portfolio questions:

```text
input: compiler-emitted HIR RMSNorm op plus runtime shape/workload distribution
decision: profile-guided selection among CUDA/Triton/PyTorch candidates
metric: kernel p95 latency, bandwidth, TPOT projection, throughput projection
```

## Compiler Decision

The compiler consumes the runtime artifact and makes a kernel-selection
decision. It does not fabricate benchmark numbers.

```text
llm.rmsnorm
  -> RMSNormKernelSelectionPass
  -> hir.fused_rmsnorm
  -> GPU PGO-like profile feedback table
  -> fused_rmsnorm_cuda or torch_rmsnorm fallback
```

The compiler-side case study artifacts are:

```text
trace/rmsnorm_compiler_runtime_case_study.json
trace/rmsnorm_compiler_runtime_case_study.md
```

They verify:

```text
input contains llm.rmsnorm
lowered graph contains hir.fused_rmsnorm
selected kernel is fused_rmsnorm_cuda
fallback kernel is torch_rmsnorm
runtime profile status is measured
correctness passed
decision is profile calibrated
GPU PGO-like gate passes input/decision/metric
```

## Why It Is Fast

RMSNorm is memory-bound in this implementation. Each token streams input for the
sum-of-squares reduction, reads input again for output, reads the weight vector,
and writes the output. The low arithmetic intensity means the interesting
performance questions are bandwidth, reduction efficiency, launch overhead, and
framework dispatch overhead.

The portfolio point is not "custom CUDA always wins." The point is:

```text
runtime repo produces evidence
compiler repo consumes evidence
compiler emits a decision
fallback remains available when evidence is missing, slower, or incorrect
```

## Serving Extension

Decode attention is the next memory bottleneck. The compiler repo keeps a
separate KV-cache bandwidth model:

```text
trace/attention_kv_bandwidth_model.json
trace/attention_kv_bandwidth_model.md
```

This is intentionally a model, not a claimed FlashAttention kernel. It shows how
KV bytes/token grow with context length and why paged KV layout, block size,
memory pressure, and scheduler admission policy must be coordinated between the
compiler and runtime.

## Interview Framing

This case study answers four concrete questions:

```text
PR1: Is the CUDA kernel correct, measured, and reproducible?
PR2: Is compiler kernel selection driven by runtime evidence?
PR3: Can the performance result be explained with a bandwidth/roofline model?
PR4: Does the system connect this to real LLM serving bottlenecks like KV cache?
```
