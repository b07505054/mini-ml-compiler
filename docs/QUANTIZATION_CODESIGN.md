# Quantization Co-Design

## Current truth boundary

Quantization is an executable compiler/runtime path for the validated fused
`ReLU(Linear(input, weight, bias))` operator on the Raspberry Pi 5. It is no
longer annotation-only or planning-only for that scope. Full-model
quantization, graph-wide mixed precision, INT4, AWQ, GPTQ, NPU, and DMA paths
remain outside this claim.

The compiler owns:

- complete quantization candidate generation and target-capability filtering;
- an evidence-backed quantization/backend/thread decision;
- deterministic calibration and packed-weight artifact identities;
- Q/DQ and integer tensor materialization in HIR;
- lowering to the selected integer kernel or explicit external-runtime plan;
- the ordered ExecutionPlan and every selected artifact identity.

The runtime validates and executes that decision. It does not select another
backend or precision, repack weights, recalibrate, or requantize the model.

## Effective schemes

The custom candidate uses static symmetric INT8 activations and weights,
per-tensor scales, zero points of zero, an INT32 accumulator, and
`packed_b_transposed_nxk` weights. Its selected kernel is
`portable_fused_matmul_bias_relu_int8_symmetric_packed_b`, compiled for
`cortex_a76_dotprod` and requiring `asimd` and `asimddp`.

The validated ExecuTorch/XNNPACK INT8 candidates use PT2E per-tensor affine
activation quantization and per-channel symmetric axis-0 weight quantization.
Their 1-thread and 4-thread forms are distinct `ImplementationCandidate`
identities. The schemes are intentionally not described as identical.

AWQ and GPTQ remain separate algorithms. Existing declarations or serving
artifacts do not imply that either is integrated into this fused-operator
candidate path.

## Compiler materialization

The selected custom path materializes real integer dataflow:

```text
FP32 input
  -> hir.quantize
  -> hir.load_quantized_weight
  -> hir.qmatmul
  -> hir.dequantize
  -> hir.portable_cpu_int8_fused_matmul_bias_relu
  -> FP32 output
```

Scale, zero point, granularity, layout, calibration identity, and packed
artifact identity are explicit. Missing or inconsistent quantization stages,
artifact hashes, layout, kernel identity, or ISA requirements are rejected.

## Complete candidate search

`ImplementationCandidate` means a complete executable implementation, not a
precision label. Relevant dimensions are:

- backend, runtime, and delegate;
- kernel and code-generation target;
- precision and quantization scheme;
- packing/layout and fixed thread count;
- target architecture, ISA, and packaged runtime capabilities;
- runner, program, calibration, packed-weight, and measurement artifacts;
- correctness and measured latency evidence.

For the canonical workload, the legal set includes portable CPU FP32,
portable packed Cortex-A76 INT8 1T, ExecuTorch/XNNPACK FP32 1T,
ExecuTorch/XNNPACK INT8 1T, and ExecuTorch/XNNPACK INT8 4T. Selection minimizes
validated steady-state invocation latency subject to correctness, stability,
artifact, capability, and policy-budget constraints. It does not favor either
backend.

## ExecutionPlan contract

The custom ordered stages are:

```text
quantize_activation
load_packed_weight
execute_int8_kernel
return_fp32_output
```

Packed-weight loading and artifact/hash validation are one-time work. Input
binding, activation quantization, integer compute, and FP32 output readiness
occur per invocation. Runtime repack and runtime redecision counters remain
zero.

An ExecuTorch/XNNPACK selection instead emits:

```text
load_executorch_program
bind_input
execute_xnnpack_delegate
return_output
```

The plan fixes the runner, `.pte`, delegate, precision, quantization scheme,
thread count, workload manifest, delegation proof, and measurement evidence.
It does not emit custom quantize or packed-weight stages for this path.

## Validated Raspberry Pi result

Slices 3E and 3F aligned the already-loaded invocation boundaries: canonical
custom ExecutionPlan invocation versus already-loaded ExecuTorch
`Method::execute`. Slice 3G imported the five-session evidence into complete
candidate selection and revalidated routing with 10 warmups and 100 samples.

| Shape | Selected candidate | Pi integration median ms | Cosine | Relative L2 |
|---|---|---:|---:|---:|
| `37x41x29` | XNNPACK INT8 1T | 0.003556 | 0.99997465 | 0.0071350 |
| `64x64x64` | XNNPACK INT8 4T | 0.0087315 | 0.99997473 | 0.0071127 |
| `128x128x128` | XNNPACK INT8 4T | 0.0222685 | 0.99997171 | 0.0075227 |
| `256x256x256` | XNNPACK INT8 4T | 0.100509 | 0.99997195 | 0.0074905 |

All four pass cosine similarity `>= 0.99` and relative L2 `<= 0.05`.
Selection agreement with the constraint-matched measured oracle is 4/4 and
normalized regret is 0.0 for every canonical shape.

These are fused-operator measurements on one Raspberry Pi, not model accuracy
or a full-model performance claim.

## Remaining work

- full-model calibration and accuracy/perplexity evaluation;
- graph-wide quantization and mixed-precision partitioning;
- INT4, AWQ, and GPTQ candidate/materialization paths;
- a predictive rather than shape-specific measured cost model;
- NPU execution, memory-space/DMA IR, and transfer-cost modeling.
