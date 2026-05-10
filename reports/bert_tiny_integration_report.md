# BERT Tiny ONNX Integration Report

## Model Summary

- Model path: `models/bert_tiny.onnx`
- Inputs: 2
- Outputs: 2
- Total nodes: 273
- Initializers: 59
- Approx operator coverage: **69.23%**

## Supported Operators

- `Add`: 52
- `Gemm`: 1
- `LayerNormalization`: 11
- `MatMul`: 40
- `Softmax`: 5

## Partially Supported Operators

- `Div`: 5
- `Mul`: 20
- `Reshape`: 30
- `Transpose`: 25

## Unsupported Operators

- `Concat`: 17 | Priority: High
- `Slice`: 16 | Priority: High
- `Shape`: 8 | Priority: Medium
- `Unsqueeze`: 7 | Priority: Medium
- `Where`: 6 | Priority: Medium
- `IsNaN`: 5 | Priority: Unassigned
- `Erf`: 5 | Priority: Medium
- `Expand`: 4 | Priority: Low
- `Gather`: 4 | Priority: Medium
- `Squeeze`: 2 | Priority: Unassigned
- `Range`: 2 | Priority: Unassigned
- `And`: 2 | Priority: Unassigned
- `GatherElements`: 1 | Priority: Unassigned
- `Cast`: 1 | Priority: Low
- `GreaterOrEqual`: 1 | Priority: Unassigned
- `Max`: 1 | Priority: Unassigned
- `GatherND`: 1 | Priority: Unassigned
- `Tanh`: 1 | Priority: Unassigned

## Top Integration Gaps

- `Concat` appears 17 times and is marked as `High` priority.
- `Slice` appears 16 times and is marked as `High` priority.
- `Shape` appears 8 times and is marked as `Medium` priority.
- `Unsqueeze` appears 7 times and is marked as `Medium` priority.
- `Where` appears 6 times and is marked as `Medium` priority.
- `IsNaN` appears 5 times and is marked as `Unassigned` priority.
- `Erf` appears 5 times and is marked as `Medium` priority.
- `Expand` appears 4 times and is marked as `Low` priority.

## Recommended Operator Onboarding Plan

1. Add `Slice` and `Concat` because they are high-frequency graph-structure operators.
2. Add `Shape`, `Unsqueeze`, and `Gather` to improve dynamic-shape graph support.
3. Add `Where` and `Erf` to support more Transformer activation and masking patterns.
4. Continue validating each operator with unit-level correctness tests and ONNX graph coverage reports.

## Engineering Notes

- `LayerNormalization` has been onboarded into the runtime as a CPU kernel.
- `MatMul`, `Add`, `Gemm`, and `Softmax` map to existing runtime operators.
- This report is intended to guide incremental model integration and runtime operator coverage expansion.