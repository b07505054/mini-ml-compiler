# Generic ONNX Frontend Pipeline

`tools/run_generic_onnx_frontend.py` composes the generic frontend stages into
one end-to-end CLI.

It does not perform domain recognition, MLIR lowering, or ExecutionPlan
generation.

An optional Phase 13 checker validates whether shape-annotated
`GenericGraphIR` satisfies the structural contract for future lowering into
existing upstream MLIR dialects. It also emits no MLIR.

Phase 15 added a separate minimal emitter for a small elementwise subset.
Phase 16 extends that emitter with selected shape/layout forms and nearest 2x
resize. The frontend driver itself still emits no MLIR.

## Pipeline

```text
ONNX
  -> ImportedGraphIR
  -> verify ImportedGraphIR
  -> GenericGraphIR
  -> verify GenericGraphIR
  -> canonicalize GenericGraphIR
  -> verify canonicalized GenericGraphIR
  -> infer shapes/types
  -> verify shape-annotated GenericGraphIR
  -> diagnostics/readiness report
```

## Existing-MLIR Lowering Contract

```bash
.venv/bin/python tools/check_generic_lowering_contract.py \
  --in artifacts/yoloseg_generic_frontend/yoloseg.shape_generic_graph_ir.json \
  --out artifacts/yoloseg_generic_frontend/yoloseg.lowering_contract.json
```

The checker distinguishes valid frontend output from operations that still
need an existing-dialect lowering strategy. See
`docs/GENERIC_GRAPH_IR_TO_MLIR_LOWERING_CONTRACT.md`.

## Minimal Existing-MLIR Emitter

```bash
.venv/bin/python tools/generic_graph_ir_to_mlir.py \
  --in path/to/shape_generic_graph_ir.json \
  --out path/to/module.mlir
```

The v0 emitter is intentionally smaller than the lowering contract. It emits
`nn.constant`, `nn.identity`, `nn.add`, `nn.sub`, `nn.mul`, `nn.div`,
`nn.relu`, `nn.sigmoid`, `nn.reshape`, `nn.transpose`, and the selected
`nn.resize` subset for static `f32` tensor forms. See
`docs/GENERIC_GRAPH_IR_TO_MLIR_EMITTER.md`.

## CLI

```bash
tools/run_generic_onnx_frontend.py model.onnx out/generic_frontend \
  --prefix model \
  --stop-after shapes
```

Options:

- `--prefix`: output filename prefix. Defaults to the ONNX file stem.
- `--stop-after imported|generic|canonicalized|shapes`: stop after a stage.
- `--keep-going`: continue report generation after a failed stage where
  possible. Defaults to false.

## Artifacts

For prefix `model`, the driver writes:

- `model.imported_graph_ir.json`
- `model.generic_graph_ir.json`
- `model.canonical_generic_graph_ir.json`
- `model.shape_generic_graph_ir.json`
- `model.diagnostics_report.json`
- `model.frontend_report.json`

Earlier stop stages write only artifacts up to that stage plus the report.
Diagnostics are emitted only after shape/type inference completes.

## Report

`frontend_report.json` includes:

- input path
- artifact paths
- per-stage status
- per-stage verifier status
- node count
- op histogram
- unknown op count
- shape inference summary for the final stage
- diagnostics artifact path and readiness status, when diagnostics were emitted
- truth boundary:
  `generic_onnx_frontend_metadata_only_no_domain_recognition_no_mlir_lowering_no_execution_plan_generation`

## Diagnostics Report

`tools/diagnose_generic_graph_ir.py` reads shape-annotated `GenericGraphIR` and
writes diagnostics JSON. It is a reporting pass only; it does not recognize
domains, lower to MLIR, or generate an ExecutionPlan.

The diagnostics report includes:

- graph name
- node, value, and initializer counts
- generic op histogram
- unknown op count
- unknown source op types
- shape inference status histogram
- nodes with `shape_inference_status` equal to `error`, `unknown`, or
  `partially_inferred`
- unresolved or missing dtype/shape metadata counts
- largest initializers by `raw_data_bytes`
- verifier status
- frontend readiness status
- truth boundary:
  `diagnostics_only_no_domain_recognition_no_mlir_lowering_no_execution_plan_generation`

Readiness statuses:

- `ready_for_generic_lowering`: IR verifies, all ops are supported by the
  generic vocabulary, and shape/type inference completed for all nodes.
- `needs_op_support`: IR verifies but contains one or more `nn.unknown` nodes.
- `needs_shape_support`: IR verifies and has supported ops, but one or more
  nodes has `error`, `unknown`, or `partially_inferred` shape status.
- `invalid_ir`: the input failed `GenericGraphIR` verification.

## Current Generic Op Coverage

The frontend currently maps these ONNX ops into model-agnostic `nn.*` ops:

- arithmetic: `Add`, `Sub`, `Mul`, `Div`
- matrix/linear: `MatMul`, `Gemm`
- convolution/pooling: `Conv`, `ConvTranspose`, `MaxPool`
- tensor shape/layout: `Reshape`, `Transpose`, `Concat`, `Split`, `Slice`,
  `Resize`
- activations/probability: `Sigmoid`, `Relu`, `Softmax`
- constants/passthrough: `Constant`, `Identity`

Unmapped ONNX ops become `nn.unknown` and are reported by
`diagnostics_report.json`.

## Shape-Bearing Constants

The frontend preserves bounded small numeric tensor literals for metadata
tensors while avoiding large weight inlining:

- up to 64 elements
- up to 512 raw bytes
- all initializers keep `raw_data_bytes` metadata
- large weights remain metadata-only

These literals allow canonicalization and shape/type inference to recover
model-agnostic static operands for ops such as `Reshape`, `Slice`, `Resize`,
and `Split`. This is still source-level graph normalization only; it does not
perform domain recognition.

## Relationship To Qwen

The generic frontend pipeline is separate from the legacy Qwen GraphFacts path.
It does not modify or replace:

```text
Qwen ONNX
  -> Qwen GraphFacts
  -> qwen-onnx-to-serving-mlir
  -> LLM dialect
  -> ExecutionPlan
```
