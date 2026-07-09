# YOLO-Seg Model Setup

## Model Path

Place the real YOLO-Seg ONNX model at:

```text
models/yolo-seg.onnx
```

If the model uses ONNX external tensor data, place the sidecar at:

```text
models/yolo-seg.onnx.data
```

The YOLO-Seg model binary is intentionally not committed. `.gitignore` excludes
`models/yolo-seg.onnx`, `models/yolo-seg.onnx.data`, and generated frontend
artifacts under `artifacts/yoloseg_generic_frontend/`.

## Run The Generic Frontend

Direct command:

```bash
.venv/bin/python tools/run_generic_onnx_frontend.py models/yolo-seg.onnx artifacts/yoloseg_generic_frontend --prefix yoloseg
```

Convenience script:

```bash
scripts/run_yoloseg_generic_frontend.sh
```

The script exits successfully with a clear message when `models/yolo-seg.onnx`
is absent, so CI does not require the real model.

## Expected Outputs

When the model is present, the frontend writes:

```text
artifacts/yoloseg_generic_frontend/yoloseg.imported_graph_ir.json
artifacts/yoloseg_generic_frontend/yoloseg.generic_graph_ir.json
artifacts/yoloseg_generic_frontend/yoloseg.canonical_generic_graph_ir.json
artifacts/yoloseg_generic_frontend/yoloseg.shape_generic_graph_ir.json
artifacts/yoloseg_generic_frontend/yoloseg.diagnostics_report.json
artifacts/yoloseg_generic_frontend/yoloseg.frontend_report.json
```

## Interpreting Diagnostics

Start with:

```text
artifacts/yoloseg_generic_frontend/yoloseg.diagnostics_report.json
```

Important fields:

- `frontend_readiness_status`: overall generic frontend readiness.
- `op_histogram`: generic op distribution after ONNX op normalization.
- `unknown_op_count`: number of `nn.unknown` nodes.
- `unknown_source_op_types`: ONNX op types that need generic op mapping.
- `shape_inference_status_histogram`: counts of `inferred`,
  `partially_inferred`, `unknown`, and `error` node statuses.
- `shape_error_nodes`: nodes with obvious shape/type inconsistencies or missing
  shape rules.
- `shape_unknown_nodes`: nodes where v0 inference could not infer output
  metadata.
- `shape_partially_inferred_nodes`: nodes with incomplete but useful output
  metadata.
- `metadata_counts`: unresolved or missing dtype/shape metadata counts.
- `top_initializers_by_raw_data_bytes`: largest initializer records.

Readiness values:

- `ready_for_generic_lowering`: the graph verifies, ops are in the generic
  vocabulary, and all node shapes were inferred.
- `needs_op_support`: the graph verifies but contains `nn.unknown`.
- `needs_shape_support`: the graph verifies but has `error`, `unknown`, or
  `partially_inferred` shape statuses.
- `invalid_ir`: the graph failed GenericGraphIR verification.

This workflow does not add YOLO/CV recognition, LLM recognition, MLIR lowering,
ExecutionPlan generation, or Qwen GraphFacts changes.
