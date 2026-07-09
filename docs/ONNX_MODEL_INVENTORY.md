# ONNX Model Inventory

This inventory separates project-owned ONNX files from dependency or generated
ONNX files. It intentionally excludes `.venv/`, `build/`, `build-mlir/`,
`mlir_passes/build/`, `third_party/`, `external/`, `artifacts/`, and
`integration_bundle/`.

Regenerate the project inventory with:

```bash
scripts/list_project_onnx_models.sh
```

## Real Project ONNX Files

| Path | Size | Classification | Recommended diagnostics use |
| --- | ---: | --- | --- |
| `models/bert_tiny.onnx` | 38 KB | frontend regression fixture | Good small real-model-style smoke for generic frontend diagnostics. |
| `models/tiny_mlp.onnx` | 534 B | tiny fixture | Good minimal import/verify smoke; too small for coverage decisions. |
| `models/matmul_add_relu.onnx` | 16 MB | frontend regression fixture | Good operator coverage fixture for MatMul/Add/Relu and initializer metadata. |

No large real project ONNX model is currently present.

## Fixture/Test ONNX Files

The current project ONNX files are all fixtures:

- `models/tiny_mlp.onnx`: tiny synthetic fixture.
- `models/bert_tiny.onnx`: small frontend regression fixture.
- `models/matmul_add_relu.onnx`: larger frontend regression fixture.

Tests also generate temporary synthetic ONNX graphs in pytest temp directories;
those are not committed project models.

## Missing Expected Models

| Path | Status | Classification |
| --- | --- | --- |
| `models/yolo-seg.onnx` | missing | missing expected model |
| `models/yolo-seg.onnx.data` | missing | optional missing expected external-data sidecar |

The YOLO-Seg model binary is intentionally not committed. `.gitignore` excludes
`models/yolo-seg.onnx`, `models/yolo-seg.onnx.data`, and generated outputs
under `artifacts/yoloseg_generic_frontend/`.

## Ignored Dependency ONNX Files

The local `.venv/` contains 1,914 ONNX files from dependency test data,
primarily under the installed `onnx` package's backend test suite. These are
not project models and should not be used for project readiness reporting.

Current ignored build-output ONNX count under `build/`, `build-mlir/`, and
`mlir_passes/build/`: 0.

## Recommended Canonical Model Paths

Use these paths for project-owned ONNX diagnostics:

- `models/tiny_mlp.onnx`
- `models/bert_tiny.onnx`
- `models/matmul_add_relu.onnx`
- `models/yolo-seg.onnx`, when locally supplied

Use this output directory for YOLO-Seg generic frontend artifacts:

```text
artifacts/yoloseg_generic_frontend/
```

## Diagnostics Priority

1. `models/tiny_mlp.onnx`: minimum end-to-end import sanity check.
2. `models/matmul_add_relu.onnx`: generic op and initializer metadata check.
3. `models/bert_tiny.onnx`: broader frontend regression fixture.
4. `models/yolo-seg.onnx`: real model candidate once the local binary is
   supplied.

Do not use dependency ONNX files under `.venv/` for canonical frontend
readiness claims.
