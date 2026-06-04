# IREE StableHLO Subset Comparison

Status: `ok`

## Path

`stablehlo textual subset -> linalg/arith/math -> iree-compile -> VM/HAL`

## Cases

| Case | Compiled | VM module | HAL executable |
|---|---:|---:|---:|
| rmsnorm | True | True | True |
| matmul_bias_relu | True | True | True |

## Notes

- This is a comparison layer only; it does not replace the HIR runtime path.
- Runtime execution is skipped unless `iree-run-module` / IREE runtime is installed.
- iree-compile: `/Users/allen/Documents/Codex/project/ml-graph-compiler-runtime/.venv/bin/iree-compile`
