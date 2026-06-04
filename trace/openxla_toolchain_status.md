# OpenXLA Toolchain Status

Status: `stablehlo_unavailable_skip_native_tests`

## Commands

- `stablehlo-opt`: available=`False`, path=`None`, version=`None`
- `torch-mlir-opt`: available=`False`, path=`None`, version=`None`
- `iree-compile`: available=`True`, path=`/Users/allen/Documents/Codex/project/ml-graph-compiler-runtime/.venv/bin/iree-compile`, version=`IREE (https://iree.dev):`
- `iree-run-module`: available=`False`, path=`None`, version=`None`
- `mlir-opt`: available=`True`, path=`/opt/homebrew/opt/llvm/bin/mlir-opt`, version=`Homebrew LLVM version 22.1.5`
- `mlir-runner`: available=`True`, path=`/opt/homebrew/opt/llvm/bin/mlir-runner`, version=`Homebrew LLVM version 22.1.5`

## Python Modules

- `jax`: available=`False`, origin=`None`
- `tensorflow`: available=`False`, origin=`None`
- `torch_mlir`: available=`False`, origin=`None`
- `stablehlo`: available=`False`, origin=`None`

## Compiler Path

- StableHLO-native tests are skipped until `stablehlo-opt` is installed.
- The current repo uses StableHLO-compatible Linalg/Arith decompositions for FileCheck coverage.
- This keeps the MLIR/HIR/LLVM executable path testable without vendoring OpenXLA tools.
