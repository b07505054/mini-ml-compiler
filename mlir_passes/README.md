## Run the Fusion Pipeline

This plugin is registered as an MLIR pass pipeline. Use the same `mlir-opt`
binary from the LLVM/MLIR build used by CMake.

```bash
/Users/allen/Developer/llvm-build/bin/mlir-opt \
  --load-pass-plugin=build-mlir/HIRMatMulBiasReluFusionPass.dylib \
  mlir_passes/test/matmul_bias_relu.mlir \
  --pass-pipeline='builtin.module(matmul-bias-relu-fusion)'