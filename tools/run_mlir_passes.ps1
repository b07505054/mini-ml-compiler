Write-Host "Running MLIR canonicalization + CSE on affine MatMul..."
mlir-opt mlir\matmul_affine.mlir --canonicalize --cse > mlir\out\matmul_canonicalized.mlir

Write-Host "Running MLIR affine loop tiling..."
mlir-opt mlir\matmul_affine.mlir --affine-loop-tile="tile-sizes=32,32,32" > mlir\out\matmul_tiled.mlir

Write-Host "Running MLIR canonicalization + CSE on linalg MatMul..."
mlir-opt mlir\matmul_linalg.mlir --canonicalize --cse > mlir\out\matmul_linalg_canonicalized.mlir

Write-Host "Lowering linalg.matmul to affine loops..."
mlir-opt mlir\matmul_linalg.mlir --convert-linalg-to-affine-loops > mlir\out\matmul_linalg_to_affine.mlir

Write-Host "Done. Outputs written to mlir\out\"