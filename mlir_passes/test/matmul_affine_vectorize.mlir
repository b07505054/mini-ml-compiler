// RUN: mlir-opt %s --affine-super-vectorize="virtual-vector-size=4 test-fastest-varying=0" | FileCheck %s

func.func @affine_vector_add(
  %A: memref<128xf32>,
  %B: memref<128xf32>,
  %C: memref<128xf32>
) {
  affine.for %i = 0 to 128 {
    %a = affine.load %A[%i] : memref<128xf32>
    %b = affine.load %B[%i] : memref<128xf32>
    %sum = arith.addf %a, %b : f32
    affine.store %sum, %C[%i] : memref<128xf32>
  }
  return
}

// CHECK: vector