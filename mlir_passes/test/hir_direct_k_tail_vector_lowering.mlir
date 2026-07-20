// RUN: mlir-opt %s --pass-pipeline='builtin.module(func.func(hir-direct-k-tail-vector-lowering))' | FileCheck %s

func.func @direct_k_tail(
    %lhs: tensor<8x15xf32>, %rhs: tensor<15x8xf32>,
    %bias: tensor<8x8xf32>) -> tensor<8x8xf32>
    attributes {llvm.emit_c_interface} {
  %0 = hir.fused_matmul_bias_relu %lhs, %rhs, %bias {
    fusion.candidate = "matmul_bias_relu",
    kernel.selection = "native_cpu",
    lowering.source = "linalg.matmul_add_relu",
    target.padding = "none"
  } : (tensor<8x15xf32>, tensor<15x8xf32>, tensor<8x8xf32>)
      -> tensor<8x8xf32>
  return %0 : tensor<8x8xf32>
}

// CHECK-LABEL: func.func @direct_k_tail
// CHECK: scf.for {{.*}} to %{{.*}} step %{{.*}} iter_args
// CHECK: vector.contract
// CHECK: scf.for {{.*}} to %{{.*}} step %{{.*}} iter_args
// CHECK: vector.transfer_read {{.*}} {in_bounds = [true], permutation_map =
// CHECK: vector.transfer_read {{.*}} {in_bounds = [true]}
// CHECK: vector.outerproduct
// CHECK: arith.addf
// CHECK: arith.maximumf
// CHECK: vector.transfer_write
// CHECK-SAME: in_bounds = [true, true]
// CHECK: lowering.k_remainder = 7
// CHECK-SAME: lowering.schedule = "tiled_vector_direct_k_tail"
// CHECK-NOT: tensor.pad
// CHECK-NOT: linalg.
