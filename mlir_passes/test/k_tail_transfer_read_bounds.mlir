// RUN: mlir-opt %s | FileCheck %s

// A transfer padding value defines the value of out-of-range lanes; it does
// not promise a masked machine load. Only the row dimension is statically
// known to be in bounds here. The dynamic K-tail dimension must remain false.
// The permutation map maps dimensions and provides no bounds protection.
func.func @k_tail_transfer_read(%source: tensor<8x?xf32>, %k: index)
    -> vector<8x8xf32> {
  %c0 = arith.constant 0 : index
  %zero = arith.constant 0.0 : f32
  %tile = vector.transfer_read %source[%c0, %k], %zero
      {in_bounds = [true, false],
       permutation_map = affine_map<(d0, d1) -> (d0, d1)>}
      : tensor<8x?xf32>, vector<8x8xf32>
  return %tile : vector<8x8xf32>
}

// CHECK-LABEL: func.func @k_tail_transfer_read
// CHECK: vector.transfer_read
// CHECK-SAME: in_bounds = [true, false]
// CHECK-NOT: in_bounds = [true, true]
