#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
module {
  func.func @matmul_bias_relu_vectorized_32x32x32(%arg0: tensor<32x32xf32>, %arg1: tensor<32x32xf32>, %arg2: tensor<32x32xf32>) -> tensor<32x32xf32> attributes {llvm.emit_c_interface} {
    %0 = ub.poison : f32
    %cst = arith.constant dense<0.000000e+00> : vector<32x32xf32>
    %c0 = arith.constant 0 : index
    %1 = vector.transfer_read %arg0[%c0, %c0], %0 {in_bounds = [true, true]} : tensor<32x32xf32>, vector<32x32xf32>
    %2 = vector.transfer_read %arg1[%c0, %c0], %0 {in_bounds = [true, true]} : tensor<32x32xf32>, vector<32x32xf32>
    %3 = vector.contract {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %1, %2, %cst : vector<32x32xf32>, vector<32x32xf32> into vector<32x32xf32>
    %4 = tensor.empty() : tensor<32x32xf32>
    %5 = vector.transfer_read %arg2[%c0, %c0], %0 {in_bounds = [true, true]} : tensor<32x32xf32>, vector<32x32xf32>
    %6 = arith.addf %3, %5 : vector<32x32xf32>
    %7 = arith.maximumf %6, %cst : vector<32x32xf32>
    %8 = vector.transfer_write %7, %4[%c0, %c0] {in_bounds = [true, true]} : vector<32x32xf32>, tensor<32x32xf32>
    return %8 : tensor<32x32xf32>
  }
}

