#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
module {
  func.func @matmul_bias_relu_tiled_32x32x32(%arg0: tensor<32x32xf32>, %arg1: tensor<32x32xf32>, %arg2: tensor<32x32xf32>) -> tensor<32x32xf32> attributes {llvm.emit_c_interface} {
    %0 = ub.poison : f32
    %cst = arith.constant dense<0.000000e+00> : vector<8x8xf32>
    %c8 = arith.constant 8 : index
    %c32 = arith.constant 32 : index
    %c0 = arith.constant 0 : index
    %1 = tensor.empty() : tensor<32x32xf32>
    %2 = tensor.empty() : tensor<32x32xf32>
    %3 = scf.for %arg3 = %c0 to %c32 step %c8 iter_args(%arg4 = %2) -> (tensor<32x32xf32>) {
      %4 = scf.for %arg5 = %c0 to %c32 step %c8 iter_args(%arg6 = %arg4) -> (tensor<32x32xf32>) {
        %extracted_slice = tensor.extract_slice %1[%arg3, %arg5] [8, 8] [1, 1] : tensor<32x32xf32> to tensor<8x8xf32>
        %5 = vector.transfer_write %cst, %extracted_slice[%c0, %c0] {in_bounds = [true, true]} : vector<8x8xf32>, tensor<8x8xf32>
        %6 = scf.for %arg7 = %c0 to %c32 step %c8 iter_args(%arg8 = %5) -> (tensor<8x8xf32>) {
          %12 = vector.transfer_read %arg0[%arg3, %arg7], %0 {in_bounds = [true, true]} : tensor<32x32xf32>, vector<8x8xf32>
          %13 = vector.transfer_read %arg1[%arg7, %arg5], %0 {in_bounds = [true, true]} : tensor<32x32xf32>, vector<8x8xf32>
          %14 = vector.transfer_read %arg8[%c0, %c0], %0 {in_bounds = [true, true]} : tensor<8x8xf32>, vector<8x8xf32>
          %15 = vector.contract {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %12, %13, %14 : vector<8x8xf32>, vector<8x8xf32> into vector<8x8xf32>
          %16 = vector.transfer_write %15, %arg8[%c0, %c0] {in_bounds = [true, true]} : vector<8x8xf32>, tensor<8x8xf32>
          scf.yield %16 : tensor<8x8xf32>
        }
        %7 = vector.transfer_read %6[%c0, %c0], %0 {in_bounds = [true, true]} : tensor<8x8xf32>, vector<8x8xf32>
        %8 = vector.transfer_read %arg2[%arg3, %arg5], %0 {in_bounds = [true, true]} : tensor<32x32xf32>, vector<8x8xf32>
        %9 = arith.addf %7, %8 : vector<8x8xf32>
        %10 = arith.maximumf %9, %cst : vector<8x8xf32>
        %11 = vector.transfer_write %10, %arg6[%arg3, %arg5] {in_bounds = [true, true]} : vector<8x8xf32>, tensor<32x32xf32>
        scf.yield %11 : tensor<32x32xf32>
      }
      scf.yield %4 : tensor<32x32xf32>
    }
    return %3 : tensor<32x32xf32>
  }
}

