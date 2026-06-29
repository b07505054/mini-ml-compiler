// Registered CV dialect parser/printer smoke test.

// CHECK-LABEL: func.func @cv_dialect_smoke
// CHECK: cv.conv2d
// CHECK: cv.batch_norm
// CHECK: cv.silu
// CHECK: cv.upsample
// CHECK: cv.concat
// CHECK: cv.detect_head
// CHECK: cv.prototype_head

func.func @cv_dialect_smoke(%x: tensor<1x4x2x2xf16>) -> tensor<1x4x2x2xf16> {
  %a = cv.conv2d %x {cv.source_op = "Conv_0"}
      : (tensor<1x4x2x2xf16>) -> tensor<1x4x2x2xf16>
  %b = cv.batch_norm %a {cv.source_op = "BatchNorm_0"}
      : (tensor<1x4x2x2xf16>) -> tensor<1x4x2x2xf16>
  %c = cv.silu %b {cv.source_op = "SiLU_0"}
      : (tensor<1x4x2x2xf16>) -> tensor<1x4x2x2xf16>
  %d = cv.upsample %c {cv.source_op = "Resize_0"}
      : (tensor<1x4x2x2xf16>) -> tensor<1x4x2x2xf16>
  %e = cv.concat %c, %d {cv.source_op = "Concat_0"}
      : (tensor<1x4x2x2xf16>, tensor<1x4x2x2xf16>) -> tensor<1x4x2x2xf16>
  %f = cv.detect_head %c, %d, %e {cv.source_op = "Detect_0"}
      : (tensor<1x4x2x2xf16>, tensor<1x4x2x2xf16>, tensor<1x4x2x2xf16>)
        -> tensor<1x4x2x2xf16>
  %g = cv.prototype_head %f {cv.source_op = "Proto_0"}
      : (tensor<1x4x2x2xf16>) -> tensor<1x4x2x2xf16>
  return %g : tensor<1x4x2x2xf16>
}
