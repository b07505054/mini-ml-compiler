module {
  func.func @matmul_linalg(
      %A: memref<128x128xf32>,
      %B: memref<128x128xf32>,
      %C: memref<128x128xf32>) {

    linalg.matmul
      ins(%A, %B : memref<128x128xf32>, memref<128x128xf32>)
      outs(%C : memref<128x128xf32>)

    return
  }
}