module {
  llvm.func @free(!llvm.ptr)
  llvm.func @malloc(i64) -> !llvm.ptr
  llvm.func @matmul_bias_relu_8x8x8(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: i64, %arg3: i64, %arg4: i64, %arg5: i64, %arg6: i64, %arg7: !llvm.ptr, %arg8: !llvm.ptr, %arg9: i64, %arg10: i64, %arg11: i64, %arg12: i64, %arg13: i64, %arg14: !llvm.ptr, %arg15: !llvm.ptr, %arg16: i64, %arg17: i64, %arg18: i64, %arg19: i64, %arg20: i64) -> !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> attributes {llvm.emit_c_interface} {
    %0 = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %1 = llvm.insertvalue %arg14, %0[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %2 = llvm.insertvalue %arg15, %1[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %3 = llvm.insertvalue %arg16, %2[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %4 = llvm.insertvalue %arg17, %3[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %5 = llvm.insertvalue %arg19, %4[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %6 = llvm.insertvalue %arg18, %5[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %7 = llvm.insertvalue %arg20, %6[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %8 = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %9 = llvm.insertvalue %arg7, %8[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %10 = llvm.insertvalue %arg8, %9[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %11 = llvm.insertvalue %arg9, %10[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %12 = llvm.insertvalue %arg10, %11[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %13 = llvm.insertvalue %arg12, %12[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %14 = llvm.insertvalue %arg11, %13[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %15 = llvm.insertvalue %arg13, %14[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %16 = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %17 = llvm.insertvalue %arg0, %16[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %18 = llvm.insertvalue %arg1, %17[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %19 = llvm.insertvalue %arg2, %18[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %20 = llvm.insertvalue %arg3, %19[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %21 = llvm.insertvalue %arg5, %20[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %22 = llvm.insertvalue %arg4, %21[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %23 = llvm.insertvalue %arg6, %22[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %24 = llvm.mlir.constant(1 : index) : i64
    %25 = llvm.mlir.constant(8 : index) : i64
    %26 = llvm.mlir.constant(0 : index) : i64
    %27 = llvm.mlir.constant(0.000000e+00 : f32) : f32
    %28 = llvm.mlir.constant(8 : index) : i64
    %29 = llvm.mlir.constant(8 : index) : i64
    %30 = llvm.mlir.constant(1 : index) : i64
    %31 = llvm.mlir.constant(64 : index) : i64
    %32 = llvm.mlir.zero : !llvm.ptr
    %33 = llvm.getelementptr %32[%31] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %34 = llvm.ptrtoint %33 : !llvm.ptr to i64
    %35 = llvm.mlir.constant(64 : index) : i64
    %36 = llvm.add %34, %35 : i64
    %37 = llvm.call @malloc(%36) : (i64) -> !llvm.ptr
    %38 = llvm.ptrtoint %37 : !llvm.ptr to i64
    %39 = llvm.mlir.constant(1 : index) : i64
    %40 = llvm.sub %35, %39 : i64
    %41 = llvm.add %38, %40 : i64
    %42 = llvm.urem %41, %35 : i64
    %43 = llvm.sub %41, %42 : i64
    %44 = llvm.inttoptr %43 : i64 to !llvm.ptr
    %45 = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %46 = llvm.insertvalue %37, %45[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %47 = llvm.insertvalue %44, %46[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %48 = llvm.mlir.constant(0 : index) : i64
    %49 = llvm.insertvalue %48, %47[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %50 = llvm.insertvalue %28, %49[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %51 = llvm.insertvalue %29, %50[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %52 = llvm.insertvalue %29, %51[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %53 = llvm.insertvalue %30, %52[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    llvm.br ^bb1(%26 : i64)
  ^bb1(%54: i64):  // 2 preds: ^bb0, ^bb8
    %55 = llvm.icmp "slt" %54, %25 : i64
    llvm.cond_br %55, ^bb2, ^bb9
  ^bb2:  // pred: ^bb1
    llvm.br ^bb3(%26 : i64)
  ^bb3(%56: i64):  // 2 preds: ^bb2, ^bb7
    %57 = llvm.icmp "slt" %56, %25 : i64
    llvm.cond_br %57, ^bb4, ^bb8
  ^bb4:  // pred: ^bb3
    llvm.br ^bb5(%26 : i64)
  ^bb5(%58: i64):  // 2 preds: ^bb4, ^bb6
    %59 = llvm.icmp "slt" %58, %25 : i64
    llvm.cond_br %59, ^bb6, ^bb7
  ^bb6:  // pred: ^bb5
    %60 = llvm.extractvalue %23[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %61 = llvm.extractvalue %23[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %62 = llvm.getelementptr %60[%61] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %63 = llvm.extractvalue %23[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %64 = llvm.mul %54, %63 overflow<nsw, nuw> : i64
    %65 = llvm.extractvalue %23[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %66 = llvm.mul %58, %65 overflow<nsw, nuw> : i64
    %67 = llvm.add %64, %66 overflow<nsw, nuw> : i64
    %68 = llvm.getelementptr inbounds|nuw %62[%67] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %69 = llvm.load %68 : !llvm.ptr -> f32
    %70 = llvm.extractvalue %15[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %71 = llvm.extractvalue %15[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %72 = llvm.getelementptr %70[%71] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %73 = llvm.extractvalue %15[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %74 = llvm.mul %58, %73 overflow<nsw, nuw> : i64
    %75 = llvm.extractvalue %15[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %76 = llvm.mul %56, %75 overflow<nsw, nuw> : i64
    %77 = llvm.add %74, %76 overflow<nsw, nuw> : i64
    %78 = llvm.getelementptr inbounds|nuw %72[%77] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %79 = llvm.load %78 : !llvm.ptr -> f32
    %80 = llvm.extractvalue %53[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %81 = llvm.mlir.constant(8 : index) : i64
    %82 = llvm.mul %54, %81 overflow<nsw, nuw> : i64
    %83 = llvm.add %82, %56 overflow<nsw, nuw> : i64
    %84 = llvm.getelementptr inbounds|nuw %80[%83] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %85 = llvm.load %84 : !llvm.ptr -> f32
    %86 = llvm.fmul %69, %79 : f32
    %87 = llvm.fadd %85, %86 : f32
    %88 = llvm.extractvalue %53[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %89 = llvm.mlir.constant(8 : index) : i64
    %90 = llvm.mul %54, %89 overflow<nsw, nuw> : i64
    %91 = llvm.add %90, %56 overflow<nsw, nuw> : i64
    %92 = llvm.getelementptr inbounds|nuw %88[%91] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %87, %92 : f32, !llvm.ptr
    %93 = llvm.add %58, %24 : i64
    llvm.br ^bb5(%93 : i64)
  ^bb7:  // pred: ^bb5
    %94 = llvm.add %56, %24 : i64
    llvm.br ^bb3(%94 : i64)
  ^bb8:  // pred: ^bb3
    %95 = llvm.add %54, %24 : i64
    llvm.br ^bb1(%95 : i64)
  ^bb9:  // pred: ^bb1
    %96 = llvm.mlir.constant(8 : index) : i64
    %97 = llvm.mlir.constant(8 : index) : i64
    %98 = llvm.mlir.constant(1 : index) : i64
    %99 = llvm.mlir.constant(64 : index) : i64
    %100 = llvm.mlir.zero : !llvm.ptr
    %101 = llvm.getelementptr %100[%99] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %102 = llvm.ptrtoint %101 : !llvm.ptr to i64
    %103 = llvm.mlir.constant(64 : index) : i64
    %104 = llvm.add %102, %103 : i64
    %105 = llvm.call @malloc(%104) : (i64) -> !llvm.ptr
    %106 = llvm.ptrtoint %105 : !llvm.ptr to i64
    %107 = llvm.mlir.constant(1 : index) : i64
    %108 = llvm.sub %103, %107 : i64
    %109 = llvm.add %106, %108 : i64
    %110 = llvm.urem %109, %103 : i64
    %111 = llvm.sub %109, %110 : i64
    %112 = llvm.inttoptr %111 : i64 to !llvm.ptr
    %113 = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %114 = llvm.insertvalue %105, %113[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %115 = llvm.insertvalue %112, %114[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %116 = llvm.mlir.constant(0 : index) : i64
    %117 = llvm.insertvalue %116, %115[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %118 = llvm.insertvalue %96, %117[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %119 = llvm.insertvalue %97, %118[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %120 = llvm.insertvalue %97, %119[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %121 = llvm.insertvalue %98, %120[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    llvm.br ^bb10(%26 : i64)
  ^bb10(%122: i64):  // 2 preds: ^bb9, ^bb14
    %123 = llvm.icmp "slt" %122, %25 : i64
    llvm.cond_br %123, ^bb11, ^bb15
  ^bb11:  // pred: ^bb10
    llvm.br ^bb12(%26 : i64)
  ^bb12(%124: i64):  // 2 preds: ^bb11, ^bb13
    %125 = llvm.icmp "slt" %124, %25 : i64
    llvm.cond_br %125, ^bb13, ^bb14
  ^bb13:  // pred: ^bb12
    %126 = llvm.extractvalue %53[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %127 = llvm.mlir.constant(8 : index) : i64
    %128 = llvm.mul %122, %127 overflow<nsw, nuw> : i64
    %129 = llvm.add %128, %124 overflow<nsw, nuw> : i64
    %130 = llvm.getelementptr inbounds|nuw %126[%129] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %131 = llvm.load %130 : !llvm.ptr -> f32
    %132 = llvm.extractvalue %7[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %133 = llvm.extractvalue %7[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %134 = llvm.getelementptr %132[%133] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %135 = llvm.extractvalue %7[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %136 = llvm.mul %122, %135 overflow<nsw, nuw> : i64
    %137 = llvm.extractvalue %7[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %138 = llvm.mul %124, %137 overflow<nsw, nuw> : i64
    %139 = llvm.add %136, %138 overflow<nsw, nuw> : i64
    %140 = llvm.getelementptr inbounds|nuw %134[%139] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %141 = llvm.load %140 : !llvm.ptr -> f32
    %142 = llvm.fadd %131, %141 : f32
    %143 = llvm.intr.maximum(%142, %27) : (f32, f32) -> f32
    %144 = llvm.extractvalue %121[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %145 = llvm.mlir.constant(8 : index) : i64
    %146 = llvm.mul %122, %145 overflow<nsw, nuw> : i64
    %147 = llvm.add %146, %124 overflow<nsw, nuw> : i64
    %148 = llvm.getelementptr inbounds|nuw %144[%147] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %143, %148 : f32, !llvm.ptr
    %149 = llvm.add %124, %24 : i64
    llvm.br ^bb12(%149 : i64)
  ^bb14:  // pred: ^bb12
    %150 = llvm.add %122, %24 : i64
    llvm.br ^bb10(%150 : i64)
  ^bb15:  // pred: ^bb10
    %151 = llvm.extractvalue %53[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    llvm.call @free(%151) : (!llvm.ptr) -> ()
    llvm.return %121 : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
  }
  llvm.func @_mlir_ciface_matmul_bias_relu_8x8x8(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: !llvm.ptr, %arg3: !llvm.ptr) attributes {llvm.emit_c_interface} {
    %0 = llvm.load %arg1 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %1 = llvm.extractvalue %0[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %2 = llvm.extractvalue %0[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %3 = llvm.extractvalue %0[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %4 = llvm.extractvalue %0[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %5 = llvm.extractvalue %0[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %6 = llvm.extractvalue %0[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %7 = llvm.extractvalue %0[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %8 = llvm.load %arg2 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %9 = llvm.extractvalue %8[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %10 = llvm.extractvalue %8[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %11 = llvm.extractvalue %8[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %12 = llvm.extractvalue %8[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %13 = llvm.extractvalue %8[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %14 = llvm.extractvalue %8[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %15 = llvm.extractvalue %8[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %16 = llvm.load %arg3 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %17 = llvm.extractvalue %16[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %18 = llvm.extractvalue %16[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %19 = llvm.extractvalue %16[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %20 = llvm.extractvalue %16[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %21 = llvm.extractvalue %16[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %22 = llvm.extractvalue %16[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %23 = llvm.extractvalue %16[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %24 = llvm.call @matmul_bias_relu_8x8x8(%1, %2, %3, %4, %5, %6, %7, %9, %10, %11, %12, %13, %14, %15, %17, %18, %19, %20, %21, %22, %23) : (!llvm.ptr, !llvm.ptr, i64, i64, i64, i64, i64, !llvm.ptr, !llvm.ptr, i64, i64, i64, i64, i64, !llvm.ptr, !llvm.ptr, i64, i64, i64, i64, i64) -> !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    llvm.store %24, %arg0 : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>, !llvm.ptr
    llvm.return
  }
}

