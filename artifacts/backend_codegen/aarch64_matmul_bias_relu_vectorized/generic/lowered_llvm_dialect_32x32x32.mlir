module {
  llvm.func @free(!llvm.ptr)
  llvm.func @malloc(i64) -> !llvm.ptr
  llvm.func @matmul_bias_relu_32x32x32(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: i64, %arg3: i64, %arg4: i64, %arg5: i64, %arg6: i64, %arg7: !llvm.ptr, %arg8: !llvm.ptr, %arg9: i64, %arg10: i64, %arg11: i64, %arg12: i64, %arg13: i64, %arg14: !llvm.ptr, %arg15: !llvm.ptr, %arg16: i64, %arg17: i64, %arg18: i64, %arg19: i64, %arg20: i64) -> !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> attributes {llvm.emit_c_interface} {
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
    %25 = llvm.mlir.constant(32 : index) : i64
    %26 = llvm.mlir.constant(0 : index) : i64
    %27 = llvm.mlir.constant(0.000000e+00 : f32) : f32
    %28 = llvm.mlir.constant(32 : index) : i64
    %29 = llvm.mlir.constant(32 : index) : i64
    %30 = llvm.mlir.constant(1 : index) : i64
    %31 = llvm.mlir.constant(1024 : index) : i64
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
  ^bb1(%54: i64):  // 2 preds: ^bb0, ^bb5
    %55 = llvm.icmp "slt" %54, %25 : i64
    llvm.cond_br %55, ^bb2, ^bb6
  ^bb2:  // pred: ^bb1
    llvm.br ^bb3(%26 : i64)
  ^bb3(%56: i64):  // 2 preds: ^bb2, ^bb4
    %57 = llvm.icmp "slt" %56, %25 : i64
    llvm.cond_br %57, ^bb4, ^bb5
  ^bb4:  // pred: ^bb3
    %58 = llvm.extractvalue %53[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %59 = llvm.mlir.constant(32 : index) : i64
    %60 = llvm.mul %54, %59 overflow<nsw, nuw> : i64
    %61 = llvm.add %60, %56 overflow<nsw, nuw> : i64
    %62 = llvm.getelementptr inbounds|nuw %58[%61] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %27, %62 : f32, !llvm.ptr
    %63 = llvm.add %56, %24 : i64
    llvm.br ^bb3(%63 : i64)
  ^bb5:  // pred: ^bb3
    %64 = llvm.add %54, %24 : i64
    llvm.br ^bb1(%64 : i64)
  ^bb6:  // pred: ^bb1
    llvm.br ^bb7(%26 : i64)
  ^bb7(%65: i64):  // 2 preds: ^bb6, ^bb14
    %66 = llvm.icmp "slt" %65, %25 : i64
    llvm.cond_br %66, ^bb8, ^bb15
  ^bb8:  // pred: ^bb7
    llvm.br ^bb9(%26 : i64)
  ^bb9(%67: i64):  // 2 preds: ^bb8, ^bb13
    %68 = llvm.icmp "slt" %67, %25 : i64
    llvm.cond_br %68, ^bb10, ^bb14
  ^bb10:  // pred: ^bb9
    llvm.br ^bb11(%26 : i64)
  ^bb11(%69: i64):  // 2 preds: ^bb10, ^bb12
    %70 = llvm.icmp "slt" %69, %25 : i64
    llvm.cond_br %70, ^bb12, ^bb13
  ^bb12:  // pred: ^bb11
    %71 = llvm.extractvalue %23[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %72 = llvm.extractvalue %23[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %73 = llvm.getelementptr %71[%72] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %74 = llvm.extractvalue %23[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %75 = llvm.mul %65, %74 overflow<nsw, nuw> : i64
    %76 = llvm.extractvalue %23[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %77 = llvm.mul %69, %76 overflow<nsw, nuw> : i64
    %78 = llvm.add %75, %77 overflow<nsw, nuw> : i64
    %79 = llvm.getelementptr inbounds|nuw %73[%78] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %80 = llvm.load %79 : !llvm.ptr -> f32
    %81 = llvm.extractvalue %15[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %82 = llvm.extractvalue %15[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %83 = llvm.getelementptr %81[%82] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %84 = llvm.extractvalue %15[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %85 = llvm.mul %69, %84 overflow<nsw, nuw> : i64
    %86 = llvm.extractvalue %15[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %87 = llvm.mul %67, %86 overflow<nsw, nuw> : i64
    %88 = llvm.add %85, %87 overflow<nsw, nuw> : i64
    %89 = llvm.getelementptr inbounds|nuw %83[%88] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %90 = llvm.load %89 : !llvm.ptr -> f32
    %91 = llvm.extractvalue %53[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %92 = llvm.mlir.constant(32 : index) : i64
    %93 = llvm.mul %65, %92 overflow<nsw, nuw> : i64
    %94 = llvm.add %93, %67 overflow<nsw, nuw> : i64
    %95 = llvm.getelementptr inbounds|nuw %91[%94] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %96 = llvm.load %95 : !llvm.ptr -> f32
    %97 = llvm.fmul %80, %90 : f32
    %98 = llvm.fadd %96, %97 : f32
    %99 = llvm.extractvalue %53[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %100 = llvm.mlir.constant(32 : index) : i64
    %101 = llvm.mul %65, %100 overflow<nsw, nuw> : i64
    %102 = llvm.add %101, %67 overflow<nsw, nuw> : i64
    %103 = llvm.getelementptr inbounds|nuw %99[%102] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %98, %103 : f32, !llvm.ptr
    %104 = llvm.add %69, %24 : i64
    llvm.br ^bb11(%104 : i64)
  ^bb13:  // pred: ^bb11
    %105 = llvm.add %67, %24 : i64
    llvm.br ^bb9(%105 : i64)
  ^bb14:  // pred: ^bb9
    %106 = llvm.add %65, %24 : i64
    llvm.br ^bb7(%106 : i64)
  ^bb15:  // pred: ^bb7
    %107 = llvm.mlir.constant(32 : index) : i64
    %108 = llvm.mlir.constant(32 : index) : i64
    %109 = llvm.mlir.constant(1 : index) : i64
    %110 = llvm.mlir.constant(1024 : index) : i64
    %111 = llvm.mlir.zero : !llvm.ptr
    %112 = llvm.getelementptr %111[%110] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %113 = llvm.ptrtoint %112 : !llvm.ptr to i64
    %114 = llvm.mlir.constant(64 : index) : i64
    %115 = llvm.add %113, %114 : i64
    %116 = llvm.call @malloc(%115) : (i64) -> !llvm.ptr
    %117 = llvm.ptrtoint %116 : !llvm.ptr to i64
    %118 = llvm.mlir.constant(1 : index) : i64
    %119 = llvm.sub %114, %118 : i64
    %120 = llvm.add %117, %119 : i64
    %121 = llvm.urem %120, %114 : i64
    %122 = llvm.sub %120, %121 : i64
    %123 = llvm.inttoptr %122 : i64 to !llvm.ptr
    %124 = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %125 = llvm.insertvalue %116, %124[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %126 = llvm.insertvalue %123, %125[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %127 = llvm.mlir.constant(0 : index) : i64
    %128 = llvm.insertvalue %127, %126[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %129 = llvm.insertvalue %107, %128[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %130 = llvm.insertvalue %108, %129[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %131 = llvm.insertvalue %108, %130[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %132 = llvm.insertvalue %109, %131[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    llvm.br ^bb16(%26 : i64)
  ^bb16(%133: i64):  // 2 preds: ^bb15, ^bb20
    %134 = llvm.icmp "slt" %133, %25 : i64
    llvm.cond_br %134, ^bb17, ^bb21
  ^bb17:  // pred: ^bb16
    llvm.br ^bb18(%26 : i64)
  ^bb18(%135: i64):  // 2 preds: ^bb17, ^bb19
    %136 = llvm.icmp "slt" %135, %25 : i64
    llvm.cond_br %136, ^bb19, ^bb20
  ^bb19:  // pred: ^bb18
    %137 = llvm.extractvalue %53[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %138 = llvm.mlir.constant(32 : index) : i64
    %139 = llvm.mul %133, %138 overflow<nsw, nuw> : i64
    %140 = llvm.add %139, %135 overflow<nsw, nuw> : i64
    %141 = llvm.getelementptr inbounds|nuw %137[%140] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %142 = llvm.load %141 : !llvm.ptr -> f32
    %143 = llvm.extractvalue %7[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %144 = llvm.extractvalue %7[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %145 = llvm.getelementptr %143[%144] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %146 = llvm.extractvalue %7[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %147 = llvm.mul %133, %146 overflow<nsw, nuw> : i64
    %148 = llvm.extractvalue %7[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %149 = llvm.mul %135, %148 overflow<nsw, nuw> : i64
    %150 = llvm.add %147, %149 overflow<nsw, nuw> : i64
    %151 = llvm.getelementptr inbounds|nuw %145[%150] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %152 = llvm.load %151 : !llvm.ptr -> f32
    %153 = llvm.fadd %142, %152 : f32
    %154 = llvm.intr.maximum(%153, %27) : (f32, f32) -> f32
    %155 = llvm.extractvalue %132[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %156 = llvm.mlir.constant(32 : index) : i64
    %157 = llvm.mul %133, %156 overflow<nsw, nuw> : i64
    %158 = llvm.add %157, %135 overflow<nsw, nuw> : i64
    %159 = llvm.getelementptr inbounds|nuw %155[%158] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %154, %159 : f32, !llvm.ptr
    %160 = llvm.add %135, %24 : i64
    llvm.br ^bb18(%160 : i64)
  ^bb20:  // pred: ^bb18
    %161 = llvm.add %133, %24 : i64
    llvm.br ^bb16(%161 : i64)
  ^bb21:  // pred: ^bb16
    %162 = llvm.extractvalue %53[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    llvm.call @free(%162) : (!llvm.ptr) -> ()
    llvm.return %132 : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
  }
  llvm.func @_mlir_ciface_matmul_bias_relu_32x32x32(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: !llvm.ptr, %arg3: !llvm.ptr) attributes {llvm.emit_c_interface} {
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
    %24 = llvm.call @matmul_bias_relu_32x32x32(%1, %2, %3, %4, %5, %6, %7, %9, %10, %11, %12, %13, %14, %15, %17, %18, %19, %20, %21, %22, %23) : (!llvm.ptr, !llvm.ptr, i64, i64, i64, i64, i64, !llvm.ptr, !llvm.ptr, i64, i64, i64, i64, i64, !llvm.ptr, !llvm.ptr, i64, i64, i64, i64, i64) -> !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    llvm.store %24, %arg0 : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>, !llvm.ptr
    llvm.return
  }
}

