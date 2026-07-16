; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"

declare void @free(ptr)

declare ptr @malloc(i64)

define { ptr, ptr, i64, [2 x i64], [2 x i64] } @matmul_bias_relu_8x8x8(ptr %0, ptr %1, i64 %2, i64 %3, i64 %4, i64 %5, i64 %6, ptr %7, ptr %8, i64 %9, i64 %10, i64 %11, i64 %12, i64 %13, ptr %14, ptr %15, i64 %16, i64 %17, i64 %18, i64 %19, i64 %20) {
  %22 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } poison, ptr %14, 0
  %23 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %22, ptr %15, 1
  %24 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %23, i64 %16, 2
  %25 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %24, i64 %17, 3, 0
  %26 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %25, i64 %19, 4, 0
  %27 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %26, i64 %18, 3, 1
  %28 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %27, i64 %20, 4, 1
  %29 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } poison, ptr %7, 0
  %30 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %29, ptr %8, 1
  %31 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %30, i64 %9, 2
  %32 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %31, i64 %10, 3, 0
  %33 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %32, i64 %12, 4, 0
  %34 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %33, i64 %11, 3, 1
  %35 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %34, i64 %13, 4, 1
  %36 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } poison, ptr %0, 0
  %37 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %36, ptr %1, 1
  %38 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %37, i64 %2, 2
  %39 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %38, i64 %3, 3, 0
  %40 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %39, i64 %5, 4, 0
  %41 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %40, i64 %4, 3, 1
  %42 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %41, i64 %6, 4, 1
  %43 = call ptr @malloc(i64 320)
  %44 = ptrtoint ptr %43 to i64
  %45 = add i64 %44, 63
  %46 = urem i64 %45, 64
  %47 = sub i64 %45, %46
  %48 = inttoptr i64 %47 to ptr
  %49 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } poison, ptr %43, 0
  %50 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %49, ptr %48, 1
  %51 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %50, i64 0, 2
  %52 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %51, i64 8, 3, 0
  %53 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %52, i64 8, 3, 1
  %54 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %53, i64 8, 4, 0
  %55 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %54, i64 1, 4, 1
  br label %56

56:                                               ; preds = %102, %21
  %57 = phi i64 [ %103, %102 ], [ 0, %21 ]
  %58 = icmp slt i64 %57, 8
  br i1 %58, label %59, label %104

59:                                               ; preds = %56
  br label %60

60:                                               ; preds = %100, %59
  %61 = phi i64 [ %101, %100 ], [ 0, %59 ]
  %62 = icmp slt i64 %61, 8
  br i1 %62, label %63, label %102

63:                                               ; preds = %60
  br label %64

64:                                               ; preds = %67, %63
  %65 = phi i64 [ %99, %67 ], [ 0, %63 ]
  %66 = icmp slt i64 %65, 8
  br i1 %66, label %67, label %100

67:                                               ; preds = %64
  %68 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %42, 1
  %69 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %42, 2
  %70 = getelementptr float, ptr %68, i64 %69
  %71 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %42, 4, 0
  %72 = mul nuw nsw i64 %57, %71
  %73 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %42, 4, 1
  %74 = mul nuw nsw i64 %65, %73
  %75 = add nuw nsw i64 %72, %74
  %76 = getelementptr inbounds nuw float, ptr %70, i64 %75
  %77 = load float, ptr %76, align 4
  %78 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %35, 1
  %79 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %35, 2
  %80 = getelementptr float, ptr %78, i64 %79
  %81 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %35, 4, 0
  %82 = mul nuw nsw i64 %65, %81
  %83 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %35, 4, 1
  %84 = mul nuw nsw i64 %61, %83
  %85 = add nuw nsw i64 %82, %84
  %86 = getelementptr inbounds nuw float, ptr %80, i64 %85
  %87 = load float, ptr %86, align 4
  %88 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %55, 1
  %89 = mul nuw nsw i64 %57, 8
  %90 = add nuw nsw i64 %89, %61
  %91 = getelementptr inbounds nuw float, ptr %88, i64 %90
  %92 = load float, ptr %91, align 4
  %93 = fmul float %77, %87
  %94 = fadd float %92, %93
  %95 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %55, 1
  %96 = mul nuw nsw i64 %57, 8
  %97 = add nuw nsw i64 %96, %61
  %98 = getelementptr inbounds nuw float, ptr %95, i64 %97
  store float %94, ptr %98, align 4
  %99 = add i64 %65, 1
  br label %64

100:                                              ; preds = %64
  %101 = add i64 %61, 1
  br label %60

102:                                              ; preds = %60
  %103 = add i64 %57, 1
  br label %56

104:                                              ; preds = %56
  %105 = call ptr @malloc(i64 320)
  %106 = ptrtoint ptr %105 to i64
  %107 = add i64 %106, 63
  %108 = urem i64 %107, 64
  %109 = sub i64 %107, %108
  %110 = inttoptr i64 %109 to ptr
  %111 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } poison, ptr %105, 0
  %112 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %111, ptr %110, 1
  %113 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %112, i64 0, 2
  %114 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %113, i64 8, 3, 0
  %115 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %114, i64 8, 3, 1
  %116 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %115, i64 8, 4, 0
  %117 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %116, i64 1, 4, 1
  br label %118

118:                                              ; preds = %148, %104
  %119 = phi i64 [ %149, %148 ], [ 0, %104 ]
  %120 = icmp slt i64 %119, 8
  br i1 %120, label %121, label %150

121:                                              ; preds = %118
  br label %122

122:                                              ; preds = %125, %121
  %123 = phi i64 [ %147, %125 ], [ 0, %121 ]
  %124 = icmp slt i64 %123, 8
  br i1 %124, label %125, label %148

125:                                              ; preds = %122
  %126 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %55, 1
  %127 = mul nuw nsw i64 %119, 8
  %128 = add nuw nsw i64 %127, %123
  %129 = getelementptr inbounds nuw float, ptr %126, i64 %128
  %130 = load float, ptr %129, align 4
  %131 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %28, 1
  %132 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %28, 2
  %133 = getelementptr float, ptr %131, i64 %132
  %134 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %28, 4, 0
  %135 = mul nuw nsw i64 %119, %134
  %136 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %28, 4, 1
  %137 = mul nuw nsw i64 %123, %136
  %138 = add nuw nsw i64 %135, %137
  %139 = getelementptr inbounds nuw float, ptr %133, i64 %138
  %140 = load float, ptr %139, align 4
  %141 = fadd float %130, %140
  %142 = call float @llvm.maximum.f32(float %141, float 0.000000e+00)
  %143 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %117, 1
  %144 = mul nuw nsw i64 %119, 8
  %145 = add nuw nsw i64 %144, %123
  %146 = getelementptr inbounds nuw float, ptr %143, i64 %145
  store float %142, ptr %146, align 4
  %147 = add i64 %123, 1
  br label %122

148:                                              ; preds = %122
  %149 = add i64 %119, 1
  br label %118

150:                                              ; preds = %118
  %151 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %55, 0
  call void @free(ptr %151)
  ret { ptr, ptr, i64, [2 x i64], [2 x i64] } %117
}

define void @_mlir_ciface_matmul_bias_relu_8x8x8(ptr %0, ptr %1, ptr %2, ptr %3) {
  %5 = load { ptr, ptr, i64, [2 x i64], [2 x i64] }, ptr %1, align 8
  %6 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %5, 0
  %7 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %5, 1
  %8 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %5, 2
  %9 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %5, 3, 0
  %10 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %5, 3, 1
  %11 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %5, 4, 0
  %12 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %5, 4, 1
  %13 = load { ptr, ptr, i64, [2 x i64], [2 x i64] }, ptr %2, align 8
  %14 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %13, 0
  %15 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %13, 1
  %16 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %13, 2
  %17 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %13, 3, 0
  %18 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %13, 3, 1
  %19 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %13, 4, 0
  %20 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %13, 4, 1
  %21 = load { ptr, ptr, i64, [2 x i64], [2 x i64] }, ptr %3, align 8
  %22 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %21, 0
  %23 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %21, 1
  %24 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %21, 2
  %25 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %21, 3, 0
  %26 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %21, 3, 1
  %27 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %21, 4, 0
  %28 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %21, 4, 1
  %29 = call { ptr, ptr, i64, [2 x i64], [2 x i64] } @matmul_bias_relu_8x8x8(ptr %6, ptr %7, i64 %8, i64 %9, i64 %10, i64 %11, i64 %12, ptr %14, ptr %15, i64 %16, i64 %17, i64 %18, i64 %19, i64 %20, ptr %22, ptr %23, i64 %24, i64 %25, i64 %26, i64 %27, i64 %28)
  store { ptr, ptr, i64, [2 x i64], [2 x i64] } %29, ptr %0, align 8
  ret void
}

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.maximum.f32(float, float) #0

attributes #0 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
