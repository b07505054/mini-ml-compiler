; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"

declare void @free(ptr)

declare ptr @malloc(i64)

define { ptr, ptr, i64, [2 x i64], [2 x i64] } @matmul_bias_relu_tiled_32x32x32(ptr %0, ptr %1, i64 %2, i64 %3, i64 %4, i64 %5, i64 %6, ptr %7, ptr %8, i64 %9, i64 %10, i64 %11, i64 %12, i64 %13, ptr %14, ptr %15, i64 %16, i64 %17, i64 %18, i64 %19, i64 %20) {
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
  %43 = call ptr @malloc(i64 4160)
  %44 = ptrtoint ptr %43 to i64
  %45 = add i64 %44, 63
  %46 = urem i64 %45, 64
  %47 = sub i64 %45, %46
  %48 = inttoptr i64 %47 to ptr
  %49 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } poison, ptr %43, 0
  %50 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %49, ptr %48, 1
  %51 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %50, i64 0, 2
  %52 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %51, i64 32, 3, 0
  %53 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %52, i64 32, 3, 1
  %54 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %53, i64 32, 4, 0
  %55 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %54, i64 1, 4, 1
  %56 = call ptr @malloc(i64 4160)
  %57 = ptrtoint ptr %56 to i64
  %58 = add i64 %57, 63
  %59 = urem i64 %58, 64
  %60 = sub i64 %58, %59
  %61 = inttoptr i64 %60 to ptr
  %62 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } poison, ptr %56, 0
  %63 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %62, ptr %61, 1
  %64 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %63, i64 0, 2
  %65 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %64, i64 32, 3, 0
  %66 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %65, i64 32, 3, 1
  %67 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %66, i64 32, 4, 0
  %68 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %67, i64 1, 4, 1
  br label %69

69:                                               ; preds = %447, %21
  %70 = phi i64 [ %448, %447 ], [ 0, %21 ]
  %71 = icmp slt i64 %70, 32
  br i1 %71, label %72, label %449

72:                                               ; preds = %69
  br label %73

73:                                               ; preds = %343, %72
  %74 = phi i64 [ %446, %343 ], [ 0, %72 ]
  %75 = icmp slt i64 %74, 32
  br i1 %75, label %76, label %447

76:                                               ; preds = %73
  %77 = mul nsw i64 %70, 32
  %78 = add i64 %77, %74
  %79 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %55, 0
  %80 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %55, 1
  %81 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } poison, ptr %79, 0
  %82 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %81, ptr %80, 1
  %83 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %82, i64 %78, 2
  %84 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %83, i64 4, 3, 0
  %85 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %84, i64 32, 4, 0
  %86 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %85, i64 8, 3, 1
  %87 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %86, i64 1, 4, 1
  %88 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %87, 1
  %89 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %87, 2
  %90 = getelementptr float, ptr %88, i64 %89
  %91 = getelementptr float, ptr %90, i64 0
  store <8 x float> zeroinitializer, ptr %91, align 4
  %92 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %87, 1
  %93 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %87, 2
  %94 = getelementptr float, ptr %92, i64 %93
  %95 = getelementptr float, ptr %94, i64 32
  store <8 x float> zeroinitializer, ptr %95, align 4
  %96 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %87, 1
  %97 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %87, 2
  %98 = getelementptr float, ptr %96, i64 %97
  %99 = getelementptr float, ptr %98, i64 64
  store <8 x float> zeroinitializer, ptr %99, align 4
  %100 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %87, 1
  %101 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %87, 2
  %102 = getelementptr float, ptr %100, i64 %101
  %103 = getelementptr float, ptr %102, i64 96
  store <8 x float> zeroinitializer, ptr %103, align 4
  br label %104

104:                                              ; preds = %107, %76
  %105 = phi i64 [ %342, %107 ], [ 0, %76 ]
  %106 = icmp slt i64 %105, 32
  br i1 %106, label %107, label %343

107:                                              ; preds = %104
  %108 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %42, 1
  %109 = mul i64 %70, 32
  %110 = add i64 %109, %105
  %111 = getelementptr float, ptr %108, i64 %110
  %112 = load <8 x float>, ptr %111, align 4
  %113 = add i64 %70, 1
  %114 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %42, 1
  %115 = mul i64 %113, 32
  %116 = add i64 %115, %105
  %117 = getelementptr float, ptr %114, i64 %116
  %118 = load <8 x float>, ptr %117, align 4
  %119 = add i64 %70, 2
  %120 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %42, 1
  %121 = mul i64 %119, 32
  %122 = add i64 %121, %105
  %123 = getelementptr float, ptr %120, i64 %122
  %124 = load <8 x float>, ptr %123, align 4
  %125 = add i64 %70, 3
  %126 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %42, 1
  %127 = mul i64 %125, 32
  %128 = add i64 %127, %105
  %129 = getelementptr float, ptr %126, i64 %128
  %130 = load <8 x float>, ptr %129, align 4
  %131 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %35, 1
  %132 = mul i64 %105, 32
  %133 = add i64 %132, %74
  %134 = getelementptr float, ptr %131, i64 %133
  %135 = load <8 x float>, ptr %134, align 4
  %136 = add i64 %105, 1
  %137 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %35, 1
  %138 = mul i64 %136, 32
  %139 = add i64 %138, %74
  %140 = getelementptr float, ptr %137, i64 %139
  %141 = load <8 x float>, ptr %140, align 4
  %142 = add i64 %105, 2
  %143 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %35, 1
  %144 = mul i64 %142, 32
  %145 = add i64 %144, %74
  %146 = getelementptr float, ptr %143, i64 %145
  %147 = load <8 x float>, ptr %146, align 4
  %148 = add i64 %105, 3
  %149 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %35, 1
  %150 = mul i64 %148, 32
  %151 = add i64 %150, %74
  %152 = getelementptr float, ptr %149, i64 %151
  %153 = load <8 x float>, ptr %152, align 4
  %154 = add i64 %105, 4
  %155 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %35, 1
  %156 = mul i64 %154, 32
  %157 = add i64 %156, %74
  %158 = getelementptr float, ptr %155, i64 %157
  %159 = load <8 x float>, ptr %158, align 4
  %160 = add i64 %105, 5
  %161 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %35, 1
  %162 = mul i64 %160, 32
  %163 = add i64 %162, %74
  %164 = getelementptr float, ptr %161, i64 %163
  %165 = load <8 x float>, ptr %164, align 4
  %166 = add i64 %105, 6
  %167 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %35, 1
  %168 = mul i64 %166, 32
  %169 = add i64 %168, %74
  %170 = getelementptr float, ptr %167, i64 %169
  %171 = load <8 x float>, ptr %170, align 4
  %172 = add i64 %105, 7
  %173 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %35, 1
  %174 = mul i64 %172, 32
  %175 = add i64 %174, %74
  %176 = getelementptr float, ptr %173, i64 %175
  %177 = load <8 x float>, ptr %176, align 4
  %178 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %87, 1
  %179 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %87, 2
  %180 = getelementptr float, ptr %178, i64 %179
  %181 = getelementptr float, ptr %180, i64 0
  %182 = load <8 x float>, ptr %181, align 4
  %183 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %87, 1
  %184 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %87, 2
  %185 = getelementptr float, ptr %183, i64 %184
  %186 = getelementptr float, ptr %185, i64 32
  %187 = load <8 x float>, ptr %186, align 4
  %188 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %87, 1
  %189 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %87, 2
  %190 = getelementptr float, ptr %188, i64 %189
  %191 = getelementptr float, ptr %190, i64 64
  %192 = load <8 x float>, ptr %191, align 4
  %193 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %87, 1
  %194 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %87, 2
  %195 = getelementptr float, ptr %193, i64 %194
  %196 = getelementptr float, ptr %195, i64 96
  %197 = load <8 x float>, ptr %196, align 4
  %198 = extractelement <8 x float> %112, i64 0
  %199 = insertelement <8 x float> poison, float %198, i32 0
  %200 = shufflevector <8 x float> %199, <8 x float> poison, <8 x i32> zeroinitializer
  %201 = call <8 x float> @llvm.fmuladd.v8f32(<8 x float> %200, <8 x float> %135, <8 x float> %182)
  %202 = extractelement <8 x float> %118, i64 0
  %203 = insertelement <8 x float> poison, float %202, i32 0
  %204 = shufflevector <8 x float> %203, <8 x float> poison, <8 x i32> zeroinitializer
  %205 = call <8 x float> @llvm.fmuladd.v8f32(<8 x float> %204, <8 x float> %135, <8 x float> %187)
  %206 = extractelement <8 x float> %124, i64 0
  %207 = insertelement <8 x float> poison, float %206, i32 0
  %208 = shufflevector <8 x float> %207, <8 x float> poison, <8 x i32> zeroinitializer
  %209 = call <8 x float> @llvm.fmuladd.v8f32(<8 x float> %208, <8 x float> %135, <8 x float> %192)
  %210 = extractelement <8 x float> %130, i64 0
  %211 = insertelement <8 x float> poison, float %210, i32 0
  %212 = shufflevector <8 x float> %211, <8 x float> poison, <8 x i32> zeroinitializer
  %213 = call <8 x float> @llvm.fmuladd.v8f32(<8 x float> %212, <8 x float> %135, <8 x float> %197)
  %214 = extractelement <8 x float> %112, i64 1
  %215 = insertelement <8 x float> poison, float %214, i32 0
  %216 = shufflevector <8 x float> %215, <8 x float> poison, <8 x i32> zeroinitializer
  %217 = call <8 x float> @llvm.fmuladd.v8f32(<8 x float> %216, <8 x float> %141, <8 x float> %201)
  %218 = extractelement <8 x float> %118, i64 1
  %219 = insertelement <8 x float> poison, float %218, i32 0
  %220 = shufflevector <8 x float> %219, <8 x float> poison, <8 x i32> zeroinitializer
  %221 = call <8 x float> @llvm.fmuladd.v8f32(<8 x float> %220, <8 x float> %141, <8 x float> %205)
  %222 = extractelement <8 x float> %124, i64 1
  %223 = insertelement <8 x float> poison, float %222, i32 0
  %224 = shufflevector <8 x float> %223, <8 x float> poison, <8 x i32> zeroinitializer
  %225 = call <8 x float> @llvm.fmuladd.v8f32(<8 x float> %224, <8 x float> %141, <8 x float> %209)
  %226 = extractelement <8 x float> %130, i64 1
  %227 = insertelement <8 x float> poison, float %226, i32 0
  %228 = shufflevector <8 x float> %227, <8 x float> poison, <8 x i32> zeroinitializer
  %229 = call <8 x float> @llvm.fmuladd.v8f32(<8 x float> %228, <8 x float> %141, <8 x float> %213)
  %230 = extractelement <8 x float> %112, i64 2
  %231 = insertelement <8 x float> poison, float %230, i32 0
  %232 = shufflevector <8 x float> %231, <8 x float> poison, <8 x i32> zeroinitializer
  %233 = call <8 x float> @llvm.fmuladd.v8f32(<8 x float> %232, <8 x float> %147, <8 x float> %217)
  %234 = extractelement <8 x float> %118, i64 2
  %235 = insertelement <8 x float> poison, float %234, i32 0
  %236 = shufflevector <8 x float> %235, <8 x float> poison, <8 x i32> zeroinitializer
  %237 = call <8 x float> @llvm.fmuladd.v8f32(<8 x float> %236, <8 x float> %147, <8 x float> %221)
  %238 = extractelement <8 x float> %124, i64 2
  %239 = insertelement <8 x float> poison, float %238, i32 0
  %240 = shufflevector <8 x float> %239, <8 x float> poison, <8 x i32> zeroinitializer
  %241 = call <8 x float> @llvm.fmuladd.v8f32(<8 x float> %240, <8 x float> %147, <8 x float> %225)
  %242 = extractelement <8 x float> %130, i64 2
  %243 = insertelement <8 x float> poison, float %242, i32 0
  %244 = shufflevector <8 x float> %243, <8 x float> poison, <8 x i32> zeroinitializer
  %245 = call <8 x float> @llvm.fmuladd.v8f32(<8 x float> %244, <8 x float> %147, <8 x float> %229)
  %246 = extractelement <8 x float> %112, i64 3
  %247 = insertelement <8 x float> poison, float %246, i32 0
  %248 = shufflevector <8 x float> %247, <8 x float> poison, <8 x i32> zeroinitializer
  %249 = call <8 x float> @llvm.fmuladd.v8f32(<8 x float> %248, <8 x float> %153, <8 x float> %233)
  %250 = extractelement <8 x float> %118, i64 3
  %251 = insertelement <8 x float> poison, float %250, i32 0
  %252 = shufflevector <8 x float> %251, <8 x float> poison, <8 x i32> zeroinitializer
  %253 = call <8 x float> @llvm.fmuladd.v8f32(<8 x float> %252, <8 x float> %153, <8 x float> %237)
  %254 = extractelement <8 x float> %124, i64 3
  %255 = insertelement <8 x float> poison, float %254, i32 0
  %256 = shufflevector <8 x float> %255, <8 x float> poison, <8 x i32> zeroinitializer
  %257 = call <8 x float> @llvm.fmuladd.v8f32(<8 x float> %256, <8 x float> %153, <8 x float> %241)
  %258 = extractelement <8 x float> %130, i64 3
  %259 = insertelement <8 x float> poison, float %258, i32 0
  %260 = shufflevector <8 x float> %259, <8 x float> poison, <8 x i32> zeroinitializer
  %261 = call <8 x float> @llvm.fmuladd.v8f32(<8 x float> %260, <8 x float> %153, <8 x float> %245)
  %262 = extractelement <8 x float> %112, i64 4
  %263 = insertelement <8 x float> poison, float %262, i32 0
  %264 = shufflevector <8 x float> %263, <8 x float> poison, <8 x i32> zeroinitializer
  %265 = call <8 x float> @llvm.fmuladd.v8f32(<8 x float> %264, <8 x float> %159, <8 x float> %249)
  %266 = extractelement <8 x float> %118, i64 4
  %267 = insertelement <8 x float> poison, float %266, i32 0
  %268 = shufflevector <8 x float> %267, <8 x float> poison, <8 x i32> zeroinitializer
  %269 = call <8 x float> @llvm.fmuladd.v8f32(<8 x float> %268, <8 x float> %159, <8 x float> %253)
  %270 = extractelement <8 x float> %124, i64 4
  %271 = insertelement <8 x float> poison, float %270, i32 0
  %272 = shufflevector <8 x float> %271, <8 x float> poison, <8 x i32> zeroinitializer
  %273 = call <8 x float> @llvm.fmuladd.v8f32(<8 x float> %272, <8 x float> %159, <8 x float> %257)
  %274 = extractelement <8 x float> %130, i64 4
  %275 = insertelement <8 x float> poison, float %274, i32 0
  %276 = shufflevector <8 x float> %275, <8 x float> poison, <8 x i32> zeroinitializer
  %277 = call <8 x float> @llvm.fmuladd.v8f32(<8 x float> %276, <8 x float> %159, <8 x float> %261)
  %278 = extractelement <8 x float> %112, i64 5
  %279 = insertelement <8 x float> poison, float %278, i32 0
  %280 = shufflevector <8 x float> %279, <8 x float> poison, <8 x i32> zeroinitializer
  %281 = call <8 x float> @llvm.fmuladd.v8f32(<8 x float> %280, <8 x float> %165, <8 x float> %265)
  %282 = extractelement <8 x float> %118, i64 5
  %283 = insertelement <8 x float> poison, float %282, i32 0
  %284 = shufflevector <8 x float> %283, <8 x float> poison, <8 x i32> zeroinitializer
  %285 = call <8 x float> @llvm.fmuladd.v8f32(<8 x float> %284, <8 x float> %165, <8 x float> %269)
  %286 = extractelement <8 x float> %124, i64 5
  %287 = insertelement <8 x float> poison, float %286, i32 0
  %288 = shufflevector <8 x float> %287, <8 x float> poison, <8 x i32> zeroinitializer
  %289 = call <8 x float> @llvm.fmuladd.v8f32(<8 x float> %288, <8 x float> %165, <8 x float> %273)
  %290 = extractelement <8 x float> %130, i64 5
  %291 = insertelement <8 x float> poison, float %290, i32 0
  %292 = shufflevector <8 x float> %291, <8 x float> poison, <8 x i32> zeroinitializer
  %293 = call <8 x float> @llvm.fmuladd.v8f32(<8 x float> %292, <8 x float> %165, <8 x float> %277)
  %294 = extractelement <8 x float> %112, i64 6
  %295 = insertelement <8 x float> poison, float %294, i32 0
  %296 = shufflevector <8 x float> %295, <8 x float> poison, <8 x i32> zeroinitializer
  %297 = call <8 x float> @llvm.fmuladd.v8f32(<8 x float> %296, <8 x float> %171, <8 x float> %281)
  %298 = extractelement <8 x float> %118, i64 6
  %299 = insertelement <8 x float> poison, float %298, i32 0
  %300 = shufflevector <8 x float> %299, <8 x float> poison, <8 x i32> zeroinitializer
  %301 = call <8 x float> @llvm.fmuladd.v8f32(<8 x float> %300, <8 x float> %171, <8 x float> %285)
  %302 = extractelement <8 x float> %124, i64 6
  %303 = insertelement <8 x float> poison, float %302, i32 0
  %304 = shufflevector <8 x float> %303, <8 x float> poison, <8 x i32> zeroinitializer
  %305 = call <8 x float> @llvm.fmuladd.v8f32(<8 x float> %304, <8 x float> %171, <8 x float> %289)
  %306 = extractelement <8 x float> %130, i64 6
  %307 = insertelement <8 x float> poison, float %306, i32 0
  %308 = shufflevector <8 x float> %307, <8 x float> poison, <8 x i32> zeroinitializer
  %309 = call <8 x float> @llvm.fmuladd.v8f32(<8 x float> %308, <8 x float> %171, <8 x float> %293)
  %310 = extractelement <8 x float> %112, i64 7
  %311 = insertelement <8 x float> poison, float %310, i32 0
  %312 = shufflevector <8 x float> %311, <8 x float> poison, <8 x i32> zeroinitializer
  %313 = call <8 x float> @llvm.fmuladd.v8f32(<8 x float> %312, <8 x float> %177, <8 x float> %297)
  %314 = extractelement <8 x float> %118, i64 7
  %315 = insertelement <8 x float> poison, float %314, i32 0
  %316 = shufflevector <8 x float> %315, <8 x float> poison, <8 x i32> zeroinitializer
  %317 = call <8 x float> @llvm.fmuladd.v8f32(<8 x float> %316, <8 x float> %177, <8 x float> %301)
  %318 = extractelement <8 x float> %124, i64 7
  %319 = insertelement <8 x float> poison, float %318, i32 0
  %320 = shufflevector <8 x float> %319, <8 x float> poison, <8 x i32> zeroinitializer
  %321 = call <8 x float> @llvm.fmuladd.v8f32(<8 x float> %320, <8 x float> %177, <8 x float> %305)
  %322 = extractelement <8 x float> %130, i64 7
  %323 = insertelement <8 x float> poison, float %322, i32 0
  %324 = shufflevector <8 x float> %323, <8 x float> poison, <8 x i32> zeroinitializer
  %325 = call <8 x float> @llvm.fmuladd.v8f32(<8 x float> %324, <8 x float> %177, <8 x float> %309)
  %326 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %87, 1
  %327 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %87, 2
  %328 = getelementptr float, ptr %326, i64 %327
  %329 = getelementptr float, ptr %328, i64 0
  store <8 x float> %313, ptr %329, align 4
  %330 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %87, 1
  %331 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %87, 2
  %332 = getelementptr float, ptr %330, i64 %331
  %333 = getelementptr float, ptr %332, i64 32
  store <8 x float> %317, ptr %333, align 4
  %334 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %87, 1
  %335 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %87, 2
  %336 = getelementptr float, ptr %334, i64 %335
  %337 = getelementptr float, ptr %336, i64 64
  store <8 x float> %321, ptr %337, align 4
  %338 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %87, 1
  %339 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %87, 2
  %340 = getelementptr float, ptr %338, i64 %339
  %341 = getelementptr float, ptr %340, i64 96
  store <8 x float> %325, ptr %341, align 4
  %342 = add i64 %105, 8
  br label %104

343:                                              ; preds = %104
  %344 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %87, 1
  %345 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %87, 2
  %346 = getelementptr float, ptr %344, i64 %345
  %347 = getelementptr float, ptr %346, i64 0
  %348 = load <8 x float>, ptr %347, align 4
  %349 = insertvalue [4 x <8 x float>] poison, <8 x float> %348, 0
  %350 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %87, 1
  %351 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %87, 2
  %352 = getelementptr float, ptr %350, i64 %351
  %353 = getelementptr float, ptr %352, i64 32
  %354 = load <8 x float>, ptr %353, align 4
  %355 = insertvalue [4 x <8 x float>] %349, <8 x float> %354, 1
  %356 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %87, 1
  %357 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %87, 2
  %358 = getelementptr float, ptr %356, i64 %357
  %359 = getelementptr float, ptr %358, i64 64
  %360 = load <8 x float>, ptr %359, align 4
  %361 = insertvalue [4 x <8 x float>] %355, <8 x float> %360, 2
  %362 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %87, 1
  %363 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %87, 2
  %364 = getelementptr float, ptr %362, i64 %363
  %365 = getelementptr float, ptr %364, i64 96
  %366 = load <8 x float>, ptr %365, align 4
  %367 = insertvalue [4 x <8 x float>] %361, <8 x float> %366, 3
  %368 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %28, 1
  %369 = mul i64 %70, 32
  %370 = add i64 %369, %74
  %371 = getelementptr float, ptr %368, i64 %370
  %372 = load <8 x float>, ptr %371, align 4
  %373 = insertvalue [4 x <8 x float>] poison, <8 x float> %372, 0
  %374 = add i64 %70, 1
  %375 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %28, 1
  %376 = mul i64 %374, 32
  %377 = add i64 %376, %74
  %378 = getelementptr float, ptr %375, i64 %377
  %379 = load <8 x float>, ptr %378, align 4
  %380 = insertvalue [4 x <8 x float>] %373, <8 x float> %379, 1
  %381 = add i64 %70, 2
  %382 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %28, 1
  %383 = mul i64 %381, 32
  %384 = add i64 %383, %74
  %385 = getelementptr float, ptr %382, i64 %384
  %386 = load <8 x float>, ptr %385, align 4
  %387 = insertvalue [4 x <8 x float>] %380, <8 x float> %386, 2
  %388 = add i64 %70, 3
  %389 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %28, 1
  %390 = mul i64 %388, 32
  %391 = add i64 %390, %74
  %392 = getelementptr float, ptr %389, i64 %391
  %393 = load <8 x float>, ptr %392, align 4
  %394 = insertvalue [4 x <8 x float>] %387, <8 x float> %393, 3
  %395 = extractvalue [4 x <8 x float>] %367, 0
  %396 = extractvalue [4 x <8 x float>] %394, 0
  %397 = fadd <8 x float> %395, %396
  %398 = insertvalue [4 x <8 x float>] poison, <8 x float> %397, 0
  %399 = extractvalue [4 x <8 x float>] %367, 1
  %400 = extractvalue [4 x <8 x float>] %394, 1
  %401 = fadd <8 x float> %399, %400
  %402 = insertvalue [4 x <8 x float>] %398, <8 x float> %401, 1
  %403 = extractvalue [4 x <8 x float>] %367, 2
  %404 = extractvalue [4 x <8 x float>] %394, 2
  %405 = fadd <8 x float> %403, %404
  %406 = insertvalue [4 x <8 x float>] %402, <8 x float> %405, 2
  %407 = extractvalue [4 x <8 x float>] %367, 3
  %408 = extractvalue [4 x <8 x float>] %394, 3
  %409 = fadd <8 x float> %407, %408
  %410 = insertvalue [4 x <8 x float>] %406, <8 x float> %409, 3
  %411 = extractvalue [4 x <8 x float>] %410, 0
  %412 = call <8 x float> @llvm.maximum.v8f32(<8 x float> %411, <8 x float> zeroinitializer)
  %413 = insertvalue [4 x <8 x float>] poison, <8 x float> %412, 0
  %414 = extractvalue [4 x <8 x float>] %410, 1
  %415 = call <8 x float> @llvm.maximum.v8f32(<8 x float> %414, <8 x float> zeroinitializer)
  %416 = insertvalue [4 x <8 x float>] %413, <8 x float> %415, 1
  %417 = extractvalue [4 x <8 x float>] %410, 2
  %418 = call <8 x float> @llvm.maximum.v8f32(<8 x float> %417, <8 x float> zeroinitializer)
  %419 = insertvalue [4 x <8 x float>] %416, <8 x float> %418, 2
  %420 = extractvalue [4 x <8 x float>] %410, 3
  %421 = call <8 x float> @llvm.maximum.v8f32(<8 x float> %420, <8 x float> zeroinitializer)
  %422 = insertvalue [4 x <8 x float>] %419, <8 x float> %421, 3
  %423 = extractvalue [4 x <8 x float>] %422, 0
  %424 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %68, 1
  %425 = mul i64 %70, 32
  %426 = add i64 %425, %74
  %427 = getelementptr float, ptr %424, i64 %426
  store <8 x float> %423, ptr %427, align 4
  %428 = add i64 %70, 1
  %429 = extractvalue [4 x <8 x float>] %422, 1
  %430 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %68, 1
  %431 = mul i64 %428, 32
  %432 = add i64 %431, %74
  %433 = getelementptr float, ptr %430, i64 %432
  store <8 x float> %429, ptr %433, align 4
  %434 = add i64 %70, 2
  %435 = extractvalue [4 x <8 x float>] %422, 2
  %436 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %68, 1
  %437 = mul i64 %434, 32
  %438 = add i64 %437, %74
  %439 = getelementptr float, ptr %436, i64 %438
  store <8 x float> %435, ptr %439, align 4
  %440 = add i64 %70, 3
  %441 = extractvalue [4 x <8 x float>] %422, 3
  %442 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %68, 1
  %443 = mul i64 %440, 32
  %444 = add i64 %443, %74
  %445 = getelementptr float, ptr %442, i64 %444
  store <8 x float> %441, ptr %445, align 4
  %446 = add i64 %74, 8
  br label %73

447:                                              ; preds = %73
  %448 = add i64 %70, 4
  br label %69

449:                                              ; preds = %69
  %450 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %55, 0
  call void @free(ptr %450)
  ret { ptr, ptr, i64, [2 x i64], [2 x i64] } %68
}

define void @_mlir_ciface_matmul_bias_relu_tiled_32x32x32(ptr %0, ptr %1, ptr %2, ptr %3) {
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
  %29 = call { ptr, ptr, i64, [2 x i64], [2 x i64] } @matmul_bias_relu_tiled_32x32x32(ptr %6, ptr %7, i64 %8, i64 %9, i64 %10, i64 %11, i64 %12, ptr %14, ptr %15, i64 %16, i64 %17, i64 %18, i64 %19, i64 %20, ptr %22, ptr %23, i64 %24, i64 %25, i64 %26, i64 %27, i64 %28)
  store { ptr, ptr, i64, [2 x i64], [2 x i64] } %29, ptr %0, align 8
  ret void
}

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare <8 x float> @llvm.maximum.v8f32(<8 x float>, <8 x float>) #0

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare <8 x float> @llvm.fmuladd.v8f32(<8 x float>, <8 x float>, <8 x float>) #0

attributes #0 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
