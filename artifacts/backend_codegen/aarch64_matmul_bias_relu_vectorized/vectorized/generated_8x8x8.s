	.file	"LLVMDialectModule"
	.text
	.globl	matmul_bias_relu_vectorized_8x8x8 // -- Begin function matmul_bias_relu_vectorized_8x8x8
	.p2align	4
	.type	matmul_bias_relu_vectorized_8x8x8,@function
matmul_bias_relu_vectorized_8x8x8:      // @matmul_bias_relu_vectorized_8x8x8
	.cfi_startproc
// %bb.0:
	stp	d9, d8, [sp, #-48]!             // 16-byte Folded Spill
	str	x29, [sp, #16]                  // 8-byte Folded Spill
	stp	x30, x19, [sp, #32]             // 16-byte Folded Spill
	sub	sp, sp, #512
	.cfi_def_cfa_offset 560
	.cfi_offset w19, -8
	.cfi_offset w30, -16
	.cfi_offset w29, -32
	.cfi_offset b8, -40
	.cfi_offset b9, -48
	ldr	x19, [sp, #616]
	ldr	x8, [sp, #560]
	ldp	q2, q1, [x1, #224]
	str	q1, [sp, #288]                  // 16-byte Folded Spill
	ldp	q0, q1, [x1, #192]
	stp	q0, q2, [sp, #144]              // 32-byte Folded Spill
	ldp	q3, q2, [x1, #160]
	stp	q1, q2, [sp, #400]              // 32-byte Folded Spill
	ldp	q0, q1, [x1, #128]
	stp	q0, q3, [sp, #96]               // 32-byte Folded Spill
	ldp	q3, q2, [x1, #96]
	stp	q1, q2, [sp, #432]              // 32-byte Folded Spill
	ldp	q0, q1, [x1, #64]
	stp	q0, q3, [sp, #48]               // 32-byte Folded Spill
	ldp	q3, q2, [x1, #32]
	stp	q1, q2, [sp, #464]              // 32-byte Folded Spill
	ldp	q0, q1, [x1]
	str	q1, [sp, #496]                  // 16-byte Folded Spill
	stp	q0, q3, [sp, #16]               // 32-byte Folded Spill
	ldp	q0, q1, [x8, #224]
	stp	q0, q1, [sp, #368]              // 32-byte Folded Spill
	ldp	q1, q0, [x8, #192]
	stp	q0, q1, [sp, #336]              // 32-byte Folded Spill
	ldp	q0, q1, [x8, #160]
	stp	q0, q1, [sp, #304]              // 32-byte Folded Spill
	ldp	q1, q0, [x8, #128]
	stp	q0, q1, [sp, #256]              // 32-byte Folded Spill
	ldp	q0, q1, [x8, #96]
	stp	q0, q1, [sp, #224]              // 32-byte Folded Spill
	ldp	q1, q0, [x8, #64]
	stp	q0, q1, [sp, #192]              // 32-byte Folded Spill
	ldp	q0, q1, [x8, #32]
	str	q1, [sp, #176]                  // 16-byte Folded Spill
	str	q0, [sp, #128]                  // 16-byte Folded Spill
	ldp	q1, q0, [x8]
	str	q1, [sp, #80]                   // 16-byte Folded Spill
	str	q0, [sp]                        // 16-byte Folded Spill
	mov	w0, #320                        // =0x140
	bl	malloc
	ldr	q0, [x19, #16]
	ldp	q3, q9, [sp]                    // 32-byte Folded Reload
	fmla	v0.4s, v3.4s, v9.s[0]
	ldr	q1, [x19, #48]
	ldp	q8, q31, [sp, #32]              // 32-byte Folded Reload
	fmla	v1.4s, v3.4s, v8.s[0]
	ldr	q2, [x19, #80]
	fmla	v2.4s, v3.4s, v31.s[0]
	ldr	q4, [x19, #112]
	ldp	q29, q28, [sp, #64]             // 32-byte Folded Reload
	fmla	v4.4s, v3.4s, v29.s[0]
	ldp	q18, q7, [x19, #128]
	ldp	q27, q26, [sp, #96]             // 32-byte Folded Reload
	fmla	v7.4s, v3.4s, v27.s[0]
	ldp	q20, q17, [x19, #160]
	fmla	v17.4s, v3.4s, v26.s[0]
	ldp	q22, q19, [x19, #192]
	ldp	q25, q24, [sp, #144]            // 32-byte Folded Reload
	fmla	v19.4s, v3.4s, v25.s[0]
	ldp	q23, q21, [x19, #224]
	fmla	v21.4s, v3.4s, v24.s[0]
	ldr	q3, [x19]
	fmla	v3.4s, v28.4s, v9.s[0]
	ldr	q5, [x19, #32]
	fmla	v5.4s, v28.4s, v8.s[0]
	ldr	q6, [x19, #64]
	fmla	v6.4s, v28.4s, v31.s[0]
	ldr	q16, [x19, #96]
	fmla	v16.4s, v28.4s, v29.s[0]
	fmla	v18.4s, v28.4s, v27.s[0]
	fmla	v20.4s, v28.4s, v26.s[0]
	fmla	v22.4s, v28.4s, v25.s[0]
	fmla	v23.4s, v28.4s, v24.s[0]
	ldr	q28, [sp, #128]                 // 16-byte Folded Reload
	fmla	v3.4s, v28.4s, v9.s[1]
	fmla	v5.4s, v28.4s, v8.s[1]
	fmla	v6.4s, v28.4s, v31.s[1]
	fmla	v16.4s, v28.4s, v29.s[1]
	fmla	v18.4s, v28.4s, v27.s[1]
	fmla	v20.4s, v28.4s, v26.s[1]
	fmla	v22.4s, v28.4s, v25.s[1]
	fmla	v23.4s, v28.4s, v24.s[1]
	ldp	q30, q28, [sp, #176]            // 32-byte Folded Reload
	fmla	v0.4s, v30.4s, v9.s[1]
	fmla	v1.4s, v30.4s, v8.s[1]
	fmla	v2.4s, v30.4s, v31.s[1]
	fmla	v4.4s, v30.4s, v29.s[1]
	fmla	v7.4s, v30.4s, v27.s[1]
	fmla	v17.4s, v30.4s, v26.s[1]
	fmla	v19.4s, v30.4s, v25.s[1]
	fmla	v21.4s, v30.4s, v24.s[1]
	fmla	v0.4s, v28.4s, v9.s[2]
	fmla	v1.4s, v28.4s, v8.s[2]
	fmla	v2.4s, v28.4s, v31.s[2]
	fmla	v4.4s, v28.4s, v29.s[2]
	fmla	v7.4s, v28.4s, v27.s[2]
	fmla	v17.4s, v28.4s, v26.s[2]
	fmla	v19.4s, v28.4s, v25.s[2]
	fmla	v21.4s, v28.4s, v24.s[2]
	ldp	q28, q30, [sp, #208]            // 32-byte Folded Reload
	fmla	v3.4s, v28.4s, v9.s[2]
	fmla	v5.4s, v28.4s, v8.s[2]
	fmla	v6.4s, v28.4s, v31.s[2]
	fmla	v16.4s, v28.4s, v29.s[2]
	fmla	v18.4s, v28.4s, v27.s[2]
	fmla	v20.4s, v28.4s, v26.s[2]
	fmla	v22.4s, v28.4s, v25.s[2]
	fmla	v23.4s, v28.4s, v24.s[2]
	fmla	v3.4s, v30.4s, v9.s[3]
	ldr	q28, [sp, #240]                 // 16-byte Folded Reload
	fmla	v0.4s, v28.4s, v9.s[3]
	fmla	v5.4s, v30.4s, v8.s[3]
	fmla	v1.4s, v28.4s, v8.s[3]
	fmla	v6.4s, v30.4s, v31.s[3]
	fmla	v2.4s, v28.4s, v31.s[3]
	fmla	v16.4s, v30.4s, v29.s[3]
	fmla	v4.4s, v28.4s, v29.s[3]
	fmla	v18.4s, v30.4s, v27.s[3]
	fmla	v7.4s, v28.4s, v27.s[3]
	fmla	v20.4s, v30.4s, v26.s[3]
	fmla	v17.4s, v28.4s, v26.s[3]
	fmla	v22.4s, v30.4s, v25.s[3]
	fmla	v19.4s, v28.4s, v25.s[3]
	fmla	v23.4s, v30.4s, v24.s[3]
	fmla	v21.4s, v28.4s, v24.s[3]
	add	x8, x0, #63
	ldp	q25, q24, [sp, #272]            // 32-byte Folded Reload
	ldr	q26, [sp, #256]                 // 16-byte Folded Reload
	fmla	v21.4s, v26.4s, v24.s[0]
	fmla	v23.4s, v25.4s, v24.s[0]
	ldp	q27, q31, [sp, #304]            // 32-byte Folded Reload
	fmla	v23.4s, v27.4s, v24.s[1]
	fmla	v21.4s, v31.4s, v24.s[1]
	ldp	q29, q28, [sp, #336]            // 32-byte Folded Reload
	fmla	v21.4s, v29.4s, v24.s[2]
	fmla	v23.4s, v28.4s, v24.s[2]
	ldp	q8, q30, [sp, #368]             // 32-byte Folded Reload
	fmla	v23.4s, v8.4s, v24.s[3]
	fmla	v21.4s, v30.4s, v24.s[3]
	movi	v24.2d, #0000000000000000
	fmax	v21.4s, v21.4s, v24.4s
	and	x1, x8, #0xffffffffffffffc0
	fmax	v23.4s, v23.4s, v24.4s
	stp	q23, q21, [x1, #224]
	ldr	q21, [sp, #400]                 // 16-byte Folded Reload
	fmla	v19.4s, v26.4s, v21.s[0]
	fmla	v22.4s, v25.4s, v21.s[0]
	fmla	v22.4s, v27.4s, v21.s[1]
	fmla	v19.4s, v31.4s, v21.s[1]
	fmla	v19.4s, v29.4s, v21.s[2]
	fmla	v22.4s, v28.4s, v21.s[2]
	fmla	v19.4s, v30.4s, v21.s[3]
	fmax	v19.4s, v19.4s, v24.4s
	fmla	v22.4s, v8.4s, v21.s[3]
	fmax	v21.4s, v22.4s, v24.4s
	stp	q21, q19, [x1, #192]
	ldr	q19, [sp, #416]                 // 16-byte Folded Reload
	fmla	v17.4s, v26.4s, v19.s[0]
	fmla	v20.4s, v25.4s, v19.s[0]
	fmla	v20.4s, v27.4s, v19.s[1]
	mov	v22.16b, v27.16b
	fmla	v17.4s, v31.4s, v19.s[1]
	fmla	v17.4s, v29.4s, v19.s[2]
	fmla	v20.4s, v28.4s, v19.s[2]
	fmla	v17.4s, v30.4s, v19.s[3]
	fmax	v17.4s, v17.4s, v24.4s
	fmla	v20.4s, v8.4s, v19.s[3]
	fmax	v19.4s, v20.4s, v24.4s
	stp	q19, q17, [x1, #160]
	ldr	q17, [sp, #432]                 // 16-byte Folded Reload
	fmla	v7.4s, v26.4s, v17.s[0]
	fmla	v18.4s, v25.4s, v17.s[0]
	fmla	v18.4s, v27.4s, v17.s[1]
	fmla	v7.4s, v31.4s, v17.s[1]
	fmla	v7.4s, v29.4s, v17.s[2]
	fmla	v18.4s, v28.4s, v17.s[2]
	fmla	v7.4s, v30.4s, v17.s[3]
	fmax	v7.4s, v7.4s, v24.4s
	fmla	v18.4s, v8.4s, v17.s[3]
	fmax	v17.4s, v18.4s, v24.4s
	stp	q17, q7, [x1, #128]
	ldr	q7, [sp, #448]                  // 16-byte Folded Reload
	fmla	v4.4s, v26.4s, v7.s[0]
	fmla	v16.4s, v25.4s, v7.s[0]
	fmla	v16.4s, v27.4s, v7.s[1]
	fmla	v4.4s, v31.4s, v7.s[1]
	fmla	v4.4s, v29.4s, v7.s[2]
	fmla	v16.4s, v28.4s, v7.s[2]
	fmla	v4.4s, v30.4s, v7.s[3]
	fmax	v4.4s, v4.4s, v24.4s
	fmla	v16.4s, v8.4s, v7.s[3]
	fmax	v7.4s, v16.4s, v24.4s
	stp	q7, q4, [x1, #96]
	ldr	q4, [sp, #464]                  // 16-byte Folded Reload
	fmla	v2.4s, v26.4s, v4.s[0]
	fmla	v6.4s, v25.4s, v4.s[0]
	fmla	v6.4s, v27.4s, v4.s[1]
	fmla	v2.4s, v31.4s, v4.s[1]
	fmla	v2.4s, v29.4s, v4.s[2]
	fmla	v6.4s, v28.4s, v4.s[2]
	fmla	v2.4s, v30.4s, v4.s[3]
	fmax	v2.4s, v2.4s, v24.4s
	fmla	v6.4s, v8.4s, v4.s[3]
	fmax	v4.4s, v6.4s, v24.4s
	stp	q4, q2, [x1, #64]
	ldr	q2, [sp, #480]                  // 16-byte Folded Reload
	fmla	v1.4s, v26.4s, v2.s[0]
	fmla	v5.4s, v25.4s, v2.s[0]
	fmla	v5.4s, v22.4s, v2.s[1]
	fmla	v1.4s, v31.4s, v2.s[1]
	fmla	v1.4s, v29.4s, v2.s[2]
	fmla	v5.4s, v28.4s, v2.s[2]
	fmla	v1.4s, v30.4s, v2.s[3]
	fmax	v1.4s, v1.4s, v24.4s
	fmla	v5.4s, v8.4s, v2.s[3]
	fmax	v2.4s, v5.4s, v24.4s
	stp	q2, q1, [x1, #32]
	ldr	q1, [sp, #496]                  // 16-byte Folded Reload
	fmla	v0.4s, v26.4s, v1.s[0]
	fmla	v3.4s, v25.4s, v1.s[0]
	fmla	v3.4s, v22.4s, v1.s[1]
	fmla	v0.4s, v31.4s, v1.s[1]
	fmla	v0.4s, v29.4s, v1.s[2]
	fmla	v3.4s, v28.4s, v1.s[2]
	fmla	v0.4s, v30.4s, v1.s[3]
	fmax	v0.4s, v0.4s, v24.4s
	fmla	v3.4s, v8.4s, v1.s[3]
	fmax	v1.4s, v3.4s, v24.4s
	stp	q1, q0, [x1]
	mov	x2, xzr
	mov	w3, #8                          // =0x8
	mov	w4, #8                          // =0x8
	mov	w5, #8                          // =0x8
	mov	w6, #1                          // =0x1
	add	sp, sp, #512
	ldp	x30, x19, [sp, #32]             // 16-byte Folded Reload
	ldr	x29, [sp, #16]                  // 8-byte Folded Reload
	ldp	d9, d8, [sp], #48               // 16-byte Folded Reload
	ret
.Lfunc_end0:
	.size	matmul_bias_relu_vectorized_8x8x8, .Lfunc_end0-matmul_bias_relu_vectorized_8x8x8
	.cfi_endproc
                                        // -- End function
	.globl	_mlir_ciface_matmul_bias_relu_vectorized_8x8x8 // -- Begin function _mlir_ciface_matmul_bias_relu_vectorized_8x8x8
	.p2align	4
	.type	_mlir_ciface_matmul_bias_relu_vectorized_8x8x8,@function
_mlir_ciface_matmul_bias_relu_vectorized_8x8x8: // @_mlir_ciface_matmul_bias_relu_vectorized_8x8x8
	.cfi_startproc
// %bb.0:
	sub	sp, sp, #128
	stp	x30, x19, [sp, #112]            // 16-byte Folded Spill
	.cfi_def_cfa_offset 128
	.cfi_offset w19, -8
	.cfi_offset w30, -16
	mov	x19, x0
	ldp	x5, x6, [x1, #40]
	ldp	x8, x4, [x1, #24]
	ldp	x10, x9, [x1, #8]
	ldr	x0, [x1]
	ldr	x7, [x2]
	ldur	q0, [x2, #8]
	ldur	q1, [x2, #24]
	ldp	x11, x12, [x2, #40]
	ldp	q2, q3, [x3]
	ldr	q4, [x3, #32]
	ldr	x13, [x3, #48]
	stp	q3, q4, [sp, #64]
	str	q2, [sp, #48]
	stp	q0, q1, [sp]
	str	x13, [sp, #96]
	stp	x11, x12, [sp, #32]
	mov	x1, x10
	mov	x2, x9
	mov	x3, x8
	bl	matmul_bias_relu_vectorized_8x8x8
	stp	x0, x1, [x19]
	stp	x2, x3, [x19, #16]
	stp	x4, x5, [x19, #32]
	str	x6, [x19, #48]
	ldp	x30, x19, [sp, #112]            // 16-byte Folded Reload
	add	sp, sp, #128
	ret
.Lfunc_end1:
	.size	_mlir_ciface_matmul_bias_relu_vectorized_8x8x8, .Lfunc_end1-_mlir_ciface_matmul_bias_relu_vectorized_8x8x8
	.cfi_endproc
                                        // -- End function
	.section	".note.GNU-stack","",@progbits
