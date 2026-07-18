	.file	"LLVMDialectModule"
	.text
	.globl	matmul_bias_relu_tiled_8x8x32   // -- Begin function matmul_bias_relu_tiled_8x8x32
	.p2align	4
	.type	matmul_bias_relu_tiled_8x8x32,@function
matmul_bias_relu_tiled_8x8x32:          // @matmul_bias_relu_tiled_8x8x32
	.cfi_startproc
// %bb.0:
	str	d10, [sp, #-96]!                // 8-byte Folded Spill
	stp	d9, d8, [sp, #16]               // 16-byte Folded Spill
	stp	x30, x25, [sp, #32]             // 16-byte Folded Spill
	stp	x24, x23, [sp, #48]             // 16-byte Folded Spill
	stp	x22, x21, [sp, #64]             // 16-byte Folded Spill
	stp	x20, x19, [sp, #80]             // 16-byte Folded Spill
	.cfi_def_cfa_offset 96
	.cfi_offset w19, -8
	.cfi_offset w20, -16
	.cfi_offset w21, -24
	.cfi_offset w22, -32
	.cfi_offset w23, -40
	.cfi_offset w24, -48
	.cfi_offset w25, -56
	.cfi_offset w30, -64
	.cfi_offset b8, -72
	.cfi_offset b9, -80
	.cfi_offset b10, -96
	mov	x22, x1
	ldr	x23, [sp, #152]
	ldr	x25, [sp, #96]
	mov	w0, #320                        // =0x140
	bl	malloc
	mov	x19, x0
	add	x8, x0, #63
	and	x24, x8, #0xffffffffffffffc0
	mov	w0, #320                        // =0x140
	bl	malloc
	mov	x20, x0
	mov	x8, xzr
	mov	x9, xzr
	movi	v0.2d, #0000000000000000
	stp	q0, q0, [x24]
	add	x10, x0, #63
	and	x21, x10, #0xffffffffffffffc0
	stp	q0, q0, [x24, #32]
	stp	q0, q0, [x24, #64]
	stp	q0, q0, [x24, #96]
	stp	q0, q0, [x24, #128]
	stp	q0, q0, [x24, #160]
	stp	q0, q0, [x24, #192]
	stp	q0, q0, [x24, #224]
	cmp	x9, #31
	b.gt	.LBB0_2
	.p2align	5, , 16
.LBB0_1:                                // =>This Inner Loop Header: Depth=1
	add	x11, x22, x8
	add	x10, x25, x9, lsl #5
	ldp	q0, q1, [x24]
	ldr	q24, [x11]
	ldp	q31, q9, [x10]
	fmla	v1.4s, v9.4s, v24.s[0]
	fmla	v0.4s, v31.4s, v24.s[0]
	ldp	q2, q3, [x24, #32]
	ldr	q25, [x11, #128]
	fmla	v3.4s, v9.4s, v25.s[0]
	fmla	v2.4s, v31.4s, v25.s[0]
	ldr	q26, [x11, #256]
	ldp	q4, q5, [x24, #64]
	fmla	v5.4s, v9.4s, v26.s[0]
	fmla	v4.4s, v31.4s, v26.s[0]
	ldr	q27, [x11, #384]
	ldp	q6, q7, [x24, #96]
	fmla	v7.4s, v9.4s, v27.s[0]
	fmla	v6.4s, v31.4s, v27.s[0]
	ldr	q28, [x11, #512]
	ldp	q16, q17, [x24, #128]
	fmla	v17.4s, v9.4s, v28.s[0]
	fmla	v16.4s, v31.4s, v28.s[0]
	ldr	q29, [x11, #640]
	ldp	q18, q19, [x24, #160]
	fmla	v19.4s, v9.4s, v29.s[0]
	fmla	v18.4s, v31.4s, v29.s[0]
	ldr	q30, [x11, #768]
	ldp	q20, q21, [x24, #192]
	fmla	v21.4s, v9.4s, v30.s[0]
	fmla	v20.4s, v31.4s, v30.s[0]
	ldr	q8, [x11, #896]
	ldp	q22, q23, [x24, #224]
	fmla	v23.4s, v9.4s, v8.s[0]
	fmla	v22.4s, v31.4s, v8.s[0]
	ldp	q9, q31, [x10, #32]
	fmla	v0.4s, v9.4s, v24.s[1]
	fmla	v1.4s, v31.4s, v24.s[1]
	fmla	v2.4s, v9.4s, v25.s[1]
	fmla	v3.4s, v31.4s, v25.s[1]
	fmla	v4.4s, v9.4s, v26.s[1]
	fmla	v5.4s, v31.4s, v26.s[1]
	fmla	v6.4s, v9.4s, v27.s[1]
	fmla	v7.4s, v31.4s, v27.s[1]
	fmla	v16.4s, v9.4s, v28.s[1]
	fmla	v17.4s, v31.4s, v28.s[1]
	fmla	v18.4s, v9.4s, v29.s[1]
	fmla	v19.4s, v31.4s, v29.s[1]
	fmla	v20.4s, v9.4s, v30.s[1]
	fmla	v21.4s, v31.4s, v30.s[1]
	fmla	v22.4s, v9.4s, v8.s[1]
	fmla	v23.4s, v31.4s, v8.s[1]
	ldp	q31, q9, [x10, #64]
	fmla	v1.4s, v9.4s, v24.s[2]
	fmla	v0.4s, v31.4s, v24.s[2]
	fmla	v3.4s, v9.4s, v25.s[2]
	fmla	v2.4s, v31.4s, v25.s[2]
	fmla	v5.4s, v9.4s, v26.s[2]
	fmla	v4.4s, v31.4s, v26.s[2]
	fmla	v7.4s, v9.4s, v27.s[2]
	fmla	v6.4s, v31.4s, v27.s[2]
	fmla	v17.4s, v9.4s, v28.s[2]
	fmla	v16.4s, v31.4s, v28.s[2]
	fmla	v19.4s, v9.4s, v29.s[2]
	fmla	v18.4s, v31.4s, v29.s[2]
	fmla	v21.4s, v9.4s, v30.s[2]
	fmla	v20.4s, v31.4s, v30.s[2]
	fmla	v23.4s, v9.4s, v8.s[2]
	fmla	v22.4s, v31.4s, v8.s[2]
	ldp	q10, q9, [x10, #96]
	fmla	v0.4s, v10.4s, v24.s[3]
	fmla	v1.4s, v9.4s, v24.s[3]
	ldr	q24, [x11, #16]
	fmla	v2.4s, v10.4s, v25.s[3]
	fmla	v3.4s, v9.4s, v25.s[3]
	ldr	q25, [x11, #144]
	fmla	v4.4s, v10.4s, v26.s[3]
	fmla	v5.4s, v9.4s, v26.s[3]
	ldr	q26, [x11, #272]
	fmla	v6.4s, v10.4s, v27.s[3]
	fmla	v7.4s, v9.4s, v27.s[3]
	ldr	q27, [x11, #400]
	fmla	v16.4s, v10.4s, v28.s[3]
	fmla	v17.4s, v9.4s, v28.s[3]
	ldr	q28, [x11, #528]
	fmla	v18.4s, v10.4s, v29.s[3]
	fmla	v19.4s, v9.4s, v29.s[3]
	ldr	q29, [x11, #656]
	fmla	v20.4s, v10.4s, v30.s[3]
	fmla	v21.4s, v9.4s, v30.s[3]
	ldr	q31, [x11, #784]
	fmla	v22.4s, v10.4s, v8.s[3]
	ldr	q30, [x11, #912]
	fmla	v23.4s, v9.4s, v8.s[3]
	ldp	q8, q9, [x10, #128]
	fmla	v1.4s, v9.4s, v24.s[0]
	fmla	v0.4s, v8.4s, v24.s[0]
	fmla	v3.4s, v9.4s, v25.s[0]
	fmla	v2.4s, v8.4s, v25.s[0]
	fmla	v5.4s, v9.4s, v26.s[0]
	fmla	v4.4s, v8.4s, v26.s[0]
	fmla	v7.4s, v9.4s, v27.s[0]
	fmla	v6.4s, v8.4s, v27.s[0]
	fmla	v17.4s, v9.4s, v28.s[0]
	fmla	v16.4s, v8.4s, v28.s[0]
	fmla	v19.4s, v9.4s, v29.s[0]
	fmla	v18.4s, v8.4s, v29.s[0]
	fmla	v21.4s, v9.4s, v31.s[0]
	fmla	v20.4s, v8.4s, v31.s[0]
	fmla	v23.4s, v9.4s, v30.s[0]
	fmla	v22.4s, v8.4s, v30.s[0]
	ldp	q9, q8, [x10, #160]
	fmla	v0.4s, v9.4s, v24.s[1]
	fmla	v1.4s, v8.4s, v24.s[1]
	fmla	v2.4s, v9.4s, v25.s[1]
	fmla	v3.4s, v8.4s, v25.s[1]
	fmla	v4.4s, v9.4s, v26.s[1]
	fmla	v5.4s, v8.4s, v26.s[1]
	fmla	v6.4s, v9.4s, v27.s[1]
	fmla	v7.4s, v8.4s, v27.s[1]
	fmla	v16.4s, v9.4s, v28.s[1]
	fmla	v17.4s, v8.4s, v28.s[1]
	fmla	v18.4s, v9.4s, v29.s[1]
	fmla	v19.4s, v8.4s, v29.s[1]
	fmla	v20.4s, v9.4s, v31.s[1]
	fmla	v21.4s, v8.4s, v31.s[1]
	fmla	v22.4s, v9.4s, v30.s[1]
	fmla	v23.4s, v8.4s, v30.s[1]
	ldp	q8, q9, [x10, #192]
	fmla	v1.4s, v9.4s, v24.s[2]
	fmla	v0.4s, v8.4s, v24.s[2]
	fmla	v3.4s, v9.4s, v25.s[2]
	fmla	v2.4s, v8.4s, v25.s[2]
	fmla	v5.4s, v9.4s, v26.s[2]
	fmla	v4.4s, v8.4s, v26.s[2]
	fmla	v7.4s, v9.4s, v27.s[2]
	fmla	v6.4s, v8.4s, v27.s[2]
	fmla	v17.4s, v9.4s, v28.s[2]
	fmla	v16.4s, v8.4s, v28.s[2]
	fmla	v19.4s, v9.4s, v29.s[2]
	fmla	v18.4s, v8.4s, v29.s[2]
	fmla	v21.4s, v9.4s, v31.s[2]
	fmla	v20.4s, v8.4s, v31.s[2]
	fmla	v23.4s, v9.4s, v30.s[2]
	fmla	v22.4s, v8.4s, v30.s[2]
	ldp	q9, q8, [x10, #224]
	fmla	v0.4s, v9.4s, v24.s[3]
	fmla	v1.4s, v8.4s, v24.s[3]
	fmla	v2.4s, v9.4s, v25.s[3]
	fmla	v3.4s, v8.4s, v25.s[3]
	fmla	v4.4s, v9.4s, v26.s[3]
	fmla	v5.4s, v8.4s, v26.s[3]
	fmla	v6.4s, v9.4s, v27.s[3]
	fmla	v7.4s, v8.4s, v27.s[3]
	fmla	v16.4s, v9.4s, v28.s[3]
	fmla	v17.4s, v8.4s, v28.s[3]
	fmla	v18.4s, v9.4s, v29.s[3]
	fmla	v19.4s, v8.4s, v29.s[3]
	fmla	v20.4s, v9.4s, v31.s[3]
	fmla	v21.4s, v8.4s, v31.s[3]
	fmla	v22.4s, v9.4s, v30.s[3]
	fmla	v23.4s, v8.4s, v30.s[3]
	stp	q0, q1, [x24]
	stp	q2, q3, [x24, #32]
	stp	q4, q5, [x24, #64]
	stp	q6, q7, [x24, #96]
	stp	q16, q17, [x24, #128]
	stp	q18, q19, [x24, #160]
	stp	q20, q21, [x24, #192]
	stp	q22, q23, [x24, #224]
	add	x9, x9, #8
	add	x8, x8, #32
	cmp	x9, #31
	b.le	.LBB0_1
.LBB0_2:
	ldp	q1, q0, [x24]
	ldp	q3, q2, [x23]
	fadd	v1.4s, v1.4s, v3.4s
	ldp	q4, q3, [x24, #32]
	fadd	v0.4s, v0.4s, v2.4s
	ldr	q2, [x23, #32]
	fadd	v2.4s, v4.4s, v2.4s
	ldp	q4, q6, [x23, #48]
	fadd	v3.4s, v3.4s, v4.4s
	ldp	q5, q4, [x24, #64]
	fadd	v5.4s, v5.4s, v6.4s
	ldp	q6, q16, [x23, #80]
	fadd	v4.4s, v4.4s, v6.4s
	ldp	q7, q6, [x24, #96]
	fadd	v7.4s, v7.4s, v16.4s
	ldp	q16, q18, [x23, #112]
	fadd	v6.4s, v6.4s, v16.4s
	ldp	q17, q16, [x24, #128]
	fadd	v17.4s, v17.4s, v18.4s
	ldp	q18, q20, [x23, #144]
	fadd	v16.4s, v16.4s, v18.4s
	ldp	q19, q18, [x24, #160]
	fadd	v19.4s, v19.4s, v20.4s
	ldp	q20, q22, [x23, #176]
	fadd	v18.4s, v18.4s, v20.4s
	ldp	q21, q20, [x24, #192]
	fadd	v21.4s, v21.4s, v22.4s
	ldp	q22, q24, [x23, #208]
	fadd	v20.4s, v20.4s, v22.4s
	ldp	q23, q22, [x24, #224]
	fadd	v23.4s, v23.4s, v24.4s
	ldr	q24, [x23, #240]
	fadd	v22.4s, v22.4s, v24.4s
	movi	v24.2d, #0000000000000000
	fmax	v0.4s, v0.4s, v24.4s
	fmax	v1.4s, v1.4s, v24.4s
	stp	q1, q0, [x21]
	fmax	v0.4s, v3.4s, v24.4s
	fmax	v1.4s, v2.4s, v24.4s
	stp	q1, q0, [x21, #32]
	fmax	v0.4s, v4.4s, v24.4s
	fmax	v1.4s, v5.4s, v24.4s
	stp	q1, q0, [x21, #64]
	fmax	v0.4s, v6.4s, v24.4s
	fmax	v1.4s, v7.4s, v24.4s
	stp	q1, q0, [x21, #96]
	fmax	v0.4s, v16.4s, v24.4s
	fmax	v1.4s, v17.4s, v24.4s
	stp	q1, q0, [x21, #128]
	fmax	v0.4s, v18.4s, v24.4s
	fmax	v1.4s, v19.4s, v24.4s
	stp	q1, q0, [x21, #160]
	fmax	v0.4s, v20.4s, v24.4s
	fmax	v1.4s, v21.4s, v24.4s
	fmax	v2.4s, v22.4s, v24.4s
	stp	q1, q0, [x21, #192]
	fmax	v0.4s, v23.4s, v24.4s
	stp	q0, q2, [x21, #224]
	mov	x0, x19
	bl	free
	mov	x0, x20
	mov	x1, x21
	mov	x2, xzr
	mov	w3, #8                          // =0x8
	mov	w4, #8                          // =0x8
	mov	w5, #8                          // =0x8
	mov	w6, #1                          // =0x1
	ldp	x20, x19, [sp, #80]             // 16-byte Folded Reload
	ldp	x22, x21, [sp, #64]             // 16-byte Folded Reload
	ldp	x24, x23, [sp, #48]             // 16-byte Folded Reload
	ldp	x30, x25, [sp, #32]             // 16-byte Folded Reload
	ldp	d9, d8, [sp, #16]               // 16-byte Folded Reload
	ldr	d10, [sp], #96                  // 8-byte Folded Reload
	ret
.Lfunc_end0:
	.size	matmul_bias_relu_tiled_8x8x32, .Lfunc_end0-matmul_bias_relu_tiled_8x8x32
	.cfi_endproc
                                        // -- End function
	.globl	_mlir_ciface_matmul_bias_relu_tiled_8x8x32 // -- Begin function _mlir_ciface_matmul_bias_relu_tiled_8x8x32
	.p2align	4
	.type	_mlir_ciface_matmul_bias_relu_tiled_8x8x32,@function
_mlir_ciface_matmul_bias_relu_tiled_8x8x32: // @_mlir_ciface_matmul_bias_relu_tiled_8x8x32
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
	bl	matmul_bias_relu_tiled_8x8x32
	stp	x0, x1, [x19]
	stp	x2, x3, [x19, #16]
	stp	x4, x5, [x19, #32]
	str	x6, [x19, #48]
	ldp	x30, x19, [sp, #112]            // 16-byte Folded Reload
	add	sp, sp, #128
	ret
.Lfunc_end1:
	.size	_mlir_ciface_matmul_bias_relu_tiled_8x8x32, .Lfunc_end1-_mlir_ciface_matmul_bias_relu_tiled_8x8x32
	.cfi_endproc
                                        // -- End function
	.section	".note.GNU-stack","",@progbits
