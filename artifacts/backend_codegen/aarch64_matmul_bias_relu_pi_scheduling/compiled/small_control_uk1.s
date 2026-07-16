	.file	"LLVMDialectModule"
	.text
	.globl	matmul_bias_relu_tiled_8x8x8    // -- Begin function matmul_bias_relu_tiled_8x8x8
	.p2align	4
	.type	matmul_bias_relu_tiled_8x8x8,@function
matmul_bias_relu_tiled_8x8x8:           // @matmul_bias_relu_tiled_8x8x8
	.cfi_startproc
// %bb.0:
	stp	x30, x21, [sp, #-32]!           // 16-byte Folded Spill
	stp	x20, x19, [sp, #16]             // 16-byte Folded Spill
	.cfi_def_cfa_offset 32
	.cfi_offset w19, -8
	.cfi_offset w20, -16
	.cfi_offset w21, -24
	.cfi_offset w30, -32
	mov	x19, x1
	ldr	x20, [sp, #88]
	ldr	x21, [sp, #32]
	mov	w0, #320                        // =0x140
	bl	malloc
	mov	x8, xzr
	add	x9, x0, #63
	and	x1, x9, #0xffffffffffffffc0
	movi	v0.2d, #0000000000000000
	cmp	x8, #7
	b.gt	.LBB0_2
	.p2align	5, , 16
.LBB0_1:                                // =>This Inner Loop Header: Depth=1
	lsl	x9, x8, #5
	add	x10, x20, x9
	ldp	q16, q18, [x10]
	ldp	q5, q6, [x10, #32]
	ldp	q3, q4, [x10, #64]
	ldp	q1, q2, [x10, #96]
	add	x10, x19, x9
	ldp	q20, q21, [x21]
	ldp	q22, q19, [x10]
	fmla	v18.4s, v21.4s, v22.s[0]
	fmla	v16.4s, v20.4s, v22.s[0]
	ldp	q23, q17, [x10, #32]
	fmla	v6.4s, v21.4s, v23.s[0]
	fmla	v5.4s, v20.4s, v23.s[0]
	ldp	q24, q7, [x10, #64]
	fmla	v4.4s, v21.4s, v24.s[0]
	fmla	v3.4s, v20.4s, v24.s[0]
	ldr	q25, [x10, #96]
	fmla	v2.4s, v21.4s, v25.s[0]
	fmla	v1.4s, v20.4s, v25.s[0]
	ldp	q21, q20, [x21, #32]
	fmla	v16.4s, v21.4s, v22.s[1]
	fmla	v18.4s, v20.4s, v22.s[1]
	fmla	v5.4s, v21.4s, v23.s[1]
	fmla	v6.4s, v20.4s, v23.s[1]
	fmla	v3.4s, v21.4s, v24.s[1]
	fmla	v4.4s, v20.4s, v24.s[1]
	fmla	v1.4s, v21.4s, v25.s[1]
	fmla	v2.4s, v20.4s, v25.s[1]
	ldp	q20, q21, [x21, #64]
	fmla	v18.4s, v21.4s, v22.s[2]
	fmla	v16.4s, v20.4s, v22.s[2]
	fmla	v6.4s, v21.4s, v23.s[2]
	fmla	v5.4s, v20.4s, v23.s[2]
	fmla	v4.4s, v21.4s, v24.s[2]
	fmla	v3.4s, v20.4s, v24.s[2]
	fmla	v2.4s, v21.4s, v25.s[2]
	fmla	v1.4s, v20.4s, v25.s[2]
	ldp	q26, q21, [x21, #96]
	fmla	v16.4s, v26.4s, v22.s[3]
	fmla	v18.4s, v21.4s, v22.s[3]
	ldr	q20, [x10, #112]
	fmla	v5.4s, v26.4s, v23.s[3]
	fmla	v6.4s, v21.4s, v23.s[3]
	fmla	v3.4s, v26.4s, v24.s[3]
	fmla	v4.4s, v21.4s, v24.s[3]
	ldp	q22, q23, [x21, #128]
	fmla	v1.4s, v26.4s, v25.s[3]
	fmla	v2.4s, v21.4s, v25.s[3]
	ldp	q24, q21, [x21, #160]
	fmla	v18.4s, v23.4s, v19.s[0]
	fmla	v16.4s, v22.4s, v19.s[0]
	fmla	v6.4s, v23.4s, v17.s[0]
	fmla	v5.4s, v22.4s, v17.s[0]
	fmla	v4.4s, v23.4s, v7.s[0]
	fmla	v3.4s, v22.4s, v7.s[0]
	fmla	v2.4s, v23.4s, v20.s[0]
	fmla	v1.4s, v22.4s, v20.s[0]
	ldp	q22, q23, [x21, #192]
	fmla	v16.4s, v24.4s, v19.s[1]
	fmla	v5.4s, v24.4s, v17.s[1]
	fmla	v3.4s, v24.4s, v7.s[1]
	fmla	v1.4s, v24.4s, v20.s[1]
	ldp	q24, q25, [x21, #224]
	fmla	v18.4s, v21.4s, v19.s[1]
	fmla	v18.4s, v23.4s, v19.s[2]
	fmla	v16.4s, v22.4s, v19.s[2]
	fmla	v16.4s, v24.4s, v19.s[3]
	fmla	v18.4s, v25.4s, v19.s[3]
	fmax	v18.4s, v18.4s, v0.4s
	fmax	v16.4s, v16.4s, v0.4s
	add	x9, x1, x9
	stp	q16, q18, [x9]
	fmla	v6.4s, v21.4s, v17.s[1]
	fmla	v6.4s, v23.4s, v17.s[2]
	fmla	v5.4s, v22.4s, v17.s[2]
	fmla	v6.4s, v25.4s, v17.s[3]
	fmax	v6.4s, v6.4s, v0.4s
	fmla	v5.4s, v24.4s, v17.s[3]
	fmax	v5.4s, v5.4s, v0.4s
	stp	q5, q6, [x9, #32]
	fmla	v4.4s, v21.4s, v7.s[1]
	fmla	v4.4s, v23.4s, v7.s[2]
	fmla	v3.4s, v22.4s, v7.s[2]
	fmla	v4.4s, v25.4s, v7.s[3]
	fmax	v4.4s, v4.4s, v0.4s
	fmla	v3.4s, v24.4s, v7.s[3]
	fmax	v3.4s, v3.4s, v0.4s
	stp	q3, q4, [x9, #64]
	fmla	v2.4s, v21.4s, v20.s[1]
	fmla	v2.4s, v23.4s, v20.s[2]
	fmla	v1.4s, v22.4s, v20.s[2]
	fmla	v2.4s, v25.4s, v20.s[3]
	fmax	v2.4s, v2.4s, v0.4s
	fmla	v1.4s, v24.4s, v20.s[3]
	fmax	v1.4s, v1.4s, v0.4s
	stp	q1, q2, [x9, #96]
	add	x8, x8, #4
	cmp	x8, #7
	b.le	.LBB0_1
.LBB0_2:
	mov	x2, xzr
	mov	w3, #8                          // =0x8
	mov	w4, #8                          // =0x8
	mov	w5, #8                          // =0x8
	mov	w6, #1                          // =0x1
	ldp	x20, x19, [sp, #16]             // 16-byte Folded Reload
	ldp	x30, x21, [sp], #32             // 16-byte Folded Reload
	ret
.Lfunc_end0:
	.size	matmul_bias_relu_tiled_8x8x8, .Lfunc_end0-matmul_bias_relu_tiled_8x8x8
	.cfi_endproc
                                        // -- End function
	.globl	_mlir_ciface_matmul_bias_relu_tiled_8x8x8 // -- Begin function _mlir_ciface_matmul_bias_relu_tiled_8x8x8
	.p2align	4
	.type	_mlir_ciface_matmul_bias_relu_tiled_8x8x8,@function
_mlir_ciface_matmul_bias_relu_tiled_8x8x8: // @_mlir_ciface_matmul_bias_relu_tiled_8x8x8
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
	bl	matmul_bias_relu_tiled_8x8x8
	stp	x0, x1, [x19]
	stp	x2, x3, [x19, #16]
	stp	x4, x5, [x19, #32]
	str	x6, [x19, #48]
	ldp	x30, x19, [sp, #112]            // 16-byte Folded Reload
	add	sp, sp, #128
	ret
.Lfunc_end1:
	.size	_mlir_ciface_matmul_bias_relu_tiled_8x8x8, .Lfunc_end1-_mlir_ciface_matmul_bias_relu_tiled_8x8x8
	.cfi_endproc
                                        // -- End function
	.section	".note.GNU-stack","",@progbits
