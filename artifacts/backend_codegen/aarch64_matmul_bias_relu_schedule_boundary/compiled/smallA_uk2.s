	.file	"LLVMDialectModule"
	.text
	.globl	matmul_bias_relu_tiled_16x16x16 // -- Begin function matmul_bias_relu_tiled_16x16x16
	.p2align	4
	.type	matmul_bias_relu_tiled_16x16x16,@function
matmul_bias_relu_tiled_16x16x16:        // @matmul_bias_relu_tiled_16x16x16
	.cfi_startproc
// %bb.0:
	sub	sp, sp, #176
	stp	d15, d14, [sp, #32]             // 16-byte Folded Spill
	stp	d13, d12, [sp, #48]             // 16-byte Folded Spill
	stp	d11, d10, [sp, #64]             // 16-byte Folded Spill
	stp	d9, d8, [sp, #80]               // 16-byte Folded Spill
	str	x29, [sp, #96]                  // 8-byte Folded Spill
	stp	x30, x25, [sp, #112]            // 16-byte Folded Spill
	stp	x24, x23, [sp, #128]            // 16-byte Folded Spill
	stp	x22, x21, [sp, #144]            // 16-byte Folded Spill
	stp	x20, x19, [sp, #160]            // 16-byte Folded Spill
	.cfi_def_cfa_offset 176
	.cfi_offset w19, -8
	.cfi_offset w20, -16
	.cfi_offset w21, -24
	.cfi_offset w22, -32
	.cfi_offset w23, -40
	.cfi_offset w24, -48
	.cfi_offset w25, -56
	.cfi_offset w30, -64
	.cfi_offset w29, -80
	.cfi_offset b8, -88
	.cfi_offset b9, -96
	.cfi_offset b10, -104
	.cfi_offset b11, -112
	.cfi_offset b12, -120
	.cfi_offset b13, -128
	.cfi_offset b14, -136
	.cfi_offset b15, -144
	mov	x19, x1
	ldr	x23, [sp, #232]
	ldr	x24, [sp, #176]
	mov	w0, #1088                       // =0x440
	bl	malloc
	mov	x20, x0
	add	x8, x0, #63
	and	x25, x8, #0xffffffffffffffc0
	mov	w0, #1088                       // =0x440
	bl	malloc
	mov	x21, x0
	mov	x8, xzr
	mov	x9, xzr
	add	x10, x0, #63
	and	x22, x10, #0xffffffffffffffc0
	movi	v24.2d, #0000000000000000
	b	.LBB0_2
	.p2align	5, , 16
.LBB0_1:                                //   in Loop: Header=BB0_2 Depth=1
	add	x9, x9, #8
	add	x8, x8, #512
.LBB0_2:                                // =>This Loop Header: Depth=1
                                        //     Child Loop BB0_5 Depth 2
                                        //       Child Loop BB0_7 Depth 3
	cmp	x9, #15
	b.gt	.LBB0_8
// %bb.3:                               // %.preheader
                                        //   in Loop: Header=BB0_2 Depth=1
	mov	x10, xzr
	mov	x11, xzr
	b	.LBB0_5
	.p2align	5, , 16
.LBB0_4:                                //   in Loop: Header=BB0_5 Depth=2
	add	x17, x11, x9, lsl #4
	add	x0, x23, x17, lsl #2
	lsl	x1, x17, #2
	add	x18, x1, #64
	add	x16, x1, #128
	add	x15, x1, #192
	add	x14, x1, #256
	add	x13, x1, #320
	ldp	q0, q1, [x0]
	add	x0, x23, x18
	add	x2, x23, x16
	ldp	q2, q3, [x0]
	add	x0, x23, x15
	ldp	q4, q5, [x2]
	add	x2, x23, x14
	ldp	q6, q7, [x0]
	add	x3, x23, x13
	ldp	q16, q17, [x2]
	add	x0, x1, #384
	ldp	q19, q18, [x3]
	add	x2, x23, x0
	ldp	q21, q20, [x2]
	add	x12, x25, x12, lsl #2
	ldp	q22, q23, [x12]
	fadd	v1.4s, v23.4s, v1.4s
	fadd	v0.4s, v22.4s, v0.4s
	ldp	q22, q23, [x12, #64]
	fadd	v3.4s, v23.4s, v3.4s
	fadd	v2.4s, v22.4s, v2.4s
	ldp	q22, q23, [x12, #128]
	fadd	v5.4s, v23.4s, v5.4s
	fadd	v4.4s, v22.4s, v4.4s
	ldp	q22, q23, [x12, #192]
	fadd	v7.4s, v23.4s, v7.4s
	fadd	v6.4s, v22.4s, v6.4s
	ldp	q22, q23, [x12, #256]
	fadd	v17.4s, v23.4s, v17.4s
	fadd	v16.4s, v22.4s, v16.4s
	ldp	q23, q22, [x12, #320]
	fadd	v19.4s, v23.4s, v19.4s
	fadd	v18.4s, v22.4s, v18.4s
	ldp	q23, q22, [x12, #384]
	fadd	v21.4s, v23.4s, v21.4s
	add	x1, x1, #448
	ldp	q24, q23, [x12, #448]
	add	x12, x23, x1
	fadd	v20.4s, v22.4s, v20.4s
	ldr	q22, [x12]
	fadd	v22.4s, v24.4s, v22.4s
	ldr	q24, [x12, #16]
	fadd	v23.4s, v23.4s, v24.4s
	movi	v24.2d, #0000000000000000
	fmax	v0.4s, v0.4s, v24.4s
	fmax	v1.4s, v1.4s, v24.4s
	add	x12, x22, x17, lsl #2
	stp	q0, q1, [x12]
	fmax	v0.4s, v2.4s, v24.4s
	fmax	v1.4s, v3.4s, v24.4s
	add	x12, x22, x18
	stp	q0, q1, [x12]
	fmax	v0.4s, v4.4s, v24.4s
	fmax	v1.4s, v5.4s, v24.4s
	add	x12, x22, x16
	stp	q0, q1, [x12]
	fmax	v0.4s, v6.4s, v24.4s
	fmax	v1.4s, v7.4s, v24.4s
	add	x12, x22, x15
	stp	q0, q1, [x12]
	fmax	v0.4s, v16.4s, v24.4s
	fmax	v1.4s, v17.4s, v24.4s
	add	x12, x22, x14
	stp	q0, q1, [x12]
	fmax	v0.4s, v18.4s, v24.4s
	fmax	v1.4s, v19.4s, v24.4s
	add	x12, x22, x13
	stp	q1, q0, [x12]
	fmax	v0.4s, v20.4s, v24.4s
	fmax	v1.4s, v21.4s, v24.4s
	add	x12, x22, x0
	stp	q1, q0, [x12]
	fmax	v0.4s, v23.4s, v24.4s
	fmax	v1.4s, v22.4s, v24.4s
	add	x12, x22, x1
	stp	q1, q0, [x12]
	add	x11, x11, #8
	add	x10, x10, #32
.LBB0_5:                                //   Parent Loop BB0_2 Depth=1
                                        // =>  This Loop Header: Depth=2
                                        //       Child Loop BB0_7 Depth 3
	cmp	x11, #15
	b.gt	.LBB0_1
// %bb.6:                               //   in Loop: Header=BB0_5 Depth=2
	mov	x13, xzr
	add	x12, x11, x9, lsl #4
	add	x14, x25, x12, lsl #2
	stp	q24, q24, [x14]
	stp	q24, q24, [x14, #64]
	stp	q24, q24, [x14, #128]
	stp	q24, q24, [x14, #192]
	stp	q24, q24, [x14, #256]
	stp	q24, q24, [x14, #320]
	stp	q24, q24, [x14, #384]
	stp	q24, q24, [x14, #448]
	mov	x14, x8
	mov	x15, x10
	cmp	x13, #15
	b.gt	.LBB0_4
	.p2align	5, , 16
.LBB0_7:                                //   Parent Loop BB0_2 Depth=1
                                        //     Parent Loop BB0_5 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	add	x18, x19, x14
	add	x17, x24, x15
	add	x16, x25, x12, lsl #2
	ldp	q10, q25, [x17]
	ldp	q8, q9, [x18]
	ldp	q23, q30, [x16]
	fmla	v30.4s, v25.4s, v8.s[0]
	ldp	q22, q24, [x16, #64]
	ldp	q11, q31, [x18, #64]
	fmla	v24.4s, v25.4s, v11.s[0]
	ldp	q6, q18, [x16, #128]
	ldp	q12, q27, [x18, #128]
	fmla	v18.4s, v25.4s, v12.s[0]
	ldp	q7, q19, [x16, #192]
	ldp	q13, q28, [x18, #192]
	fmla	v19.4s, v25.4s, v13.s[0]
	ldp	q16, q20, [x16, #256]
	ldp	q14, q29, [x18, #256]
	fmla	v20.4s, v25.4s, v14.s[0]
	ldp	q3, q5, [x16, #320]
	ldp	q15, q21, [x18, #320]
	fmla	v5.4s, v25.4s, v15.s[0]
	ldr	q4, [x16, #400]
	ldp	q2, q0, [x18, #384]
	str	q0, [sp]                        // 16-byte Folded Spill
	fmla	v4.4s, v25.4s, v2.s[0]
	ldp	q26, q17, [x16, #448]
	ldp	q1, q0, [x18, #448]
	str	q0, [sp, #16]                   // 16-byte Folded Spill
	fmla	v17.4s, v25.4s, v1.s[0]
	ldr	q25, [x16, #384]
	fmla	v23.4s, v10.4s, v8.s[0]
	fmla	v22.4s, v10.4s, v11.s[0]
	fmla	v6.4s, v10.4s, v12.s[0]
	fmla	v7.4s, v10.4s, v13.s[0]
	fmla	v16.4s, v10.4s, v14.s[0]
	fmla	v3.4s, v10.4s, v15.s[0]
	fmla	v25.4s, v10.4s, v2.s[0]
	fmla	v26.4s, v10.4s, v1.s[0]
	ldp	q0, q10, [x17, #64]
	fmla	v23.4s, v0.4s, v8.s[1]
	fmla	v22.4s, v0.4s, v11.s[1]
	fmla	v6.4s, v0.4s, v12.s[1]
	fmla	v7.4s, v0.4s, v13.s[1]
	fmla	v16.4s, v0.4s, v14.s[1]
	fmla	v3.4s, v0.4s, v15.s[1]
	fmla	v25.4s, v0.4s, v2.s[1]
	fmla	v26.4s, v0.4s, v1.s[1]
	fmla	v30.4s, v10.4s, v8.s[1]
	fmla	v24.4s, v10.4s, v11.s[1]
	fmla	v18.4s, v10.4s, v12.s[1]
	fmla	v19.4s, v10.4s, v13.s[1]
	fmla	v20.4s, v10.4s, v14.s[1]
	fmla	v5.4s, v10.4s, v15.s[1]
	fmla	v4.4s, v10.4s, v2.s[1]
	fmla	v17.4s, v10.4s, v1.s[1]
	ldp	q10, q0, [x17, #128]
	fmla	v30.4s, v0.4s, v8.s[2]
	fmla	v24.4s, v0.4s, v11.s[2]
	fmla	v18.4s, v0.4s, v12.s[2]
	fmla	v19.4s, v0.4s, v13.s[2]
	fmla	v20.4s, v0.4s, v14.s[2]
	fmla	v5.4s, v0.4s, v15.s[2]
	fmla	v4.4s, v0.4s, v2.s[2]
	fmla	v17.4s, v0.4s, v1.s[2]
	fmla	v23.4s, v10.4s, v8.s[2]
	fmla	v22.4s, v10.4s, v11.s[2]
	fmla	v6.4s, v10.4s, v12.s[2]
	fmla	v7.4s, v10.4s, v13.s[2]
	fmla	v16.4s, v10.4s, v14.s[2]
	fmla	v3.4s, v10.4s, v15.s[2]
	fmla	v25.4s, v10.4s, v2.s[2]
	fmla	v26.4s, v10.4s, v1.s[2]
	ldp	q10, q0, [x17, #192]
	fmla	v23.4s, v10.4s, v8.s[3]
	fmla	v30.4s, v0.4s, v8.s[3]
	fmla	v22.4s, v10.4s, v11.s[3]
	fmla	v24.4s, v0.4s, v11.s[3]
	fmla	v6.4s, v10.4s, v12.s[3]
	fmla	v18.4s, v0.4s, v12.s[3]
	fmla	v7.4s, v10.4s, v13.s[3]
	fmla	v19.4s, v0.4s, v13.s[3]
	fmla	v16.4s, v10.4s, v14.s[3]
	fmla	v20.4s, v0.4s, v14.s[3]
	fmla	v3.4s, v10.4s, v15.s[3]
	fmla	v5.4s, v0.4s, v15.s[3]
	fmla	v25.4s, v10.4s, v2.s[3]
	fmla	v4.4s, v0.4s, v2.s[3]
	fmla	v26.4s, v10.4s, v1.s[3]
	fmla	v17.4s, v0.4s, v1.s[3]
	ldp	q0, q1, [x17, #256]
	ldp	q12, q8, [x17, #320]
	ldp	q10, q13, [x17, #384]
	ldp	q14, q11, [x17, #448]
	fmla	v30.4s, v1.4s, v9.s[0]
	fmla	v23.4s, v0.4s, v9.s[0]
	fmla	v24.4s, v1.4s, v31.s[0]
	fmla	v22.4s, v0.4s, v31.s[0]
	fmla	v18.4s, v1.4s, v27.s[0]
	fmla	v6.4s, v0.4s, v27.s[0]
	fmla	v19.4s, v1.4s, v28.s[0]
	fmla	v7.4s, v0.4s, v28.s[0]
	fmla	v20.4s, v1.4s, v29.s[0]
	fmla	v16.4s, v0.4s, v29.s[0]
	fmla	v5.4s, v1.4s, v21.s[0]
	fmla	v3.4s, v0.4s, v21.s[0]
	ldp	q15, q2, [sp]                   // 32-byte Folded Reload
	fmla	v4.4s, v1.4s, v15.s[0]
	fmla	v25.4s, v0.4s, v15.s[0]
	fmla	v17.4s, v1.4s, v2.s[0]
	fmla	v26.4s, v0.4s, v2.s[0]
	fmla	v23.4s, v12.4s, v9.s[1]
	fmla	v30.4s, v8.4s, v9.s[1]
	fmla	v22.4s, v12.4s, v31.s[1]
	fmla	v24.4s, v8.4s, v31.s[1]
	fmla	v6.4s, v12.4s, v27.s[1]
	fmla	v18.4s, v8.4s, v27.s[1]
	fmla	v7.4s, v12.4s, v28.s[1]
	fmla	v19.4s, v8.4s, v28.s[1]
	fmla	v16.4s, v12.4s, v29.s[1]
	fmla	v20.4s, v8.4s, v29.s[1]
	fmla	v3.4s, v12.4s, v21.s[1]
	fmla	v5.4s, v8.4s, v21.s[1]
	fmla	v25.4s, v12.4s, v15.s[1]
	fmla	v30.4s, v13.4s, v9.s[2]
	fmla	v23.4s, v10.4s, v9.s[2]
	fmla	v24.4s, v13.4s, v31.s[2]
	fmla	v23.4s, v14.4s, v9.s[3]
	fmla	v30.4s, v11.4s, v9.s[3]
	stp	q23, q30, [x16]
	fmla	v22.4s, v10.4s, v31.s[2]
	fmla	v22.4s, v14.4s, v31.s[3]
	fmla	v24.4s, v11.4s, v31.s[3]
	stp	q22, q24, [x16, #64]
	fmla	v18.4s, v13.4s, v27.s[2]
	fmla	v6.4s, v10.4s, v27.s[2]
	fmla	v19.4s, v13.4s, v28.s[2]
	fmla	v7.4s, v10.4s, v28.s[2]
	fmla	v6.4s, v14.4s, v27.s[3]
	fmla	v18.4s, v11.4s, v27.s[3]
	stp	q6, q18, [x16, #128]
	fmla	v20.4s, v13.4s, v29.s[2]
	fmla	v7.4s, v14.4s, v28.s[3]
	fmla	v19.4s, v11.4s, v28.s[3]
	stp	q7, q19, [x16, #192]
	fmla	v16.4s, v10.4s, v29.s[2]
	fmla	v16.4s, v14.4s, v29.s[3]
	fmla	v20.4s, v11.4s, v29.s[3]
	stp	q16, q20, [x16, #256]
	fmla	v4.4s, v8.4s, v15.s[1]
	fmla	v5.4s, v13.4s, v21.s[2]
	fmla	v3.4s, v10.4s, v21.s[2]
	fmla	v4.4s, v13.4s, v15.s[2]
	fmla	v3.4s, v14.4s, v21.s[3]
	fmla	v5.4s, v11.4s, v21.s[3]
	stp	q3, q5, [x16, #320]
	fmla	v25.4s, v10.4s, v15.s[2]
	fmla	v25.4s, v14.4s, v15.s[3]
	fmla	v4.4s, v11.4s, v15.s[3]
	stp	q25, q4, [x16, #384]
	fmla	v26.4s, v12.4s, v2.s[1]
	fmla	v17.4s, v8.4s, v2.s[1]
	fmla	v17.4s, v13.4s, v2.s[2]
	fmla	v26.4s, v10.4s, v2.s[2]
	fmla	v26.4s, v14.4s, v2.s[3]
	fmla	v17.4s, v11.4s, v2.s[3]
	stp	q26, q17, [x16, #448]
	add	x13, x13, #8
	add	x15, x15, #512
	add	x14, x14, #32
	cmp	x13, #15
	b.le	.LBB0_7
	b	.LBB0_4
.LBB0_8:
	mov	x0, x20
	bl	free
	mov	x0, x21
	mov	x1, x22
	mov	x2, xzr
	mov	w3, #16                         // =0x10
	mov	w4, #16                         // =0x10
	mov	w5, #16                         // =0x10
	mov	w6, #1                          // =0x1
	ldp	x20, x19, [sp, #160]            // 16-byte Folded Reload
	ldp	x22, x21, [sp, #144]            // 16-byte Folded Reload
	ldp	x24, x23, [sp, #128]            // 16-byte Folded Reload
	ldp	x30, x25, [sp, #112]            // 16-byte Folded Reload
	ldr	x29, [sp, #96]                  // 8-byte Folded Reload
	ldp	d9, d8, [sp, #80]               // 16-byte Folded Reload
	ldp	d11, d10, [sp, #64]             // 16-byte Folded Reload
	ldp	d13, d12, [sp, #48]             // 16-byte Folded Reload
	ldp	d15, d14, [sp, #32]             // 16-byte Folded Reload
	add	sp, sp, #176
	ret
.Lfunc_end0:
	.size	matmul_bias_relu_tiled_16x16x16, .Lfunc_end0-matmul_bias_relu_tiled_16x16x16
	.cfi_endproc
                                        // -- End function
	.globl	_mlir_ciface_matmul_bias_relu_tiled_16x16x16 // -- Begin function _mlir_ciface_matmul_bias_relu_tiled_16x16x16
	.p2align	4
	.type	_mlir_ciface_matmul_bias_relu_tiled_16x16x16,@function
_mlir_ciface_matmul_bias_relu_tiled_16x16x16: // @_mlir_ciface_matmul_bias_relu_tiled_16x16x16
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
	bl	matmul_bias_relu_tiled_16x16x16
	stp	x0, x1, [x19]
	stp	x2, x3, [x19, #16]
	stp	x4, x5, [x19, #32]
	str	x6, [x19, #48]
	ldp	x30, x19, [sp, #112]            // 16-byte Folded Reload
	add	sp, sp, #128
	ret
.Lfunc_end1:
	.size	_mlir_ciface_matmul_bias_relu_tiled_16x16x16, .Lfunc_end1-_mlir_ciface_matmul_bias_relu_tiled_16x16x16
	.cfi_endproc
                                        // -- End function
	.section	".note.GNU-stack","",@progbits
