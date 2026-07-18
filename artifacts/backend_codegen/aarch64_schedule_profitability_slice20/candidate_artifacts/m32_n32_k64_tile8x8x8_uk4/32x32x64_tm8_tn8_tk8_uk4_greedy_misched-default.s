	.file	"LLVMDialectModule"
	.text
	.globl	matmul_bias_relu_tiled_32x32x64 // -- Begin function matmul_bias_relu_tiled_32x32x64
	.p2align	4
	.type	matmul_bias_relu_tiled_32x32x64,@function
matmul_bias_relu_tiled_32x32x64:        // @matmul_bias_relu_tiled_32x32x64
	.cfi_startproc
// %bb.0:
	stp	d11, d10, [sp, #-96]!           // 16-byte Folded Spill
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
	.cfi_offset b10, -88
	.cfi_offset b11, -96
	mov	x19, x1
	ldr	x23, [sp, #152]
	ldr	x24, [sp, #96]
	mov	w0, #4160                       // =0x1040
	bl	malloc
	mov	x20, x0
	add	x8, x0, #63
	and	x25, x8, #0xffffffffffffffc0
	mov	w0, #4160                       // =0x1040
	bl	malloc
	mov	x21, x0
	mov	x8, xzr
	mov	x9, xzr
	add	x10, x0, #63
	and	x22, x10, #0xffffffffffffffc0
	movi	v0.2d, #0000000000000000
	b	.LBB0_2
	.p2align	5, , 16
.LBB0_1:                                //   in Loop: Header=BB0_2 Depth=1
	add	x9, x9, #8
	add	x8, x8, #2048
.LBB0_2:                                // =>This Loop Header: Depth=1
                                        //     Child Loop BB0_5 Depth 2
                                        //       Child Loop BB0_7 Depth 3
	cmp	x9, #31
	b.gt	.LBB0_8
// %bb.3:                               // %.preheader
                                        //   in Loop: Header=BB0_2 Depth=1
	mov	x10, xzr
	mov	x11, xzr
	add	x12, x19, x8
	b	.LBB0_5
	.p2align	5, , 16
.LBB0_4:                                //   in Loop: Header=BB0_5 Depth=2
	add	x17, x23, x13, lsl #2
	lsl	x1, x13, #2
	add	x16, x1, #128
	add	x15, x1, #256
	add	x14, x1, #384
	ldp	q1, q2, [x17]
	add	x17, x23, x16
	add	x18, x23, x15
	add	x0, x23, x14
	ldp	q3, q4, [x17]
	add	x17, x1, #512
	add	x2, x23, x17
	ldp	q5, q6, [x18]
	add	x18, x1, #640
	ldp	q7, q16, [x0]
	add	x3, x23, x18
	ldp	q17, q18, [x2]
	add	x0, x1, #768
	ldp	q20, q19, [x3]
	add	x2, x23, x0
	ldp	q22, q21, [x2]
	add	x2, x25, x13, lsl #2
	ldp	q23, q24, [x2]
	fadd	v2.4s, v24.4s, v2.4s
	fadd	v1.4s, v23.4s, v1.4s
	ldp	q23, q24, [x2, #128]
	fadd	v4.4s, v24.4s, v4.4s
	fadd	v3.4s, v23.4s, v3.4s
	ldp	q23, q24, [x2, #256]
	fadd	v6.4s, v24.4s, v6.4s
	fadd	v5.4s, v23.4s, v5.4s
	ldp	q23, q24, [x2, #384]
	fadd	v16.4s, v24.4s, v16.4s
	fadd	v7.4s, v23.4s, v7.4s
	ldp	q23, q24, [x2, #512]
	fadd	v18.4s, v24.4s, v18.4s
	fadd	v17.4s, v23.4s, v17.4s
	ldp	q24, q23, [x2, #640]
	fadd	v20.4s, v24.4s, v20.4s
	fadd	v19.4s, v23.4s, v19.4s
	ldp	q24, q23, [x2, #768]
	fadd	v22.4s, v24.4s, v22.4s
	add	x1, x1, #896
	ldp	q25, q24, [x2, #896]
	add	x2, x23, x1
	fadd	v21.4s, v23.4s, v21.4s
	ldr	q23, [x2]
	fadd	v23.4s, v25.4s, v23.4s
	ldr	q25, [x2, #16]
	fadd	v24.4s, v24.4s, v25.4s
	fmax	v1.4s, v1.4s, v0.4s
	fmax	v2.4s, v2.4s, v0.4s
	add	x13, x22, x13, lsl #2
	stp	q1, q2, [x13]
	fmax	v1.4s, v3.4s, v0.4s
	fmax	v2.4s, v4.4s, v0.4s
	add	x13, x22, x16
	stp	q1, q2, [x13]
	fmax	v1.4s, v5.4s, v0.4s
	fmax	v2.4s, v6.4s, v0.4s
	add	x13, x22, x15
	stp	q1, q2, [x13]
	fmax	v1.4s, v7.4s, v0.4s
	fmax	v2.4s, v16.4s, v0.4s
	add	x13, x22, x14
	stp	q1, q2, [x13]
	fmax	v1.4s, v17.4s, v0.4s
	fmax	v2.4s, v18.4s, v0.4s
	add	x13, x22, x17
	stp	q1, q2, [x13]
	fmax	v1.4s, v19.4s, v0.4s
	fmax	v2.4s, v20.4s, v0.4s
	add	x13, x22, x18
	stp	q2, q1, [x13]
	fmax	v1.4s, v21.4s, v0.4s
	fmax	v2.4s, v22.4s, v0.4s
	add	x13, x22, x0
	stp	q2, q1, [x13]
	fmax	v1.4s, v24.4s, v0.4s
	fmax	v2.4s, v23.4s, v0.4s
	add	x13, x22, x1
	stp	q2, q1, [x13]
	add	x11, x11, #8
	add	x10, x10, #32
.LBB0_5:                                //   Parent Loop BB0_2 Depth=1
                                        // =>  This Loop Header: Depth=2
                                        //       Child Loop BB0_7 Depth 3
	cmp	x11, #31
	b.gt	.LBB0_1
// %bb.6:                               //   in Loop: Header=BB0_5 Depth=2
	mov	x14, xzr
	mov	x15, xzr
	add	x13, x11, x9, lsl #5
	add	x16, x25, x13, lsl #2
	stp	q0, q0, [x16]
	stp	q0, q0, [x16, #128]
	stp	q0, q0, [x16, #256]
	stp	q0, q0, [x16, #384]
	stp	q0, q0, [x16, #512]
	stp	q0, q0, [x16, #640]
	stp	q0, q0, [x16, #768]
	stp	q0, q0, [x16, #896]
	add	x16, x24, x10
	cmp	x15, #63
	b.gt	.LBB0_4
	.p2align	5, , 16
.LBB0_7:                                //   Parent Loop BB0_2 Depth=1
                                        //     Parent Loop BB0_5 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	add	x0, x12, x14
	add	x18, x16, x14, lsl #5
	add	x17, x25, x13, lsl #2
	ldr	q25, [x0]
	ldp	q8, q10, [x18]
	ldp	q1, q2, [x17]
	fmla	v2.4s, v10.4s, v25.s[0]
	fmla	v1.4s, v8.4s, v25.s[0]
	ldr	q26, [x0, #256]
	ldp	q3, q4, [x17, #128]
	fmla	v4.4s, v10.4s, v26.s[0]
	fmla	v3.4s, v8.4s, v26.s[0]
	ldr	q27, [x0, #512]
	ldp	q5, q6, [x17, #256]
	fmla	v6.4s, v10.4s, v27.s[0]
	fmla	v5.4s, v8.4s, v27.s[0]
	ldr	q28, [x0, #768]
	ldp	q7, q16, [x17, #384]
	fmla	v16.4s, v10.4s, v28.s[0]
	fmla	v7.4s, v8.4s, v28.s[0]
	ldr	q29, [x0, #1024]
	ldp	q17, q18, [x17, #512]
	fmla	v18.4s, v10.4s, v29.s[0]
	fmla	v17.4s, v8.4s, v29.s[0]
	ldr	q30, [x0, #1280]
	ldp	q19, q20, [x17, #640]
	fmla	v20.4s, v10.4s, v30.s[0]
	fmla	v19.4s, v8.4s, v30.s[0]
	ldr	q31, [x0, #1536]
	ldp	q21, q22, [x17, #768]
	fmla	v22.4s, v10.4s, v31.s[0]
	fmla	v21.4s, v8.4s, v31.s[0]
	ldr	q9, [x0, #1792]
	ldp	q23, q24, [x17, #896]
	fmla	v24.4s, v10.4s, v9.s[0]
	fmla	v23.4s, v8.4s, v9.s[0]
	ldp	q10, q8, [x18, #128]
	fmla	v1.4s, v10.4s, v25.s[1]
	fmla	v2.4s, v8.4s, v25.s[1]
	fmla	v3.4s, v10.4s, v26.s[1]
	fmla	v4.4s, v8.4s, v26.s[1]
	fmla	v5.4s, v10.4s, v27.s[1]
	fmla	v6.4s, v8.4s, v27.s[1]
	fmla	v7.4s, v10.4s, v28.s[1]
	fmla	v16.4s, v8.4s, v28.s[1]
	fmla	v17.4s, v10.4s, v29.s[1]
	fmla	v18.4s, v8.4s, v29.s[1]
	fmla	v19.4s, v10.4s, v30.s[1]
	fmla	v20.4s, v8.4s, v30.s[1]
	fmla	v21.4s, v10.4s, v31.s[1]
	fmla	v22.4s, v8.4s, v31.s[1]
	fmla	v23.4s, v10.4s, v9.s[1]
	fmla	v24.4s, v8.4s, v9.s[1]
	ldp	q8, q10, [x18, #256]
	fmla	v2.4s, v10.4s, v25.s[2]
	fmla	v1.4s, v8.4s, v25.s[2]
	fmla	v4.4s, v10.4s, v26.s[2]
	fmla	v3.4s, v8.4s, v26.s[2]
	fmla	v6.4s, v10.4s, v27.s[2]
	fmla	v5.4s, v8.4s, v27.s[2]
	fmla	v16.4s, v10.4s, v28.s[2]
	fmla	v7.4s, v8.4s, v28.s[2]
	fmla	v18.4s, v10.4s, v29.s[2]
	fmla	v17.4s, v8.4s, v29.s[2]
	fmla	v20.4s, v10.4s, v30.s[2]
	fmla	v19.4s, v8.4s, v30.s[2]
	fmla	v22.4s, v10.4s, v31.s[2]
	fmla	v21.4s, v8.4s, v31.s[2]
	fmla	v24.4s, v10.4s, v9.s[2]
	fmla	v23.4s, v8.4s, v9.s[2]
	ldp	q11, q10, [x18, #384]
	fmla	v1.4s, v11.4s, v25.s[3]
	fmla	v2.4s, v10.4s, v25.s[3]
	ldr	q25, [x0, #16]
	fmla	v3.4s, v11.4s, v26.s[3]
	fmla	v4.4s, v10.4s, v26.s[3]
	ldr	q26, [x0, #272]
	fmla	v5.4s, v11.4s, v27.s[3]
	fmla	v6.4s, v10.4s, v27.s[3]
	ldr	q27, [x0, #528]
	fmla	v7.4s, v11.4s, v28.s[3]
	fmla	v16.4s, v10.4s, v28.s[3]
	ldr	q28, [x0, #784]
	fmla	v17.4s, v11.4s, v29.s[3]
	fmla	v18.4s, v10.4s, v29.s[3]
	ldr	q29, [x0, #1040]
	fmla	v19.4s, v11.4s, v30.s[3]
	fmla	v20.4s, v10.4s, v30.s[3]
	ldr	q30, [x0, #1296]
	fmla	v21.4s, v11.4s, v31.s[3]
	fmla	v22.4s, v10.4s, v31.s[3]
	ldr	q8, [x0, #1552]
	fmla	v23.4s, v11.4s, v9.s[3]
	ldr	q31, [x0, #1808]
	fmla	v24.4s, v10.4s, v9.s[3]
	ldp	q9, q10, [x18, #512]
	fmla	v2.4s, v10.4s, v25.s[0]
	fmla	v1.4s, v9.4s, v25.s[0]
	fmla	v4.4s, v10.4s, v26.s[0]
	fmla	v3.4s, v9.4s, v26.s[0]
	fmla	v6.4s, v10.4s, v27.s[0]
	fmla	v5.4s, v9.4s, v27.s[0]
	fmla	v16.4s, v10.4s, v28.s[0]
	fmla	v7.4s, v9.4s, v28.s[0]
	fmla	v18.4s, v10.4s, v29.s[0]
	fmla	v17.4s, v9.4s, v29.s[0]
	fmla	v20.4s, v10.4s, v30.s[0]
	fmla	v19.4s, v9.4s, v30.s[0]
	fmla	v22.4s, v10.4s, v8.s[0]
	fmla	v21.4s, v9.4s, v8.s[0]
	fmla	v24.4s, v10.4s, v31.s[0]
	fmla	v23.4s, v9.4s, v31.s[0]
	ldp	q10, q9, [x18, #640]
	fmla	v1.4s, v10.4s, v25.s[1]
	fmla	v2.4s, v9.4s, v25.s[1]
	fmla	v3.4s, v10.4s, v26.s[1]
	fmla	v4.4s, v9.4s, v26.s[1]
	fmla	v5.4s, v10.4s, v27.s[1]
	fmla	v6.4s, v9.4s, v27.s[1]
	fmla	v7.4s, v10.4s, v28.s[1]
	fmla	v16.4s, v9.4s, v28.s[1]
	fmla	v17.4s, v10.4s, v29.s[1]
	fmla	v18.4s, v9.4s, v29.s[1]
	fmla	v19.4s, v10.4s, v30.s[1]
	fmla	v20.4s, v9.4s, v30.s[1]
	fmla	v21.4s, v10.4s, v8.s[1]
	fmla	v22.4s, v9.4s, v8.s[1]
	fmla	v23.4s, v10.4s, v31.s[1]
	fmla	v24.4s, v9.4s, v31.s[1]
	ldp	q9, q10, [x18, #768]
	fmla	v2.4s, v10.4s, v25.s[2]
	fmla	v1.4s, v9.4s, v25.s[2]
	fmla	v4.4s, v10.4s, v26.s[2]
	fmla	v3.4s, v9.4s, v26.s[2]
	fmla	v6.4s, v10.4s, v27.s[2]
	fmla	v5.4s, v9.4s, v27.s[2]
	fmla	v16.4s, v10.4s, v28.s[2]
	fmla	v7.4s, v9.4s, v28.s[2]
	fmla	v18.4s, v10.4s, v29.s[2]
	fmla	v17.4s, v9.4s, v29.s[2]
	fmla	v20.4s, v10.4s, v30.s[2]
	fmla	v19.4s, v9.4s, v30.s[2]
	fmla	v22.4s, v10.4s, v8.s[2]
	fmla	v21.4s, v9.4s, v8.s[2]
	fmla	v24.4s, v10.4s, v31.s[2]
	fmla	v23.4s, v9.4s, v31.s[2]
	ldp	q10, q9, [x18, #896]
	fmla	v1.4s, v10.4s, v25.s[3]
	fmla	v2.4s, v9.4s, v25.s[3]
	fmla	v3.4s, v10.4s, v26.s[3]
	fmla	v4.4s, v9.4s, v26.s[3]
	fmla	v5.4s, v10.4s, v27.s[3]
	fmla	v6.4s, v9.4s, v27.s[3]
	fmla	v7.4s, v10.4s, v28.s[3]
	fmla	v16.4s, v9.4s, v28.s[3]
	fmla	v17.4s, v10.4s, v29.s[3]
	fmla	v18.4s, v9.4s, v29.s[3]
	fmla	v19.4s, v10.4s, v30.s[3]
	fmla	v20.4s, v9.4s, v30.s[3]
	fmla	v21.4s, v10.4s, v8.s[3]
	fmla	v22.4s, v9.4s, v8.s[3]
	fmla	v23.4s, v10.4s, v31.s[3]
	fmla	v24.4s, v9.4s, v31.s[3]
	ldr	q30, [x0, #32]
	ldr	q31, [x0, #288]
	ldr	q8, [x0, #544]
	ldr	q29, [x0, #800]
	ldr	q28, [x0, #1056]
	ldr	q27, [x0, #1312]
	ldr	q26, [x0, #1568]
	ldr	q25, [x0, #1824]
	ldr	q9, [x18, #1024]
	ldr	q10, [x18, #1040]
	fmla	v2.4s, v10.4s, v30.s[0]
	fmla	v4.4s, v10.4s, v31.s[0]
	fmla	v6.4s, v10.4s, v8.s[0]
	fmla	v16.4s, v10.4s, v29.s[0]
	fmla	v18.4s, v10.4s, v28.s[0]
	fmla	v20.4s, v10.4s, v27.s[0]
	fmla	v22.4s, v10.4s, v26.s[0]
	fmla	v24.4s, v10.4s, v25.s[0]
	ldr	q10, [x18, #1168]
	fmla	v1.4s, v9.4s, v30.s[0]
	fmla	v3.4s, v9.4s, v31.s[0]
	fmla	v5.4s, v9.4s, v8.s[0]
	fmla	v7.4s, v9.4s, v29.s[0]
	fmla	v17.4s, v9.4s, v28.s[0]
	fmla	v19.4s, v9.4s, v27.s[0]
	fmla	v21.4s, v9.4s, v26.s[0]
	fmla	v23.4s, v9.4s, v25.s[0]
	ldr	q9, [x18, #1152]
	fmla	v1.4s, v9.4s, v30.s[1]
	fmla	v3.4s, v9.4s, v31.s[1]
	fmla	v5.4s, v9.4s, v8.s[1]
	fmla	v7.4s, v9.4s, v29.s[1]
	fmla	v17.4s, v9.4s, v28.s[1]
	fmla	v19.4s, v9.4s, v27.s[1]
	fmla	v21.4s, v9.4s, v26.s[1]
	fmla	v23.4s, v9.4s, v25.s[1]
	ldr	q9, [x18, #1280]
	fmla	v2.4s, v10.4s, v30.s[1]
	fmla	v4.4s, v10.4s, v31.s[1]
	fmla	v6.4s, v10.4s, v8.s[1]
	fmla	v16.4s, v10.4s, v29.s[1]
	fmla	v18.4s, v10.4s, v28.s[1]
	fmla	v20.4s, v10.4s, v27.s[1]
	fmla	v22.4s, v10.4s, v26.s[1]
	fmla	v24.4s, v10.4s, v25.s[1]
	ldr	q10, [x18, #1296]
	fmla	v2.4s, v10.4s, v30.s[2]
	fmla	v4.4s, v10.4s, v31.s[2]
	fmla	v6.4s, v10.4s, v8.s[2]
	fmla	v16.4s, v10.4s, v29.s[2]
	fmla	v18.4s, v10.4s, v28.s[2]
	fmla	v20.4s, v10.4s, v27.s[2]
	fmla	v22.4s, v10.4s, v26.s[2]
	fmla	v24.4s, v10.4s, v25.s[2]
	ldr	q10, [x18, #1424]
	fmla	v1.4s, v9.4s, v30.s[2]
	fmla	v3.4s, v9.4s, v31.s[2]
	fmla	v5.4s, v9.4s, v8.s[2]
	fmla	v7.4s, v9.4s, v29.s[2]
	fmla	v17.4s, v9.4s, v28.s[2]
	fmla	v19.4s, v9.4s, v27.s[2]
	fmla	v21.4s, v9.4s, v26.s[2]
	fmla	v23.4s, v9.4s, v25.s[2]
	ldr	q11, [x18, #1408]
	fmla	v1.4s, v11.4s, v30.s[3]
	fmla	v2.4s, v10.4s, v30.s[3]
	ldr	q30, [x0, #48]
	fmla	v3.4s, v11.4s, v31.s[3]
	fmla	v4.4s, v10.4s, v31.s[3]
	ldr	q31, [x0, #304]
	fmla	v5.4s, v11.4s, v8.s[3]
	fmla	v6.4s, v10.4s, v8.s[3]
	ldr	q8, [x0, #560]
	fmla	v7.4s, v11.4s, v29.s[3]
	fmla	v16.4s, v10.4s, v29.s[3]
	ldr	q29, [x0, #816]
	fmla	v17.4s, v11.4s, v28.s[3]
	fmla	v18.4s, v10.4s, v28.s[3]
	ldr	q28, [x0, #1072]
	fmla	v19.4s, v11.4s, v27.s[3]
	fmla	v20.4s, v10.4s, v27.s[3]
	ldr	q27, [x0, #1328]
	fmla	v21.4s, v11.4s, v26.s[3]
	fmla	v22.4s, v10.4s, v26.s[3]
	ldr	q9, [x0, #1584]
	fmla	v23.4s, v11.4s, v25.s[3]
	ldr	q26, [x0, #1840]
	fmla	v24.4s, v10.4s, v25.s[3]
	ldr	q25, [x18, #1536]
	ldr	q10, [x18, #1552]
	fmla	v2.4s, v10.4s, v30.s[0]
	fmla	v4.4s, v10.4s, v31.s[0]
	fmla	v6.4s, v10.4s, v8.s[0]
	fmla	v16.4s, v10.4s, v29.s[0]
	fmla	v18.4s, v10.4s, v28.s[0]
	fmla	v20.4s, v10.4s, v27.s[0]
	fmla	v22.4s, v10.4s, v9.s[0]
	fmla	v24.4s, v10.4s, v26.s[0]
	ldr	q10, [x18, #1680]
	fmla	v1.4s, v25.4s, v30.s[0]
	fmla	v3.4s, v25.4s, v31.s[0]
	fmla	v5.4s, v25.4s, v8.s[0]
	fmla	v7.4s, v25.4s, v29.s[0]
	fmla	v17.4s, v25.4s, v28.s[0]
	fmla	v19.4s, v25.4s, v27.s[0]
	fmla	v21.4s, v25.4s, v9.s[0]
	fmla	v23.4s, v25.4s, v26.s[0]
	ldr	q25, [x18, #1664]
	fmla	v1.4s, v25.4s, v30.s[1]
	fmla	v3.4s, v25.4s, v31.s[1]
	fmla	v5.4s, v25.4s, v8.s[1]
	fmla	v7.4s, v25.4s, v29.s[1]
	fmla	v17.4s, v25.4s, v28.s[1]
	fmla	v19.4s, v25.4s, v27.s[1]
	fmla	v21.4s, v25.4s, v9.s[1]
	fmla	v23.4s, v25.4s, v26.s[1]
	ldr	q25, [x18, #1792]
	fmla	v2.4s, v10.4s, v30.s[1]
	fmla	v4.4s, v10.4s, v31.s[1]
	fmla	v6.4s, v10.4s, v8.s[1]
	fmla	v16.4s, v10.4s, v29.s[1]
	fmla	v18.4s, v10.4s, v28.s[1]
	fmla	v20.4s, v10.4s, v27.s[1]
	fmla	v22.4s, v10.4s, v9.s[1]
	fmla	v24.4s, v10.4s, v26.s[1]
	ldr	q10, [x18, #1808]
	fmla	v2.4s, v10.4s, v30.s[2]
	fmla	v4.4s, v10.4s, v31.s[2]
	fmla	v6.4s, v10.4s, v8.s[2]
	fmla	v16.4s, v10.4s, v29.s[2]
	fmla	v18.4s, v10.4s, v28.s[2]
	fmla	v20.4s, v10.4s, v27.s[2]
	fmla	v22.4s, v10.4s, v9.s[2]
	fmla	v24.4s, v10.4s, v26.s[2]
	ldr	q10, [x18, #1936]
	fmla	v1.4s, v25.4s, v30.s[2]
	fmla	v3.4s, v25.4s, v31.s[2]
	fmla	v5.4s, v25.4s, v8.s[2]
	fmla	v7.4s, v25.4s, v29.s[2]
	fmla	v17.4s, v25.4s, v28.s[2]
	fmla	v19.4s, v25.4s, v27.s[2]
	fmla	v21.4s, v25.4s, v9.s[2]
	fmla	v23.4s, v25.4s, v26.s[2]
	ldr	q25, [x18, #1920]
	fmla	v1.4s, v25.4s, v30.s[3]
	fmla	v2.4s, v10.4s, v30.s[3]
	fmla	v3.4s, v25.4s, v31.s[3]
	fmla	v4.4s, v10.4s, v31.s[3]
	fmla	v5.4s, v25.4s, v8.s[3]
	fmla	v6.4s, v10.4s, v8.s[3]
	fmla	v7.4s, v25.4s, v29.s[3]
	fmla	v16.4s, v10.4s, v29.s[3]
	fmla	v17.4s, v25.4s, v28.s[3]
	fmla	v18.4s, v10.4s, v28.s[3]
	fmla	v19.4s, v25.4s, v27.s[3]
	fmla	v20.4s, v10.4s, v27.s[3]
	fmla	v21.4s, v25.4s, v9.s[3]
	fmla	v22.4s, v10.4s, v9.s[3]
	fmla	v23.4s, v25.4s, v26.s[3]
	fmla	v24.4s, v10.4s, v26.s[3]
	ldr	q30, [x0, #64]
	ldr	q31, [x0, #320]
	ldr	q8, [x0, #576]
	ldr	q29, [x0, #832]
	ldr	q28, [x0, #1088]
	ldr	q27, [x0, #1344]
	ldr	q26, [x0, #1600]
	ldr	q25, [x0, #1856]
	ldr	q9, [x18, #2048]
	ldr	q10, [x18, #2064]
	fmla	v2.4s, v10.4s, v30.s[0]
	fmla	v4.4s, v10.4s, v31.s[0]
	fmla	v6.4s, v10.4s, v8.s[0]
	fmla	v16.4s, v10.4s, v29.s[0]
	fmla	v18.4s, v10.4s, v28.s[0]
	fmla	v20.4s, v10.4s, v27.s[0]
	fmla	v22.4s, v10.4s, v26.s[0]
	fmla	v24.4s, v10.4s, v25.s[0]
	ldr	q10, [x18, #2192]
	fmla	v1.4s, v9.4s, v30.s[0]
	fmla	v3.4s, v9.4s, v31.s[0]
	fmla	v5.4s, v9.4s, v8.s[0]
	fmla	v7.4s, v9.4s, v29.s[0]
	fmla	v17.4s, v9.4s, v28.s[0]
	fmla	v19.4s, v9.4s, v27.s[0]
	fmla	v21.4s, v9.4s, v26.s[0]
	fmla	v23.4s, v9.4s, v25.s[0]
	ldr	q9, [x18, #2176]
	fmla	v1.4s, v9.4s, v30.s[1]
	fmla	v3.4s, v9.4s, v31.s[1]
	fmla	v5.4s, v9.4s, v8.s[1]
	fmla	v7.4s, v9.4s, v29.s[1]
	fmla	v17.4s, v9.4s, v28.s[1]
	fmla	v19.4s, v9.4s, v27.s[1]
	fmla	v21.4s, v9.4s, v26.s[1]
	fmla	v23.4s, v9.4s, v25.s[1]
	ldr	q9, [x18, #2304]
	fmla	v2.4s, v10.4s, v30.s[1]
	fmla	v4.4s, v10.4s, v31.s[1]
	fmla	v6.4s, v10.4s, v8.s[1]
	fmla	v16.4s, v10.4s, v29.s[1]
	fmla	v18.4s, v10.4s, v28.s[1]
	fmla	v20.4s, v10.4s, v27.s[1]
	fmla	v22.4s, v10.4s, v26.s[1]
	fmla	v24.4s, v10.4s, v25.s[1]
	ldr	q10, [x18, #2320]
	fmla	v2.4s, v10.4s, v30.s[2]
	fmla	v4.4s, v10.4s, v31.s[2]
	fmla	v6.4s, v10.4s, v8.s[2]
	fmla	v16.4s, v10.4s, v29.s[2]
	fmla	v18.4s, v10.4s, v28.s[2]
	fmla	v20.4s, v10.4s, v27.s[2]
	fmla	v22.4s, v10.4s, v26.s[2]
	fmla	v24.4s, v10.4s, v25.s[2]
	ldr	q10, [x18, #2448]
	fmla	v1.4s, v9.4s, v30.s[2]
	fmla	v3.4s, v9.4s, v31.s[2]
	fmla	v5.4s, v9.4s, v8.s[2]
	fmla	v7.4s, v9.4s, v29.s[2]
	fmla	v17.4s, v9.4s, v28.s[2]
	fmla	v19.4s, v9.4s, v27.s[2]
	fmla	v21.4s, v9.4s, v26.s[2]
	fmla	v23.4s, v9.4s, v25.s[2]
	ldr	q11, [x18, #2432]
	fmla	v1.4s, v11.4s, v30.s[3]
	fmla	v2.4s, v10.4s, v30.s[3]
	ldr	q30, [x0, #80]
	fmla	v3.4s, v11.4s, v31.s[3]
	fmla	v4.4s, v10.4s, v31.s[3]
	ldr	q31, [x0, #336]
	fmla	v5.4s, v11.4s, v8.s[3]
	fmla	v6.4s, v10.4s, v8.s[3]
	ldr	q8, [x0, #592]
	fmla	v7.4s, v11.4s, v29.s[3]
	fmla	v16.4s, v10.4s, v29.s[3]
	ldr	q29, [x0, #848]
	fmla	v17.4s, v11.4s, v28.s[3]
	fmla	v18.4s, v10.4s, v28.s[3]
	ldr	q28, [x0, #1104]
	fmla	v19.4s, v11.4s, v27.s[3]
	fmla	v20.4s, v10.4s, v27.s[3]
	ldr	q27, [x0, #1360]
	fmla	v21.4s, v11.4s, v26.s[3]
	fmla	v22.4s, v10.4s, v26.s[3]
	ldr	q9, [x0, #1616]
	fmla	v23.4s, v11.4s, v25.s[3]
	ldr	q26, [x0, #1872]
	fmla	v24.4s, v10.4s, v25.s[3]
	ldr	q25, [x18, #2560]
	ldr	q10, [x18, #2576]
	fmla	v2.4s, v10.4s, v30.s[0]
	fmla	v4.4s, v10.4s, v31.s[0]
	fmla	v6.4s, v10.4s, v8.s[0]
	fmla	v16.4s, v10.4s, v29.s[0]
	fmla	v18.4s, v10.4s, v28.s[0]
	fmla	v20.4s, v10.4s, v27.s[0]
	fmla	v22.4s, v10.4s, v9.s[0]
	fmla	v24.4s, v10.4s, v26.s[0]
	ldr	q10, [x18, #2704]
	fmla	v1.4s, v25.4s, v30.s[0]
	fmla	v3.4s, v25.4s, v31.s[0]
	fmla	v5.4s, v25.4s, v8.s[0]
	fmla	v7.4s, v25.4s, v29.s[0]
	fmla	v17.4s, v25.4s, v28.s[0]
	fmla	v19.4s, v25.4s, v27.s[0]
	fmla	v21.4s, v25.4s, v9.s[0]
	fmla	v23.4s, v25.4s, v26.s[0]
	ldr	q25, [x18, #2688]
	fmla	v1.4s, v25.4s, v30.s[1]
	fmla	v3.4s, v25.4s, v31.s[1]
	fmla	v5.4s, v25.4s, v8.s[1]
	fmla	v7.4s, v25.4s, v29.s[1]
	fmla	v17.4s, v25.4s, v28.s[1]
	fmla	v19.4s, v25.4s, v27.s[1]
	fmla	v21.4s, v25.4s, v9.s[1]
	fmla	v23.4s, v25.4s, v26.s[1]
	ldr	q25, [x18, #2816]
	fmla	v2.4s, v10.4s, v30.s[1]
	fmla	v4.4s, v10.4s, v31.s[1]
	fmla	v6.4s, v10.4s, v8.s[1]
	fmla	v16.4s, v10.4s, v29.s[1]
	fmla	v18.4s, v10.4s, v28.s[1]
	fmla	v20.4s, v10.4s, v27.s[1]
	fmla	v22.4s, v10.4s, v9.s[1]
	fmla	v24.4s, v10.4s, v26.s[1]
	ldr	q10, [x18, #2832]
	fmla	v2.4s, v10.4s, v30.s[2]
	fmla	v4.4s, v10.4s, v31.s[2]
	fmla	v6.4s, v10.4s, v8.s[2]
	fmla	v16.4s, v10.4s, v29.s[2]
	fmla	v18.4s, v10.4s, v28.s[2]
	fmla	v20.4s, v10.4s, v27.s[2]
	fmla	v22.4s, v10.4s, v9.s[2]
	fmla	v24.4s, v10.4s, v26.s[2]
	ldr	q10, [x18, #2960]
	fmla	v1.4s, v25.4s, v30.s[2]
	fmla	v3.4s, v25.4s, v31.s[2]
	fmla	v5.4s, v25.4s, v8.s[2]
	fmla	v7.4s, v25.4s, v29.s[2]
	fmla	v17.4s, v25.4s, v28.s[2]
	fmla	v19.4s, v25.4s, v27.s[2]
	fmla	v21.4s, v25.4s, v9.s[2]
	fmla	v23.4s, v25.4s, v26.s[2]
	ldr	q25, [x18, #2944]
	fmla	v1.4s, v25.4s, v30.s[3]
	fmla	v2.4s, v10.4s, v30.s[3]
	fmla	v3.4s, v25.4s, v31.s[3]
	fmla	v4.4s, v10.4s, v31.s[3]
	fmla	v5.4s, v25.4s, v8.s[3]
	fmla	v6.4s, v10.4s, v8.s[3]
	fmla	v7.4s, v25.4s, v29.s[3]
	fmla	v16.4s, v10.4s, v29.s[3]
	fmla	v17.4s, v25.4s, v28.s[3]
	fmla	v18.4s, v10.4s, v28.s[3]
	fmla	v19.4s, v25.4s, v27.s[3]
	fmla	v20.4s, v10.4s, v27.s[3]
	fmla	v21.4s, v25.4s, v9.s[3]
	fmla	v22.4s, v10.4s, v9.s[3]
	fmla	v23.4s, v25.4s, v26.s[3]
	fmla	v24.4s, v10.4s, v26.s[3]
	ldr	q30, [x0, #96]
	ldr	q31, [x0, #352]
	ldr	q8, [x0, #608]
	ldr	q29, [x0, #864]
	ldr	q28, [x0, #1120]
	ldr	q27, [x0, #1376]
	ldr	q26, [x0, #1632]
	ldr	q25, [x0, #1888]
	ldr	q9, [x18, #3072]
	ldr	q10, [x18, #3088]
	fmla	v2.4s, v10.4s, v30.s[0]
	fmla	v4.4s, v10.4s, v31.s[0]
	fmla	v6.4s, v10.4s, v8.s[0]
	fmla	v16.4s, v10.4s, v29.s[0]
	fmla	v18.4s, v10.4s, v28.s[0]
	fmla	v20.4s, v10.4s, v27.s[0]
	fmla	v22.4s, v10.4s, v26.s[0]
	fmla	v24.4s, v10.4s, v25.s[0]
	ldr	q10, [x18, #3216]
	fmla	v1.4s, v9.4s, v30.s[0]
	fmla	v3.4s, v9.4s, v31.s[0]
	fmla	v5.4s, v9.4s, v8.s[0]
	fmla	v7.4s, v9.4s, v29.s[0]
	fmla	v17.4s, v9.4s, v28.s[0]
	fmla	v19.4s, v9.4s, v27.s[0]
	fmla	v21.4s, v9.4s, v26.s[0]
	fmla	v23.4s, v9.4s, v25.s[0]
	ldr	q9, [x18, #3200]
	fmla	v1.4s, v9.4s, v30.s[1]
	fmla	v3.4s, v9.4s, v31.s[1]
	fmla	v5.4s, v9.4s, v8.s[1]
	fmla	v7.4s, v9.4s, v29.s[1]
	fmla	v17.4s, v9.4s, v28.s[1]
	fmla	v19.4s, v9.4s, v27.s[1]
	fmla	v21.4s, v9.4s, v26.s[1]
	fmla	v23.4s, v9.4s, v25.s[1]
	ldr	q9, [x18, #3328]
	fmla	v2.4s, v10.4s, v30.s[1]
	fmla	v4.4s, v10.4s, v31.s[1]
	fmla	v6.4s, v10.4s, v8.s[1]
	fmla	v16.4s, v10.4s, v29.s[1]
	fmla	v18.4s, v10.4s, v28.s[1]
	fmla	v20.4s, v10.4s, v27.s[1]
	fmla	v22.4s, v10.4s, v26.s[1]
	fmla	v24.4s, v10.4s, v25.s[1]
	ldr	q10, [x18, #3344]
	fmla	v2.4s, v10.4s, v30.s[2]
	fmla	v4.4s, v10.4s, v31.s[2]
	fmla	v6.4s, v10.4s, v8.s[2]
	fmla	v16.4s, v10.4s, v29.s[2]
	fmla	v18.4s, v10.4s, v28.s[2]
	fmla	v20.4s, v10.4s, v27.s[2]
	fmla	v22.4s, v10.4s, v26.s[2]
	fmla	v24.4s, v10.4s, v25.s[2]
	ldr	q10, [x18, #3472]
	fmla	v1.4s, v9.4s, v30.s[2]
	fmla	v3.4s, v9.4s, v31.s[2]
	fmla	v5.4s, v9.4s, v8.s[2]
	fmla	v7.4s, v9.4s, v29.s[2]
	fmla	v17.4s, v9.4s, v28.s[2]
	fmla	v19.4s, v9.4s, v27.s[2]
	fmla	v21.4s, v9.4s, v26.s[2]
	fmla	v23.4s, v9.4s, v25.s[2]
	ldr	q11, [x18, #3456]
	fmla	v1.4s, v11.4s, v30.s[3]
	fmla	v2.4s, v10.4s, v30.s[3]
	ldr	q30, [x0, #112]
	fmla	v3.4s, v11.4s, v31.s[3]
	fmla	v4.4s, v10.4s, v31.s[3]
	ldr	q31, [x0, #368]
	fmla	v5.4s, v11.4s, v8.s[3]
	fmla	v6.4s, v10.4s, v8.s[3]
	ldr	q8, [x0, #624]
	fmla	v7.4s, v11.4s, v29.s[3]
	fmla	v16.4s, v10.4s, v29.s[3]
	ldr	q29, [x0, #880]
	fmla	v17.4s, v11.4s, v28.s[3]
	fmla	v18.4s, v10.4s, v28.s[3]
	ldr	q28, [x0, #1136]
	fmla	v19.4s, v11.4s, v27.s[3]
	fmla	v20.4s, v10.4s, v27.s[3]
	ldr	q27, [x0, #1392]
	fmla	v21.4s, v11.4s, v26.s[3]
	fmla	v22.4s, v10.4s, v26.s[3]
	ldr	q9, [x0, #1648]
	fmla	v23.4s, v11.4s, v25.s[3]
	ldr	q26, [x0, #1904]
	fmla	v24.4s, v10.4s, v25.s[3]
	ldr	q25, [x18, #3584]
	ldr	q10, [x18, #3600]
	fmla	v2.4s, v10.4s, v30.s[0]
	fmla	v4.4s, v10.4s, v31.s[0]
	fmla	v6.4s, v10.4s, v8.s[0]
	fmla	v16.4s, v10.4s, v29.s[0]
	fmla	v18.4s, v10.4s, v28.s[0]
	fmla	v20.4s, v10.4s, v27.s[0]
	fmla	v22.4s, v10.4s, v9.s[0]
	fmla	v24.4s, v10.4s, v26.s[0]
	ldr	q10, [x18, #3728]
	fmla	v1.4s, v25.4s, v30.s[0]
	fmla	v3.4s, v25.4s, v31.s[0]
	fmla	v5.4s, v25.4s, v8.s[0]
	fmla	v7.4s, v25.4s, v29.s[0]
	fmla	v17.4s, v25.4s, v28.s[0]
	fmla	v19.4s, v25.4s, v27.s[0]
	fmla	v21.4s, v25.4s, v9.s[0]
	fmla	v23.4s, v25.4s, v26.s[0]
	ldr	q25, [x18, #3712]
	fmla	v1.4s, v25.4s, v30.s[1]
	fmla	v3.4s, v25.4s, v31.s[1]
	fmla	v5.4s, v25.4s, v8.s[1]
	fmla	v7.4s, v25.4s, v29.s[1]
	fmla	v17.4s, v25.4s, v28.s[1]
	fmla	v19.4s, v25.4s, v27.s[1]
	fmla	v21.4s, v25.4s, v9.s[1]
	fmla	v23.4s, v25.4s, v26.s[1]
	ldr	q25, [x18, #3840]
	fmla	v2.4s, v10.4s, v30.s[1]
	fmla	v4.4s, v10.4s, v31.s[1]
	fmla	v6.4s, v10.4s, v8.s[1]
	fmla	v16.4s, v10.4s, v29.s[1]
	fmla	v18.4s, v10.4s, v28.s[1]
	fmla	v20.4s, v10.4s, v27.s[1]
	fmla	v22.4s, v10.4s, v9.s[1]
	fmla	v24.4s, v10.4s, v26.s[1]
	ldr	q10, [x18, #3856]
	fmla	v2.4s, v10.4s, v30.s[2]
	fmla	v4.4s, v10.4s, v31.s[2]
	fmla	v6.4s, v10.4s, v8.s[2]
	fmla	v16.4s, v10.4s, v29.s[2]
	fmla	v18.4s, v10.4s, v28.s[2]
	fmla	v20.4s, v10.4s, v27.s[2]
	fmla	v22.4s, v10.4s, v9.s[2]
	fmla	v24.4s, v10.4s, v26.s[2]
	ldr	q10, [x18, #3984]
	fmla	v1.4s, v25.4s, v30.s[2]
	fmla	v3.4s, v25.4s, v31.s[2]
	fmla	v5.4s, v25.4s, v8.s[2]
	fmla	v7.4s, v25.4s, v29.s[2]
	fmla	v17.4s, v25.4s, v28.s[2]
	fmla	v19.4s, v25.4s, v27.s[2]
	fmla	v21.4s, v25.4s, v9.s[2]
	fmla	v23.4s, v25.4s, v26.s[2]
	ldr	q25, [x18, #3968]
	fmla	v1.4s, v25.4s, v30.s[3]
	fmla	v2.4s, v10.4s, v30.s[3]
	fmla	v3.4s, v25.4s, v31.s[3]
	fmla	v4.4s, v10.4s, v31.s[3]
	fmla	v5.4s, v25.4s, v8.s[3]
	fmla	v6.4s, v10.4s, v8.s[3]
	fmla	v7.4s, v25.4s, v29.s[3]
	fmla	v16.4s, v10.4s, v29.s[3]
	fmla	v17.4s, v25.4s, v28.s[3]
	fmla	v18.4s, v10.4s, v28.s[3]
	fmla	v19.4s, v25.4s, v27.s[3]
	fmla	v20.4s, v10.4s, v27.s[3]
	fmla	v21.4s, v25.4s, v9.s[3]
	fmla	v22.4s, v10.4s, v9.s[3]
	fmla	v23.4s, v25.4s, v26.s[3]
	fmla	v24.4s, v10.4s, v26.s[3]
	stp	q1, q2, [x17]
	stp	q3, q4, [x17, #128]
	stp	q5, q6, [x17, #256]
	stp	q7, q16, [x17, #384]
	stp	q17, q18, [x17, #512]
	stp	q19, q20, [x17, #640]
	stp	q21, q22, [x17, #768]
	stp	q23, q24, [x17, #896]
	add	x15, x15, #32
	add	x14, x14, #128
	cmp	x15, #63
	b.le	.LBB0_7
	b	.LBB0_4
.LBB0_8:
	mov	x0, x20
	bl	free
	mov	x0, x21
	mov	x1, x22
	mov	x2, xzr
	mov	w3, #32                         // =0x20
	mov	w4, #32                         // =0x20
	mov	w5, #32                         // =0x20
	mov	w6, #1                          // =0x1
	ldp	x20, x19, [sp, #80]             // 16-byte Folded Reload
	ldp	x22, x21, [sp, #64]             // 16-byte Folded Reload
	ldp	x24, x23, [sp, #48]             // 16-byte Folded Reload
	ldp	x30, x25, [sp, #32]             // 16-byte Folded Reload
	ldp	d9, d8, [sp, #16]               // 16-byte Folded Reload
	ldp	d11, d10, [sp], #96             // 16-byte Folded Reload
	ret
.Lfunc_end0:
	.size	matmul_bias_relu_tiled_32x32x64, .Lfunc_end0-matmul_bias_relu_tiled_32x32x64
	.cfi_endproc
                                        // -- End function
	.globl	_mlir_ciface_matmul_bias_relu_tiled_32x32x64 // -- Begin function _mlir_ciface_matmul_bias_relu_tiled_32x32x64
	.p2align	4
	.type	_mlir_ciface_matmul_bias_relu_tiled_32x32x64,@function
_mlir_ciface_matmul_bias_relu_tiled_32x32x64: // @_mlir_ciface_matmul_bias_relu_tiled_32x32x64
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
	bl	matmul_bias_relu_tiled_32x32x64
	stp	x0, x1, [x19]
	stp	x2, x3, [x19, #16]
	stp	x4, x5, [x19, #32]
	str	x6, [x19, #48]
	ldp	x30, x19, [sp, #112]            // 16-byte Folded Reload
	add	sp, sp, #128
	ret
.Lfunc_end1:
	.size	_mlir_ciface_matmul_bias_relu_tiled_32x32x64, .Lfunc_end1-_mlir_ciface_matmul_bias_relu_tiled_32x32x64
	.cfi_endproc
                                        // -- End function
	.section	".note.GNU-stack","",@progbits
