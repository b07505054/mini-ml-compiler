	.file	"LLVMDialectModule"
	.text
	.globl	matmul_bias_relu_tiled_32x32x32 // -- Begin function matmul_bias_relu_tiled_32x32x32
	.p2align	4
	.type	matmul_bias_relu_tiled_32x32x32,@function
matmul_bias_relu_tiled_32x32x32:        // @matmul_bias_relu_tiled_32x32x32
	.cfi_startproc
// %bb.0:
	sub	sp, sp, #224
	stp	d15, d14, [sp, #112]            // 16-byte Folded Spill
	stp	d13, d12, [sp, #128]            // 16-byte Folded Spill
	stp	d11, d10, [sp, #144]            // 16-byte Folded Spill
	stp	d9, d8, [sp, #160]              // 16-byte Folded Spill
	str	x29, [sp, #176]                 // 8-byte Folded Spill
	stp	x30, x21, [sp, #192]            // 16-byte Folded Spill
	stp	x20, x19, [sp, #208]            // 16-byte Folded Spill
	.cfi_def_cfa_offset 224
	.cfi_offset w19, -8
	.cfi_offset w20, -16
	.cfi_offset w21, -24
	.cfi_offset w30, -32
	.cfi_offset w29, -48
	.cfi_offset b8, -56
	.cfi_offset b9, -64
	.cfi_offset b10, -72
	.cfi_offset b11, -80
	.cfi_offset b12, -88
	.cfi_offset b13, -96
	.cfi_offset b14, -104
	.cfi_offset b15, -112
	mov	x19, x1
	ldr	x20, [sp, #280]
	ldr	x21, [sp, #224]
	mov	w0, #4160                       // =0x1040
	bl	malloc
	mov	x8, xzr
	mov	x9, xzr
	add	x10, x0, #63
	and	x1, x10, #0xffffffffffffffc0
	b	.LBB0_2
	.p2align	5, , 16
.LBB0_1:                                //   in Loop: Header=BB0_2 Depth=1
	add	x9, x9, #8
	add	x8, x8, #1024
.LBB0_2:                                // =>This Loop Header: Depth=1
                                        //     Child Loop BB0_4 Depth 2
	cmp	x9, #31
	b.gt	.LBB0_5
// %bb.3:                               // %.preheader
                                        //   in Loop: Header=BB0_2 Depth=1
	mov	x10, xzr
	mov	x11, xzr
	add	x12, x19, x9, lsl #7
	add	x13, x20, x8
	cmp	x11, #31
	b.gt	.LBB0_1
	.p2align	5, , 16
.LBB0_4:                                //   Parent Loop BB0_2 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	movi	v3.2d, #0000000000000000
	ldp	q30, q2, [x12]
	add	x14, x21, x10
	ldp	q14, q1, [x14]
	fmla	v3.4s, v1.4s, v30.s[0]
	movi	v4.2d, #0000000000000000
	fmla	v4.4s, v14.4s, v30.s[0]
	movi	v5.2d, #0000000000000000
	ldp	q31, q0, [x12, #128]
	stp	q0, q2, [sp, #80]               // 32-byte Folded Spill
	fmla	v5.4s, v1.4s, v31.s[0]
	movi	v6.2d, #0000000000000000
	movi	v7.2d, #0000000000000000
	fmla	v6.4s, v14.4s, v31.s[0]
	ldp	q10, q26, [x12, #256]
	fmla	v7.4s, v1.4s, v10.s[0]
	movi	v16.2d, #0000000000000000
	fmla	v16.4s, v14.4s, v10.s[0]
	movi	v17.2d, #0000000000000000
	ldp	q12, q27, [x12, #384]
	fmla	v17.4s, v1.4s, v12.s[0]
	movi	v18.2d, #0000000000000000
	fmla	v18.4s, v14.4s, v12.s[0]
	ldp	q11, q25, [x12, #512]
	movi	v20.2d, #0000000000000000
	fmla	v20.4s, v1.4s, v11.s[0]
	movi	v19.2d, #0000000000000000
	ldp	q13, q2, [x12, #640]
	fmla	v19.4s, v14.4s, v11.s[0]
	ldr	q9, [x12, #768]
	movi	v21.2d, #0000000000000000
	ldr	q8, [x12, #896]
	fmla	v21.4s, v1.4s, v13.s[0]
	movi	v22.2d, #0000000000000000
	fmla	v22.4s, v14.4s, v13.s[0]
	movi	v24.2d, #0000000000000000
	fmla	v24.4s, v1.4s, v9.s[0]
	movi	v23.2d, #0000000000000000
	fmla	v23.4s, v14.4s, v9.s[0]
	movi	v29.2d, #0000000000000000
	fmla	v29.4s, v1.4s, v8.s[0]
	movi	v28.2d, #0000000000000000
	fmla	v28.4s, v14.4s, v8.s[0]
	ldp	q15, q14, [x14, #128]
	fmla	v4.4s, v15.4s, v30.s[1]
	fmla	v3.4s, v14.4s, v30.s[1]
	fmla	v6.4s, v15.4s, v31.s[1]
	fmla	v5.4s, v14.4s, v31.s[1]
	fmla	v16.4s, v15.4s, v10.s[1]
	fmla	v7.4s, v14.4s, v10.s[1]
	fmla	v18.4s, v15.4s, v12.s[1]
	fmla	v17.4s, v14.4s, v12.s[1]
	fmla	v19.4s, v15.4s, v11.s[1]
	fmla	v20.4s, v14.4s, v11.s[1]
	fmla	v22.4s, v15.4s, v13.s[1]
	fmla	v21.4s, v14.4s, v13.s[1]
	fmla	v23.4s, v15.4s, v9.s[1]
	fmla	v24.4s, v14.4s, v9.s[1]
	fmla	v28.4s, v15.4s, v8.s[1]
	fmla	v29.4s, v14.4s, v8.s[1]
	ldp	q14, q15, [x14, #256]
	fmla	v3.4s, v15.4s, v30.s[2]
	fmla	v4.4s, v14.4s, v30.s[2]
	fmla	v5.4s, v15.4s, v31.s[2]
	fmla	v6.4s, v14.4s, v31.s[2]
	fmla	v7.4s, v15.4s, v10.s[2]
	fmla	v16.4s, v14.4s, v10.s[2]
	fmla	v17.4s, v15.4s, v12.s[2]
	fmla	v18.4s, v14.4s, v12.s[2]
	fmla	v20.4s, v15.4s, v11.s[2]
	fmla	v19.4s, v14.4s, v11.s[2]
	fmla	v21.4s, v15.4s, v13.s[2]
	fmla	v22.4s, v14.4s, v13.s[2]
	fmla	v24.4s, v15.4s, v9.s[2]
	fmla	v23.4s, v14.4s, v9.s[2]
	fmla	v29.4s, v15.4s, v8.s[2]
	fmla	v28.4s, v14.4s, v8.s[2]
	ldp	q15, q14, [x14, #384]
	fmla	v4.4s, v15.4s, v30.s[3]
	fmla	v3.4s, v14.4s, v30.s[3]
	ldr	q30, [x12, #784]
	fmla	v6.4s, v15.4s, v31.s[3]
	fmla	v5.4s, v14.4s, v31.s[3]
	ldr	q31, [x12, #912]
	fmla	v16.4s, v15.4s, v10.s[3]
	fmla	v7.4s, v14.4s, v10.s[3]
	fmla	v18.4s, v15.4s, v12.s[3]
	fmla	v17.4s, v14.4s, v12.s[3]
	ldp	q0, q10, [x14, #512]
	fmla	v19.4s, v15.4s, v11.s[3]
	fmla	v20.4s, v14.4s, v11.s[3]
	fmla	v22.4s, v15.4s, v13.s[3]
	fmla	v21.4s, v14.4s, v13.s[3]
	ldp	q13, q12, [x14, #640]
	fmla	v23.4s, v15.4s, v9.s[3]
	fmla	v24.4s, v14.4s, v9.s[3]
	fmla	v28.4s, v15.4s, v8.s[3]
	ldp	q15, q11, [x14, #768]
	fmla	v29.4s, v14.4s, v8.s[3]
	ldp	q9, q8, [x14, #896]
	ldp	q14, q1, [sp, #80]              // 32-byte Folded Reload
	fmla	v3.4s, v10.4s, v1.s[0]
	fmla	v4.4s, v0.4s, v1.s[0]
	fmla	v5.4s, v10.4s, v14.s[0]
	fmla	v6.4s, v0.4s, v14.s[0]
	fmla	v7.4s, v10.4s, v26.s[0]
	fmla	v16.4s, v0.4s, v26.s[0]
	fmla	v17.4s, v10.4s, v27.s[0]
	fmla	v18.4s, v0.4s, v27.s[0]
	fmla	v20.4s, v10.4s, v25.s[0]
	fmla	v19.4s, v0.4s, v25.s[0]
	fmla	v21.4s, v10.4s, v2.s[0]
	fmla	v22.4s, v0.4s, v2.s[0]
	fmla	v24.4s, v10.4s, v30.s[0]
	fmla	v23.4s, v0.4s, v30.s[0]
	fmla	v29.4s, v10.4s, v31.s[0]
	fmla	v28.4s, v0.4s, v31.s[0]
	fmla	v4.4s, v13.4s, v1.s[1]
	fmla	v3.4s, v12.4s, v1.s[1]
	fmla	v6.4s, v13.4s, v14.s[1]
	fmla	v5.4s, v12.4s, v14.s[1]
	fmla	v16.4s, v13.4s, v26.s[1]
	fmla	v7.4s, v12.4s, v26.s[1]
	fmla	v18.4s, v13.4s, v27.s[1]
	fmla	v17.4s, v12.4s, v27.s[1]
	fmla	v19.4s, v13.4s, v25.s[1]
	fmla	v20.4s, v12.4s, v25.s[1]
	fmla	v22.4s, v13.4s, v2.s[1]
	fmla	v21.4s, v12.4s, v2.s[1]
	fmla	v23.4s, v13.4s, v30.s[1]
	fmla	v24.4s, v12.4s, v30.s[1]
	fmla	v28.4s, v13.4s, v31.s[1]
	fmla	v29.4s, v12.4s, v31.s[1]
	fmla	v3.4s, v11.4s, v1.s[2]
	fmla	v4.4s, v15.4s, v1.s[2]
	fmla	v5.4s, v11.4s, v14.s[2]
	fmla	v6.4s, v15.4s, v14.s[2]
	fmla	v7.4s, v11.4s, v26.s[2]
	fmla	v16.4s, v15.4s, v26.s[2]
	fmla	v17.4s, v11.4s, v27.s[2]
	fmla	v18.4s, v15.4s, v27.s[2]
	fmla	v20.4s, v11.4s, v25.s[2]
	fmla	v19.4s, v15.4s, v25.s[2]
	fmla	v21.4s, v11.4s, v2.s[2]
	fmla	v22.4s, v15.4s, v2.s[2]
	fmla	v24.4s, v11.4s, v30.s[2]
	fmla	v23.4s, v15.4s, v30.s[2]
	fmla	v29.4s, v11.4s, v31.s[2]
	fmla	v28.4s, v15.4s, v31.s[2]
	fmla	v4.4s, v9.4s, v1.s[3]
	fmla	v3.4s, v8.4s, v1.s[3]
	fmla	v6.4s, v9.4s, v14.s[3]
	fmla	v5.4s, v8.4s, v14.s[3]
	fmla	v16.4s, v9.4s, v26.s[3]
	fmla	v7.4s, v8.4s, v26.s[3]
	fmla	v18.4s, v9.4s, v27.s[3]
	fmla	v17.4s, v8.4s, v27.s[3]
	fmla	v19.4s, v9.4s, v25.s[3]
	fmla	v20.4s, v8.4s, v25.s[3]
	fmla	v22.4s, v9.4s, v2.s[3]
	fmla	v21.4s, v8.4s, v2.s[3]
	fmla	v23.4s, v9.4s, v30.s[3]
	fmla	v24.4s, v8.4s, v30.s[3]
	fmla	v28.4s, v9.4s, v31.s[3]
	fmla	v29.4s, v8.4s, v31.s[3]
	ldp	q12, q30, [x12, #32]
	ldp	q13, q2, [x12, #160]
	ldp	q11, q26, [x12, #288]
	ldp	q10, q27, [x12, #416]
	ldp	q9, q1, [x12, #544]
	ldp	q8, q0, [x12, #672]
	stp	q1, q0, [sp, #80]               // 32-byte Folded Spill
	ldr	q31, [x12, #800]
	ldr	q25, [x12, #928]
	ldr	q14, [x14, #1024]
	ldr	q15, [x14, #1040]
	fmla	v3.4s, v15.4s, v12.s[0]
	fmla	v5.4s, v15.4s, v13.s[0]
	fmla	v7.4s, v15.4s, v11.s[0]
	fmla	v17.4s, v15.4s, v10.s[0]
	fmla	v20.4s, v15.4s, v9.s[0]
	fmla	v21.4s, v15.4s, v8.s[0]
	fmla	v24.4s, v15.4s, v31.s[0]
	fmla	v29.4s, v15.4s, v25.s[0]
	ldr	q15, [x14, #1168]
	fmla	v4.4s, v14.4s, v12.s[0]
	fmla	v6.4s, v14.4s, v13.s[0]
	fmla	v16.4s, v14.4s, v11.s[0]
	fmla	v18.4s, v14.4s, v10.s[0]
	fmla	v19.4s, v14.4s, v9.s[0]
	fmla	v22.4s, v14.4s, v8.s[0]
	fmla	v23.4s, v14.4s, v31.s[0]
	fmla	v28.4s, v14.4s, v25.s[0]
	ldr	q14, [x14, #1152]
	fmla	v4.4s, v14.4s, v12.s[1]
	fmla	v6.4s, v14.4s, v13.s[1]
	fmla	v16.4s, v14.4s, v11.s[1]
	fmla	v18.4s, v14.4s, v10.s[1]
	fmla	v19.4s, v14.4s, v9.s[1]
	fmla	v22.4s, v14.4s, v8.s[1]
	fmla	v23.4s, v14.4s, v31.s[1]
	fmla	v28.4s, v14.4s, v25.s[1]
	ldr	q14, [x14, #1280]
	fmla	v3.4s, v15.4s, v12.s[1]
	fmla	v5.4s, v15.4s, v13.s[1]
	fmla	v7.4s, v15.4s, v11.s[1]
	fmla	v17.4s, v15.4s, v10.s[1]
	fmla	v20.4s, v15.4s, v9.s[1]
	fmla	v21.4s, v15.4s, v8.s[1]
	fmla	v24.4s, v15.4s, v31.s[1]
	fmla	v29.4s, v15.4s, v25.s[1]
	ldr	q15, [x14, #1296]
	fmla	v3.4s, v15.4s, v12.s[2]
	fmla	v5.4s, v15.4s, v13.s[2]
	fmla	v7.4s, v15.4s, v11.s[2]
	fmla	v17.4s, v15.4s, v10.s[2]
	fmla	v20.4s, v15.4s, v9.s[2]
	fmla	v21.4s, v15.4s, v8.s[2]
	fmla	v24.4s, v15.4s, v31.s[2]
	fmla	v29.4s, v15.4s, v25.s[2]
	ldr	q15, [x14, #1424]
	fmla	v4.4s, v14.4s, v12.s[2]
	fmla	v6.4s, v14.4s, v13.s[2]
	fmla	v16.4s, v14.4s, v11.s[2]
	fmla	v18.4s, v14.4s, v10.s[2]
	fmla	v19.4s, v14.4s, v9.s[2]
	fmla	v22.4s, v14.4s, v8.s[2]
	fmla	v23.4s, v14.4s, v31.s[2]
	fmla	v28.4s, v14.4s, v25.s[2]
	ldr	q14, [x14, #1408]
	fmla	v4.4s, v14.4s, v12.s[3]
	fmla	v3.4s, v15.4s, v12.s[3]
	ldr	q12, [x12, #816]
	fmla	v6.4s, v14.4s, v13.s[3]
	fmla	v5.4s, v15.4s, v13.s[3]
	ldr	q13, [x12, #944]
	fmla	v16.4s, v14.4s, v11.s[3]
	fmla	v7.4s, v15.4s, v11.s[3]
	ldr	q1, [x14, #1536]
	fmla	v18.4s, v14.4s, v10.s[3]
	fmla	v17.4s, v15.4s, v10.s[3]
	ldr	q0, [x14, #1552]
	fmla	v19.4s, v14.4s, v9.s[3]
	fmla	v20.4s, v15.4s, v9.s[3]
	ldr	q10, [x14, #1680]
	fmla	v22.4s, v14.4s, v8.s[3]
	fmla	v21.4s, v15.4s, v8.s[3]
	ldr	q11, [x14, #1664]
	fmla	v23.4s, v14.4s, v31.s[3]
	fmla	v24.4s, v15.4s, v31.s[3]
	ldr	q8, [x14, #1792]
	fmla	v28.4s, v14.4s, v25.s[3]
	ldr	q9, [x14, #1808]
	fmla	v29.4s, v15.4s, v25.s[3]
	ldr	q25, [x14, #1936]
	ldr	q31, [x14, #1920]
	fmla	v3.4s, v0.4s, v30.s[0]
	fmla	v4.4s, v1.4s, v30.s[0]
	fmla	v5.4s, v0.4s, v2.s[0]
	fmla	v6.4s, v1.4s, v2.s[0]
	fmla	v7.4s, v0.4s, v26.s[0]
	fmla	v16.4s, v1.4s, v26.s[0]
	fmla	v17.4s, v0.4s, v27.s[0]
	fmla	v18.4s, v1.4s, v27.s[0]
	ldp	q15, q14, [sp, #80]             // 32-byte Folded Reload
	fmla	v20.4s, v0.4s, v15.s[0]
	fmla	v19.4s, v1.4s, v15.s[0]
	fmla	v21.4s, v0.4s, v14.s[0]
	fmla	v22.4s, v1.4s, v14.s[0]
	fmla	v24.4s, v0.4s, v12.s[0]
	fmla	v23.4s, v1.4s, v12.s[0]
	fmla	v29.4s, v0.4s, v13.s[0]
	fmla	v28.4s, v1.4s, v13.s[0]
	fmla	v4.4s, v11.4s, v30.s[1]
	fmla	v3.4s, v10.4s, v30.s[1]
	fmla	v6.4s, v11.4s, v2.s[1]
	fmla	v5.4s, v10.4s, v2.s[1]
	fmla	v16.4s, v11.4s, v26.s[1]
	fmla	v7.4s, v10.4s, v26.s[1]
	fmla	v18.4s, v11.4s, v27.s[1]
	fmla	v17.4s, v10.4s, v27.s[1]
	fmla	v19.4s, v11.4s, v15.s[1]
	fmla	v20.4s, v10.4s, v15.s[1]
	fmla	v22.4s, v11.4s, v14.s[1]
	fmla	v21.4s, v10.4s, v14.s[1]
	fmla	v23.4s, v11.4s, v12.s[1]
	fmla	v24.4s, v10.4s, v12.s[1]
	fmla	v28.4s, v11.4s, v13.s[1]
	fmla	v29.4s, v10.4s, v13.s[1]
	fmla	v3.4s, v9.4s, v30.s[2]
	fmla	v4.4s, v8.4s, v30.s[2]
	fmla	v5.4s, v9.4s, v2.s[2]
	fmla	v6.4s, v8.4s, v2.s[2]
	fmla	v7.4s, v9.4s, v26.s[2]
	fmla	v16.4s, v8.4s, v26.s[2]
	fmla	v17.4s, v9.4s, v27.s[2]
	fmla	v18.4s, v8.4s, v27.s[2]
	fmla	v20.4s, v9.4s, v15.s[2]
	fmla	v19.4s, v8.4s, v15.s[2]
	fmla	v21.4s, v9.4s, v14.s[2]
	fmla	v22.4s, v8.4s, v14.s[2]
	fmla	v24.4s, v9.4s, v12.s[2]
	fmla	v23.4s, v8.4s, v12.s[2]
	fmla	v29.4s, v9.4s, v13.s[2]
	fmla	v28.4s, v8.4s, v13.s[2]
	fmla	v4.4s, v31.4s, v30.s[3]
	fmla	v3.4s, v25.4s, v30.s[3]
	fmla	v6.4s, v31.4s, v2.s[3]
	fmla	v5.4s, v25.4s, v2.s[3]
	fmla	v16.4s, v31.4s, v26.s[3]
	fmla	v7.4s, v25.4s, v26.s[3]
	fmla	v18.4s, v31.4s, v27.s[3]
	fmla	v17.4s, v25.4s, v27.s[3]
	fmla	v19.4s, v31.4s, v15.s[3]
	fmla	v20.4s, v25.4s, v15.s[3]
	fmla	v22.4s, v31.4s, v14.s[3]
	fmla	v21.4s, v25.4s, v14.s[3]
	fmla	v23.4s, v31.4s, v12.s[3]
	fmla	v24.4s, v25.4s, v12.s[3]
	fmla	v28.4s, v31.4s, v13.s[3]
	fmla	v29.4s, v25.4s, v13.s[3]
	ldp	q12, q8, [x12, #64]
	ldp	q13, q31, [x12, #192]
	ldp	q11, q30, [x12, #320]
	ldp	q10, q2, [x12, #448]
	ldp	q9, q1, [x12, #576]
	ldp	q26, q27, [x12, #704]
	ldr	q25, [x12, #832]
	ldr	q0, [x12, #960]
	ldr	q14, [x14, #2048]
	ldr	q15, [x14, #2064]
	fmla	v3.4s, v15.4s, v12.s[0]
	fmla	v5.4s, v15.4s, v13.s[0]
	fmla	v7.4s, v15.4s, v11.s[0]
	fmla	v17.4s, v15.4s, v10.s[0]
	fmla	v20.4s, v15.4s, v9.s[0]
	fmla	v21.4s, v15.4s, v26.s[0]
	fmla	v24.4s, v15.4s, v25.s[0]
	fmla	v29.4s, v15.4s, v0.s[0]
	ldr	q15, [x14, #2192]
	fmla	v4.4s, v14.4s, v12.s[0]
	fmla	v6.4s, v14.4s, v13.s[0]
	fmla	v16.4s, v14.4s, v11.s[0]
	fmla	v18.4s, v14.4s, v10.s[0]
	fmla	v19.4s, v14.4s, v9.s[0]
	fmla	v22.4s, v14.4s, v26.s[0]
	fmla	v23.4s, v14.4s, v25.s[0]
	fmla	v28.4s, v14.4s, v0.s[0]
	ldr	q14, [x14, #2176]
	fmla	v4.4s, v14.4s, v12.s[1]
	fmla	v6.4s, v14.4s, v13.s[1]
	fmla	v16.4s, v14.4s, v11.s[1]
	fmla	v18.4s, v14.4s, v10.s[1]
	fmla	v19.4s, v14.4s, v9.s[1]
	fmla	v22.4s, v14.4s, v26.s[1]
	fmla	v23.4s, v14.4s, v25.s[1]
	fmla	v28.4s, v14.4s, v0.s[1]
	ldr	q14, [x14, #2304]
	fmla	v3.4s, v15.4s, v12.s[1]
	fmla	v5.4s, v15.4s, v13.s[1]
	fmla	v7.4s, v15.4s, v11.s[1]
	fmla	v17.4s, v15.4s, v10.s[1]
	fmla	v20.4s, v15.4s, v9.s[1]
	fmla	v21.4s, v15.4s, v26.s[1]
	fmla	v24.4s, v15.4s, v25.s[1]
	fmla	v29.4s, v15.4s, v0.s[1]
	ldr	q15, [x14, #2320]
	fmla	v3.4s, v15.4s, v12.s[2]
	fmla	v5.4s, v15.4s, v13.s[2]
	fmla	v7.4s, v15.4s, v11.s[2]
	fmla	v17.4s, v15.4s, v10.s[2]
	fmla	v20.4s, v15.4s, v9.s[2]
	fmla	v21.4s, v15.4s, v26.s[2]
	fmla	v24.4s, v15.4s, v25.s[2]
	fmla	v29.4s, v15.4s, v0.s[2]
	ldr	q15, [x14, #2448]
	fmla	v4.4s, v14.4s, v12.s[2]
	fmla	v6.4s, v14.4s, v13.s[2]
	fmla	v16.4s, v14.4s, v11.s[2]
	fmla	v18.4s, v14.4s, v10.s[2]
	fmla	v19.4s, v14.4s, v9.s[2]
	fmla	v22.4s, v14.4s, v26.s[2]
	fmla	v23.4s, v14.4s, v25.s[2]
	fmla	v28.4s, v14.4s, v0.s[2]
	ldr	q14, [x14, #2432]
	fmla	v4.4s, v14.4s, v12.s[3]
	fmla	v3.4s, v15.4s, v12.s[3]
	ldr	q12, [x12, #848]
	fmla	v6.4s, v14.4s, v13.s[3]
	fmla	v5.4s, v15.4s, v13.s[3]
	ldr	q13, [x12, #976]
	str	q13, [sp, #96]                  // 16-byte Folded Spill
	fmla	v16.4s, v14.4s, v11.s[3]
	fmla	v7.4s, v15.4s, v11.s[3]
	ldr	q11, [x14, #2560]
	fmla	v18.4s, v14.4s, v10.s[3]
	fmla	v17.4s, v15.4s, v10.s[3]
	ldr	q13, [x14, #2576]
	fmla	v19.4s, v14.4s, v9.s[3]
	fmla	v20.4s, v15.4s, v9.s[3]
	ldr	q9, [x14, #2704]
	fmla	v22.4s, v14.4s, v26.s[3]
	fmla	v21.4s, v15.4s, v26.s[3]
	ldr	q10, [x14, #2688]
	fmla	v23.4s, v14.4s, v25.s[3]
	fmla	v24.4s, v15.4s, v25.s[3]
	ldr	q25, [x14, #2816]
	fmla	v28.4s, v14.4s, v0.s[3]
	ldr	q26, [x14, #2832]
	fmla	v29.4s, v15.4s, v0.s[3]
	ldr	q15, [x14, #2960]
	ldr	q0, [x14, #2944]
	fmla	v3.4s, v13.4s, v8.s[0]
	fmla	v4.4s, v11.4s, v8.s[0]
	fmla	v5.4s, v13.4s, v31.s[0]
	fmla	v6.4s, v11.4s, v31.s[0]
	fmla	v7.4s, v13.4s, v30.s[0]
	fmla	v16.4s, v11.4s, v30.s[0]
	fmla	v17.4s, v13.4s, v2.s[0]
	fmla	v18.4s, v11.4s, v2.s[0]
	fmla	v20.4s, v13.4s, v1.s[0]
	fmla	v19.4s, v11.4s, v1.s[0]
	fmla	v21.4s, v13.4s, v27.s[0]
	fmla	v22.4s, v11.4s, v27.s[0]
	fmla	v24.4s, v13.4s, v12.s[0]
	fmla	v23.4s, v11.4s, v12.s[0]
	ldr	q14, [sp, #96]                  // 16-byte Folded Reload
	fmla	v29.4s, v13.4s, v14.s[0]
	fmla	v28.4s, v11.4s, v14.s[0]
	fmla	v4.4s, v10.4s, v8.s[1]
	fmla	v3.4s, v9.4s, v8.s[1]
	fmla	v6.4s, v10.4s, v31.s[1]
	fmla	v5.4s, v9.4s, v31.s[1]
	fmla	v16.4s, v10.4s, v30.s[1]
	fmla	v7.4s, v9.4s, v30.s[1]
	fmla	v18.4s, v10.4s, v2.s[1]
	fmla	v17.4s, v9.4s, v2.s[1]
	fmla	v19.4s, v10.4s, v1.s[1]
	fmla	v20.4s, v9.4s, v1.s[1]
	fmla	v22.4s, v10.4s, v27.s[1]
	fmla	v21.4s, v9.4s, v27.s[1]
	fmla	v23.4s, v10.4s, v12.s[1]
	fmla	v24.4s, v9.4s, v12.s[1]
	fmla	v28.4s, v10.4s, v14.s[1]
	fmla	v29.4s, v9.4s, v14.s[1]
	fmla	v3.4s, v26.4s, v8.s[2]
	fmla	v4.4s, v25.4s, v8.s[2]
	fmla	v5.4s, v26.4s, v31.s[2]
	fmla	v6.4s, v25.4s, v31.s[2]
	fmla	v7.4s, v26.4s, v30.s[2]
	fmla	v16.4s, v25.4s, v30.s[2]
	fmla	v17.4s, v26.4s, v2.s[2]
	fmla	v18.4s, v25.4s, v2.s[2]
	fmla	v20.4s, v26.4s, v1.s[2]
	fmla	v19.4s, v25.4s, v1.s[2]
	fmla	v21.4s, v26.4s, v27.s[2]
	fmla	v22.4s, v25.4s, v27.s[2]
	fmla	v24.4s, v26.4s, v12.s[2]
	fmla	v23.4s, v25.4s, v12.s[2]
	fmla	v29.4s, v26.4s, v14.s[2]
	fmla	v28.4s, v25.4s, v14.s[2]
	fmla	v4.4s, v0.4s, v8.s[3]
	str	q15, [sp, #80]                  // 16-byte Folded Spill
	fmla	v3.4s, v15.4s, v8.s[3]
	fmla	v6.4s, v0.4s, v31.s[3]
	fmla	v5.4s, v15.4s, v31.s[3]
	fmla	v16.4s, v0.4s, v30.s[3]
	fmla	v7.4s, v15.4s, v30.s[3]
	fmla	v18.4s, v0.4s, v2.s[3]
	fmla	v17.4s, v15.4s, v2.s[3]
	fmla	v19.4s, v0.4s, v1.s[3]
	fmla	v20.4s, v15.4s, v1.s[3]
	fmla	v22.4s, v0.4s, v27.s[3]
	fmla	v21.4s, v15.4s, v27.s[3]
	fmla	v23.4s, v0.4s, v12.s[3]
	fmla	v24.4s, v15.4s, v12.s[3]
	fmla	v28.4s, v0.4s, v14.s[3]
	ldp	q11, q14, [x12, #96]
	ldr	q27, [x12, #224]
	ldr	q26, [x12, #352]
	ldr	q25, [x12, #480]
	ldr	q0, [x12, #608]
	ldr	q8, [x14, #3072]
	ldr	q10, [x14, #3088]
	ldr	q31, [x14, #3216]
	ldr	q9, [x14, #3200]
	ldr	q2, [x14, #3328]
	ldr	q12, [x14, #3344]
	ldr	q1, [x14, #3472]
	ldr	q30, [x14, #3456]
	fmla	v3.4s, v10.4s, v11.s[0]
	fmla	v4.4s, v8.4s, v11.s[0]
	fmla	v4.4s, v9.4s, v11.s[1]
	fmla	v3.4s, v31.4s, v11.s[1]
	fmla	v3.4s, v12.4s, v11.s[2]
	fmla	v4.4s, v2.4s, v11.s[2]
	fmla	v4.4s, v30.4s, v11.s[3]
	fmla	v3.4s, v1.4s, v11.s[3]
	ldr	q15, [x12, #736]
	fmla	v5.4s, v10.4s, v27.s[0]
	mov	v11.16b, v8.16b
	fmla	v6.4s, v8.4s, v27.s[0]
	fmla	v6.4s, v9.4s, v27.s[1]
	fmla	v5.4s, v31.4s, v27.s[1]
	fmla	v5.4s, v12.4s, v27.s[2]
	fmla	v6.4s, v2.4s, v27.s[2]
	fmla	v6.4s, v30.4s, v27.s[3]
	fmla	v5.4s, v1.4s, v27.s[3]
	mov	v8.16b, v1.16b
	ldr	q27, [x12, #864]
	fmla	v7.4s, v10.4s, v26.s[0]
	fmla	v16.4s, v11.4s, v26.s[0]
	mov	v1.16b, v11.16b
	fmla	v16.4s, v9.4s, v26.s[1]
	fmla	v7.4s, v31.4s, v26.s[1]
	fmla	v7.4s, v12.4s, v26.s[2]
	fmla	v16.4s, v2.4s, v26.s[2]
	fmla	v16.4s, v30.4s, v26.s[3]
	fmla	v7.4s, v8.4s, v26.s[3]
	ldr	q11, [x14, #3584]
	fmla	v17.4s, v10.4s, v25.s[0]
	fmla	v18.4s, v1.4s, v25.s[0]
	fmla	v18.4s, v9.4s, v25.s[1]
	fmla	v17.4s, v31.4s, v25.s[1]
	fmla	v17.4s, v12.4s, v25.s[2]
	mov	v26.16b, v12.16b
	fmla	v18.4s, v2.4s, v25.s[2]
	fmla	v18.4s, v30.4s, v25.s[3]
	fmla	v17.4s, v8.4s, v25.s[3]
	ldr	q12, [x14, #3600]
	fmla	v20.4s, v10.4s, v0.s[0]
	fmla	v19.4s, v1.4s, v0.s[0]
	fmla	v19.4s, v9.4s, v0.s[1]
	fmla	v20.4s, v31.4s, v0.s[1]
	fmla	v20.4s, v26.4s, v0.s[2]
	fmla	v19.4s, v2.4s, v0.s[2]
	fmla	v19.4s, v30.4s, v0.s[3]
	stp	q26, q30, [sp, #32]             // 32-byte Folded Spill
	fmla	v20.4s, v8.4s, v0.s[3]
	ldr	q13, [x14, #3728]
	fmla	v21.4s, v10.4s, v15.s[0]
	stp	q1, q31, [sp]                   // 32-byte Folded Spill
	fmla	v22.4s, v1.4s, v15.s[0]
	fmla	v22.4s, v9.4s, v15.s[1]
	fmla	v21.4s, v31.4s, v15.s[1]
	fmla	v21.4s, v26.4s, v15.s[2]
	str	q2, [sp, #64]                   // 16-byte Folded Spill
	fmla	v22.4s, v2.4s, v15.s[2]
	fmla	v22.4s, v30.4s, v15.s[3]
	fmla	v21.4s, v8.4s, v15.s[3]
	ldr	q15, [x14, #3712]
	fmla	v24.4s, v10.4s, v27.s[0]
	fmla	v23.4s, v1.4s, v27.s[0]
	fmla	v23.4s, v9.4s, v27.s[1]
	fmla	v24.4s, v31.4s, v27.s[1]
	fmla	v24.4s, v26.4s, v27.s[2]
	fmla	v23.4s, v2.4s, v27.s[2]
	fmla	v23.4s, v30.4s, v27.s[3]
	fmla	v24.4s, v8.4s, v27.s[3]
	ldr	q30, [x14, #3856]
	fmla	v3.4s, v12.4s, v14.s[0]
	fmla	v4.4s, v11.4s, v14.s[0]
	fmla	v4.4s, v15.4s, v14.s[1]
	fmla	v3.4s, v13.4s, v14.s[1]
	fmla	v3.4s, v30.4s, v14.s[2]
	ldr	q26, [x14, #3840]
	fmla	v4.4s, v26.4s, v14.s[2]
	ldr	q27, [x14, #3968]
	fmla	v4.4s, v27.4s, v14.s[3]
	ldr	q25, [x14, #3984]
	fmla	v3.4s, v25.4s, v14.s[3]
	ldr	q14, [x12, #240]
	fmla	v5.4s, v12.4s, v14.s[0]
	fmla	v6.4s, v11.4s, v14.s[0]
	fmla	v6.4s, v15.4s, v14.s[1]
	fmla	v5.4s, v13.4s, v14.s[1]
	fmla	v5.4s, v30.4s, v14.s[2]
	fmla	v6.4s, v26.4s, v14.s[2]
	fmla	v6.4s, v27.4s, v14.s[3]
	fmla	v5.4s, v25.4s, v14.s[3]
	ldr	q14, [x12, #368]
	fmla	v7.4s, v12.4s, v14.s[0]
	fmla	v16.4s, v11.4s, v14.s[0]
	fmla	v16.4s, v15.4s, v14.s[1]
	fmla	v7.4s, v13.4s, v14.s[1]
	fmla	v7.4s, v30.4s, v14.s[2]
	fmla	v16.4s, v26.4s, v14.s[2]
	fmla	v16.4s, v27.4s, v14.s[3]
	fmla	v7.4s, v25.4s, v14.s[3]
	ldr	q14, [x12, #496]
	fmla	v17.4s, v12.4s, v14.s[0]
	fmla	v18.4s, v11.4s, v14.s[0]
	fmla	v18.4s, v15.4s, v14.s[1]
	fmla	v17.4s, v13.4s, v14.s[1]
	fmla	v17.4s, v30.4s, v14.s[2]
	fmla	v18.4s, v26.4s, v14.s[2]
	fmla	v18.4s, v27.4s, v14.s[3]
	fmla	v17.4s, v25.4s, v14.s[3]
	ldr	q14, [x12, #624]
	fmla	v20.4s, v12.4s, v14.s[0]
	fmla	v19.4s, v11.4s, v14.s[0]
	fmla	v19.4s, v15.4s, v14.s[1]
	fmla	v20.4s, v13.4s, v14.s[1]
	fmla	v20.4s, v30.4s, v14.s[2]
	fmla	v19.4s, v26.4s, v14.s[2]
	fmla	v19.4s, v27.4s, v14.s[3]
	fmla	v20.4s, v25.4s, v14.s[3]
	ldr	q14, [x12, #752]
	fmla	v21.4s, v12.4s, v14.s[0]
	fmla	v22.4s, v11.4s, v14.s[0]
	fmla	v22.4s, v15.4s, v14.s[1]
	fmla	v21.4s, v13.4s, v14.s[1]
	fmla	v21.4s, v30.4s, v14.s[2]
	fmla	v22.4s, v26.4s, v14.s[2]
	fmla	v22.4s, v27.4s, v14.s[3]
	fmla	v21.4s, v25.4s, v14.s[3]
	ldr	q14, [x12, #880]
	fmla	v24.4s, v12.4s, v14.s[0]
	fmla	v23.4s, v11.4s, v14.s[0]
	fmla	v23.4s, v15.4s, v14.s[1]
	fmla	v24.4s, v13.4s, v14.s[1]
	fmla	v24.4s, v30.4s, v14.s[2]
	fmla	v23.4s, v26.4s, v14.s[2]
	fmla	v23.4s, v27.4s, v14.s[3]
	fmla	v24.4s, v25.4s, v14.s[3]
	add	x14, x13, x10
	ldp	q14, q0, [x14]
	fadd	v3.4s, v3.4s, v0.4s
	fadd	v4.4s, v4.4s, v14.4s
	ldp	q14, q0, [x14, #128]
	fadd	v31.4s, v5.4s, v0.4s
	fadd	v6.4s, v6.4s, v14.4s
	ldp	q14, q0, [x14, #256]
	fadd	v7.4s, v7.4s, v0.4s
	fadd	v16.4s, v16.4s, v14.4s
	ldp	q14, q0, [x14, #384]
	fadd	v17.4s, v17.4s, v0.4s
	fadd	v18.4s, v18.4s, v14.4s
	ldp	q14, q0, [x14, #512]
	fadd	v20.4s, v20.4s, v0.4s
	fadd	v19.4s, v19.4s, v14.4s
	ldp	q14, q0, [x14, #640]
	fadd	v21.4s, v21.4s, v0.4s
	fadd	v22.4s, v22.4s, v14.4s
	ldp	q14, q0, [x14, #768]
	fadd	v24.4s, v24.4s, v0.4s
	fadd	v23.4s, v23.4s, v14.4s
	add	x15, x1, x8
	add	x15, x15, x10
	movi	v2.2d, #0000000000000000
	fmax	v4.4s, v4.4s, v2.4s
	fmax	v3.4s, v3.4s, v2.4s
	ldp	q14, q5, [x14, #896]
	ldp	q0, q1, [x12, #992]
	stp	q4, q3, [x15]
	ldp	q3, q4, [sp, #80]               // 32-byte Folded Reload
	fmla	v29.4s, v3.4s, v4.s[3]
	fmax	v3.4s, v6.4s, v2.4s
	fmla	v29.4s, v10.4s, v0.s[0]
	fmax	v4.4s, v31.4s, v2.4s
	stp	q3, q4, [x15, #128]
	ldr	q3, [sp]                        // 16-byte Folded Reload
	fmla	v28.4s, v3.4s, v0.s[0]
	fmax	v3.4s, v16.4s, v2.4s
	fmla	v28.4s, v9.4s, v0.s[1]
	fmax	v4.4s, v7.4s, v2.4s
	stp	q3, q4, [x15, #256]
	ldp	q3, q4, [sp, #16]               // 32-byte Folded Reload
	fmla	v29.4s, v3.4s, v0.s[1]
	fmax	v3.4s, v18.4s, v2.4s
	fmla	v29.4s, v4.4s, v0.s[2]
	fmax	v4.4s, v17.4s, v2.4s
	stp	q3, q4, [x15, #384]
	ldp	q4, q3, [sp, #48]               // 32-byte Folded Reload
	fmla	v28.4s, v3.4s, v0.s[2]
	fmax	v3.4s, v19.4s, v2.4s
	fmla	v28.4s, v4.4s, v0.s[3]
	fmax	v4.4s, v20.4s, v2.4s
	stp	q3, q4, [x15, #512]
	fmla	v29.4s, v8.4s, v0.s[3]
	fmax	v0.4s, v22.4s, v2.4s
	fmla	v29.4s, v12.4s, v1.s[0]
	fmax	v3.4s, v21.4s, v2.4s
	stp	q0, q3, [x15, #640]
	fmla	v28.4s, v11.4s, v1.s[0]
	fmax	v0.4s, v23.4s, v2.4s
	fmla	v28.4s, v15.4s, v1.s[1]
	fmax	v3.4s, v24.4s, v2.4s
	stp	q0, q3, [x15, #768]
	fmla	v29.4s, v13.4s, v1.s[1]
	fmla	v29.4s, v30.4s, v1.s[2]
	fmla	v28.4s, v26.4s, v1.s[2]
	fmla	v29.4s, v25.4s, v1.s[3]
	fadd	v0.4s, v29.4s, v5.4s
	fmla	v28.4s, v27.4s, v1.s[3]
	fadd	v1.4s, v28.4s, v14.4s
	fmax	v1.4s, v1.4s, v2.4s
	fmax	v0.4s, v0.4s, v2.4s
	stp	q1, q0, [x15, #896]
	add	x11, x11, #8
	add	x10, x10, #32
	cmp	x11, #31
	b.le	.LBB0_4
	b	.LBB0_1
.LBB0_5:
	mov	x2, xzr
	mov	w3, #32                         // =0x20
	mov	w4, #32                         // =0x20
	mov	w5, #32                         // =0x20
	mov	w6, #1                          // =0x1
	ldp	x20, x19, [sp, #208]            // 16-byte Folded Reload
	ldp	x30, x21, [sp, #192]            // 16-byte Folded Reload
	ldr	x29, [sp, #176]                 // 8-byte Folded Reload
	ldp	d9, d8, [sp, #160]              // 16-byte Folded Reload
	ldp	d11, d10, [sp, #144]            // 16-byte Folded Reload
	ldp	d13, d12, [sp, #128]            // 16-byte Folded Reload
	ldp	d15, d14, [sp, #112]            // 16-byte Folded Reload
	add	sp, sp, #224
	ret
.Lfunc_end0:
	.size	matmul_bias_relu_tiled_32x32x32, .Lfunc_end0-matmul_bias_relu_tiled_32x32x32
	.cfi_endproc
                                        // -- End function
	.globl	_mlir_ciface_matmul_bias_relu_tiled_32x32x32 // -- Begin function _mlir_ciface_matmul_bias_relu_tiled_32x32x32
	.p2align	4
	.type	_mlir_ciface_matmul_bias_relu_tiled_32x32x32,@function
_mlir_ciface_matmul_bias_relu_tiled_32x32x32: // @_mlir_ciface_matmul_bias_relu_tiled_32x32x32
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
	bl	matmul_bias_relu_tiled_32x32x32
	stp	x0, x1, [x19]
	stp	x2, x3, [x19, #16]
	stp	x4, x5, [x19, #32]
	str	x6, [x19, #48]
	ldp	x30, x19, [sp, #112]            // 16-byte Folded Reload
	add	sp, sp, #128
	ret
.Lfunc_end1:
	.size	_mlir_ciface_matmul_bias_relu_tiled_32x32x32, .Lfunc_end1-_mlir_ciface_matmul_bias_relu_tiled_32x32x32
	.cfi_endproc
                                        // -- End function
	.section	".note.GNU-stack","",@progbits
