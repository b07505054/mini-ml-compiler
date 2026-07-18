	.file	"LLVMDialectModule"
	.text
	.globl	matmul_bias_relu_tiled_16x64x32 // -- Begin function matmul_bias_relu_tiled_16x64x32
	.p2align	4
	.type	matmul_bias_relu_tiled_16x64x32,@function
matmul_bias_relu_tiled_16x64x32:        // @matmul_bias_relu_tiled_16x64x32
	.cfi_startproc
// %bb.0:
	sub	sp, sp, #368
	stp	d15, d14, [sp, #256]            // 16-byte Folded Spill
	stp	d13, d12, [sp, #272]            // 16-byte Folded Spill
	stp	d11, d10, [sp, #288]            // 16-byte Folded Spill
	stp	d9, d8, [sp, #304]              // 16-byte Folded Spill
	str	x29, [sp, #320]                 // 8-byte Folded Spill
	stp	x30, x21, [sp, #336]            // 16-byte Folded Spill
	stp	x20, x19, [sp, #352]            // 16-byte Folded Spill
	.cfi_def_cfa_offset 368
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
	ldr	x20, [sp, #424]
	ldr	x21, [sp, #368]
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
	add	x8, x8, #2048
.LBB0_2:                                // =>This Loop Header: Depth=1
                                        //     Child Loop BB0_4 Depth 2
	cmp	x9, #15
	b.gt	.LBB0_5
// %bb.3:                               // %.preheader
                                        //   in Loop: Header=BB0_2 Depth=1
	mov	x10, xzr
	mov	x11, xzr
	add	x12, x19, x9, lsl #7
	add	x13, x20, x8
	cmp	x11, #63
	b.gt	.LBB0_1
	.p2align	5, , 16
.LBB0_4:                                //   Parent Loop BB0_2 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	movi	v15.2d, #0000000000000000
	ldp	q2, q1, [x12]
	add	x14, x21, x10
	ldp	q30, q31, [x14]
	fmla	v15.4s, v31.4s, v2.s[0]
	movi	v14.2d, #0000000000000000
	fmla	v14.4s, v30.4s, v2.s[0]
	movi	v11.2d, #0000000000000000
	ldp	q4, q0, [x12, #128]
	stp	q0, q1, [sp, #224]              // 32-byte Folded Spill
	fmla	v11.4s, v31.4s, v4.s[0]
	movi	v10.2d, #0000000000000000
	movi	v9.2d, #0000000000000000
	fmla	v10.4s, v30.4s, v4.s[0]
	ldp	q6, q25, [x12, #256]
	fmla	v9.4s, v31.4s, v6.s[0]
	movi	v26.2d, #0000000000000000
	fmla	v26.4s, v30.4s, v6.s[0]
	movi	v16.2d, #0000000000000000
	ldp	q8, q27, [x12, #384]
	fmla	v16.4s, v31.4s, v8.s[0]
	movi	v17.2d, #0000000000000000
	fmla	v17.4s, v30.4s, v8.s[0]
	ldp	q12, q28, [x12, #512]
	movi	v19.2d, #0000000000000000
	fmla	v19.4s, v31.4s, v12.s[0]
	movi	v18.2d, #0000000000000000
	ldp	q13, q29, [x12, #640]
	fmla	v18.4s, v30.4s, v12.s[0]
	ldr	q5, [x12, #768]
	movi	v20.2d, #0000000000000000
	ldr	q3, [x12, #896]
	fmla	v20.4s, v31.4s, v13.s[0]
	movi	v21.2d, #0000000000000000
	fmla	v21.4s, v30.4s, v13.s[0]
	movi	v23.2d, #0000000000000000
	fmla	v23.4s, v31.4s, v5.s[0]
	movi	v22.2d, #0000000000000000
	fmla	v22.4s, v30.4s, v5.s[0]
	movi	v24.2d, #0000000000000000
	fmla	v24.4s, v31.4s, v3.s[0]
	movi	v7.2d, #0000000000000000
	fmla	v7.4s, v30.4s, v3.s[0]
	ldp	q31, q30, [x14, #256]
	fmla	v14.4s, v31.4s, v2.s[1]
	fmla	v15.4s, v30.4s, v2.s[1]
	fmla	v10.4s, v31.4s, v4.s[1]
	fmla	v11.4s, v30.4s, v4.s[1]
	fmla	v26.4s, v31.4s, v6.s[1]
	fmla	v9.4s, v30.4s, v6.s[1]
	fmla	v17.4s, v31.4s, v8.s[1]
	fmla	v16.4s, v30.4s, v8.s[1]
	fmla	v18.4s, v31.4s, v12.s[1]
	fmla	v19.4s, v30.4s, v12.s[1]
	fmla	v21.4s, v31.4s, v13.s[1]
	fmla	v20.4s, v30.4s, v13.s[1]
	fmla	v22.4s, v31.4s, v5.s[1]
	fmla	v23.4s, v30.4s, v5.s[1]
	fmla	v7.4s, v31.4s, v3.s[1]
	fmla	v24.4s, v30.4s, v3.s[1]
	ldp	q30, q31, [x14, #512]
	fmla	v15.4s, v31.4s, v2.s[2]
	fmla	v14.4s, v30.4s, v2.s[2]
	fmla	v11.4s, v31.4s, v4.s[2]
	fmla	v10.4s, v30.4s, v4.s[2]
	fmla	v9.4s, v31.4s, v6.s[2]
	fmla	v26.4s, v30.4s, v6.s[2]
	fmla	v16.4s, v31.4s, v8.s[2]
	fmla	v17.4s, v30.4s, v8.s[2]
	fmla	v19.4s, v31.4s, v12.s[2]
	fmla	v18.4s, v30.4s, v12.s[2]
	fmla	v20.4s, v31.4s, v13.s[2]
	fmla	v21.4s, v30.4s, v13.s[2]
	fmla	v23.4s, v31.4s, v5.s[2]
	fmla	v22.4s, v30.4s, v5.s[2]
	fmla	v24.4s, v31.4s, v3.s[2]
	fmla	v7.4s, v30.4s, v3.s[2]
	ldp	q0, q1, [x14, #768]
	fmla	v14.4s, v0.4s, v2.s[3]
	fmla	v15.4s, v1.4s, v2.s[3]
	ldr	q30, [x12, #784]
	fmla	v10.4s, v0.4s, v4.s[3]
	fmla	v11.4s, v1.4s, v4.s[3]
	ldr	q31, [x12, #912]
	fmla	v26.4s, v0.4s, v6.s[3]
	fmla	v9.4s, v1.4s, v6.s[3]
	ldr	q2, [x14, #1024]
	fmla	v17.4s, v0.4s, v8.s[3]
	fmla	v16.4s, v1.4s, v8.s[3]
	ldr	q8, [x14, #1040]
	fmla	v18.4s, v0.4s, v12.s[3]
	fmla	v19.4s, v1.4s, v12.s[3]
	ldr	q6, [x14, #1296]
	fmla	v21.4s, v0.4s, v13.s[3]
	fmla	v20.4s, v1.4s, v13.s[3]
	ldr	q12, [x14, #1280]
	fmla	v22.4s, v0.4s, v5.s[3]
	fmla	v23.4s, v1.4s, v5.s[3]
	ldr	q4, [x14, #1536]
	fmla	v7.4s, v0.4s, v3.s[3]
	ldr	q5, [x14, #1552]
	fmla	v24.4s, v1.4s, v3.s[3]
	ldr	q1, [x14, #1808]
	ldr	q0, [x14, #1792]
	ldp	q13, q3, [sp, #224]             // 32-byte Folded Reload
	fmla	v15.4s, v8.4s, v3.s[0]
	fmla	v14.4s, v2.4s, v3.s[0]
	fmla	v11.4s, v8.4s, v13.s[0]
	fmla	v10.4s, v2.4s, v13.s[0]
	fmla	v9.4s, v8.4s, v25.s[0]
	fmla	v26.4s, v2.4s, v25.s[0]
	fmla	v16.4s, v8.4s, v27.s[0]
	fmla	v17.4s, v2.4s, v27.s[0]
	fmla	v19.4s, v8.4s, v28.s[0]
	fmla	v18.4s, v2.4s, v28.s[0]
	fmla	v20.4s, v8.4s, v29.s[0]
	fmla	v21.4s, v2.4s, v29.s[0]
	fmla	v23.4s, v8.4s, v30.s[0]
	fmla	v22.4s, v2.4s, v30.s[0]
	fmla	v24.4s, v8.4s, v31.s[0]
	fmla	v7.4s, v2.4s, v31.s[0]
	fmla	v14.4s, v12.4s, v3.s[1]
	fmla	v15.4s, v6.4s, v3.s[1]
	fmla	v10.4s, v12.4s, v13.s[1]
	fmla	v11.4s, v6.4s, v13.s[1]
	fmla	v26.4s, v12.4s, v25.s[1]
	fmla	v9.4s, v6.4s, v25.s[1]
	fmla	v17.4s, v12.4s, v27.s[1]
	fmla	v16.4s, v6.4s, v27.s[1]
	fmla	v18.4s, v12.4s, v28.s[1]
	fmla	v19.4s, v6.4s, v28.s[1]
	fmla	v21.4s, v12.4s, v29.s[1]
	fmla	v20.4s, v6.4s, v29.s[1]
	fmla	v22.4s, v12.4s, v30.s[1]
	fmla	v23.4s, v6.4s, v30.s[1]
	fmla	v7.4s, v12.4s, v31.s[1]
	fmla	v24.4s, v6.4s, v31.s[1]
	fmla	v15.4s, v5.4s, v3.s[2]
	fmla	v14.4s, v4.4s, v3.s[2]
	fmla	v11.4s, v5.4s, v13.s[2]
	fmla	v10.4s, v4.4s, v13.s[2]
	fmla	v9.4s, v5.4s, v25.s[2]
	fmla	v26.4s, v4.4s, v25.s[2]
	fmla	v16.4s, v5.4s, v27.s[2]
	fmla	v17.4s, v4.4s, v27.s[2]
	fmla	v19.4s, v5.4s, v28.s[2]
	fmla	v18.4s, v4.4s, v28.s[2]
	fmla	v20.4s, v5.4s, v29.s[2]
	fmla	v21.4s, v4.4s, v29.s[2]
	fmla	v23.4s, v5.4s, v30.s[2]
	fmla	v22.4s, v4.4s, v30.s[2]
	fmla	v24.4s, v5.4s, v31.s[2]
	fmla	v7.4s, v4.4s, v31.s[2]
	fmla	v14.4s, v0.4s, v3.s[3]
	fmla	v15.4s, v1.4s, v3.s[3]
	fmla	v10.4s, v0.4s, v13.s[3]
	fmla	v11.4s, v1.4s, v13.s[3]
	fmla	v26.4s, v0.4s, v25.s[3]
	fmla	v9.4s, v1.4s, v25.s[3]
	fmla	v17.4s, v0.4s, v27.s[3]
	fmla	v16.4s, v1.4s, v27.s[3]
	fmla	v18.4s, v0.4s, v28.s[3]
	fmla	v19.4s, v1.4s, v28.s[3]
	fmla	v21.4s, v0.4s, v29.s[3]
	fmla	v20.4s, v1.4s, v29.s[3]
	fmla	v22.4s, v0.4s, v30.s[3]
	fmla	v23.4s, v1.4s, v30.s[3]
	fmla	v7.4s, v0.4s, v31.s[3]
	fmla	v24.4s, v1.4s, v31.s[3]
	ldp	q12, q30, [x12, #32]
	ldp	q13, q29, [x12, #160]
	ldp	q4, q27, [x12, #288]
	ldp	q3, q1, [x12, #416]
	ldp	q2, q25, [x12, #544]
	ldp	q8, q0, [x12, #672]
	ldr	q31, [x12, #800]
	ldr	q28, [x12, #928]
	ldr	q5, [x14, #2048]
	ldr	q6, [x14, #2064]
	fmla	v15.4s, v6.4s, v12.s[0]
	fmla	v11.4s, v6.4s, v13.s[0]
	fmla	v9.4s, v6.4s, v4.s[0]
	fmla	v16.4s, v6.4s, v3.s[0]
	fmla	v19.4s, v6.4s, v2.s[0]
	fmla	v20.4s, v6.4s, v8.s[0]
	fmla	v23.4s, v6.4s, v31.s[0]
	fmla	v24.4s, v6.4s, v28.s[0]
	ldr	q6, [x14, #2320]
	fmla	v14.4s, v5.4s, v12.s[0]
	fmla	v10.4s, v5.4s, v13.s[0]
	fmla	v26.4s, v5.4s, v4.s[0]
	fmla	v17.4s, v5.4s, v3.s[0]
	fmla	v18.4s, v5.4s, v2.s[0]
	fmla	v21.4s, v5.4s, v8.s[0]
	fmla	v22.4s, v5.4s, v31.s[0]
	fmla	v7.4s, v5.4s, v28.s[0]
	ldr	q5, [x14, #2304]
	fmla	v14.4s, v5.4s, v12.s[1]
	fmla	v10.4s, v5.4s, v13.s[1]
	fmla	v26.4s, v5.4s, v4.s[1]
	fmla	v17.4s, v5.4s, v3.s[1]
	fmla	v18.4s, v5.4s, v2.s[1]
	fmla	v21.4s, v5.4s, v8.s[1]
	fmla	v22.4s, v5.4s, v31.s[1]
	fmla	v7.4s, v5.4s, v28.s[1]
	ldr	q5, [x14, #2560]
	fmla	v15.4s, v6.4s, v12.s[1]
	fmla	v11.4s, v6.4s, v13.s[1]
	fmla	v9.4s, v6.4s, v4.s[1]
	fmla	v16.4s, v6.4s, v3.s[1]
	fmla	v19.4s, v6.4s, v2.s[1]
	fmla	v20.4s, v6.4s, v8.s[1]
	fmla	v23.4s, v6.4s, v31.s[1]
	fmla	v24.4s, v6.4s, v28.s[1]
	ldr	q6, [x14, #2576]
	fmla	v15.4s, v6.4s, v12.s[2]
	fmla	v11.4s, v6.4s, v13.s[2]
	fmla	v9.4s, v6.4s, v4.s[2]
	fmla	v16.4s, v6.4s, v3.s[2]
	fmla	v19.4s, v6.4s, v2.s[2]
	fmla	v20.4s, v6.4s, v8.s[2]
	fmla	v23.4s, v6.4s, v31.s[2]
	fmla	v24.4s, v6.4s, v28.s[2]
	ldr	q6, [x14, #2832]
	fmla	v14.4s, v5.4s, v12.s[2]
	fmla	v10.4s, v5.4s, v13.s[2]
	fmla	v26.4s, v5.4s, v4.s[2]
	fmla	v17.4s, v5.4s, v3.s[2]
	fmla	v18.4s, v5.4s, v2.s[2]
	fmla	v21.4s, v5.4s, v8.s[2]
	fmla	v22.4s, v5.4s, v31.s[2]
	fmla	v7.4s, v5.4s, v28.s[2]
	ldr	q5, [x14, #2816]
	fmla	v14.4s, v5.4s, v12.s[3]
	fmla	v15.4s, v6.4s, v12.s[3]
	ldr	q12, [x12, #816]
	fmla	v10.4s, v5.4s, v13.s[3]
	fmla	v11.4s, v6.4s, v13.s[3]
	ldr	q13, [x12, #944]
	str	q13, [sp, #224]                 // 16-byte Folded Spill
	fmla	v26.4s, v5.4s, v4.s[3]
	fmla	v9.4s, v6.4s, v4.s[3]
	ldr	q4, [x14, #3072]
	fmla	v17.4s, v5.4s, v3.s[3]
	fmla	v16.4s, v6.4s, v3.s[3]
	ldr	q13, [x14, #3088]
	fmla	v18.4s, v5.4s, v2.s[3]
	fmla	v19.4s, v6.4s, v2.s[3]
	ldr	q3, [x14, #3344]
	fmla	v21.4s, v5.4s, v8.s[3]
	fmla	v20.4s, v6.4s, v8.s[3]
	ldr	q8, [x14, #3328]
	fmla	v22.4s, v5.4s, v31.s[3]
	fmla	v23.4s, v6.4s, v31.s[3]
	ldr	q2, [x14, #3584]
	str	q2, [sp, #240]                  // 16-byte Folded Spill
	mov	v31.16b, v7.16b
	fmla	v31.4s, v5.4s, v28.s[3]
	ldr	q2, [x14, #3600]
	fmla	v24.4s, v6.4s, v28.s[3]
	ldr	q28, [x14, #3856]
	ldr	q5, [x14, #3840]
	fmla	v15.4s, v13.4s, v30.s[0]
	fmla	v14.4s, v4.4s, v30.s[0]
	fmla	v11.4s, v13.4s, v29.s[0]
	fmla	v10.4s, v4.4s, v29.s[0]
	fmla	v9.4s, v13.4s, v27.s[0]
	fmla	v26.4s, v4.4s, v27.s[0]
	fmla	v16.4s, v13.4s, v1.s[0]
	fmla	v17.4s, v4.4s, v1.s[0]
	fmla	v19.4s, v13.4s, v25.s[0]
	fmla	v18.4s, v4.4s, v25.s[0]
	fmla	v20.4s, v13.4s, v0.s[0]
	fmla	v21.4s, v4.4s, v0.s[0]
	fmla	v23.4s, v13.4s, v12.s[0]
	fmla	v22.4s, v4.4s, v12.s[0]
	ldr	q7, [sp, #224]                  // 16-byte Folded Reload
	fmla	v24.4s, v13.4s, v7.s[0]
	fmla	v31.4s, v4.4s, v7.s[0]
	stp	q8, q31, [sp, #192]             // 32-byte Folded Spill
	fmla	v14.4s, v8.4s, v30.s[1]
	fmla	v15.4s, v3.4s, v30.s[1]
	fmla	v10.4s, v8.4s, v29.s[1]
	fmla	v11.4s, v3.4s, v29.s[1]
	fmla	v26.4s, v8.4s, v27.s[1]
	fmla	v9.4s, v3.4s, v27.s[1]
	fmla	v17.4s, v8.4s, v1.s[1]
	fmla	v16.4s, v3.4s, v1.s[1]
	fmla	v18.4s, v8.4s, v25.s[1]
	fmla	v19.4s, v3.4s, v25.s[1]
	fmla	v21.4s, v8.4s, v0.s[1]
	fmla	v20.4s, v3.4s, v0.s[1]
	fmla	v22.4s, v8.4s, v12.s[1]
	fmla	v23.4s, v3.4s, v12.s[1]
	fmla	v24.4s, v3.4s, v7.s[1]
	fmla	v15.4s, v2.4s, v30.s[2]
	ldr	q3, [sp, #240]                  // 16-byte Folded Reload
	fmla	v14.4s, v3.4s, v30.s[2]
	fmla	v11.4s, v2.4s, v29.s[2]
	fmla	v10.4s, v3.4s, v29.s[2]
	fmla	v9.4s, v2.4s, v27.s[2]
	fmla	v26.4s, v3.4s, v27.s[2]
	fmla	v16.4s, v2.4s, v1.s[2]
	fmla	v17.4s, v3.4s, v1.s[2]
	fmla	v19.4s, v2.4s, v25.s[2]
	fmla	v18.4s, v3.4s, v25.s[2]
	fmla	v20.4s, v2.4s, v0.s[2]
	fmla	v21.4s, v3.4s, v0.s[2]
	fmla	v23.4s, v2.4s, v12.s[2]
	fmla	v22.4s, v3.4s, v12.s[2]
	fmla	v24.4s, v2.4s, v7.s[2]
	str	q5, [sp, #176]                  // 16-byte Folded Spill
	fmla	v14.4s, v5.4s, v30.s[3]
	fmla	v15.4s, v28.4s, v30.s[3]
	fmla	v10.4s, v5.4s, v29.s[3]
	fmla	v11.4s, v28.4s, v29.s[3]
	fmla	v26.4s, v5.4s, v27.s[3]
	fmla	v9.4s, v28.4s, v27.s[3]
	fmla	v17.4s, v5.4s, v1.s[3]
	fmla	v16.4s, v28.4s, v1.s[3]
	fmla	v18.4s, v5.4s, v25.s[3]
	fmla	v19.4s, v28.4s, v25.s[3]
	fmla	v21.4s, v5.4s, v0.s[3]
	fmla	v20.4s, v28.4s, v0.s[3]
	fmla	v22.4s, v5.4s, v12.s[3]
	fmla	v23.4s, v28.4s, v12.s[3]
	fmla	v24.4s, v28.4s, v7.s[3]
	ldr	q30, [x12, #64]
	ldr	q27, [x12, #192]
	ldr	q28, [x12, #320]
	ldr	q29, [x12, #448]
	ldr	q1, [x12, #576]
	ldr	q25, [x12, #704]
	ldr	q0, [x12, #832]
	ldr	q7, [x12, #960]
	ldr	q6, [x14, #4096]
	ldr	q3, [x14, #4112]
	ldr	q2, [x14, #4368]
	ldr	q4, [x14, #4352]
	fmla	v15.4s, v3.4s, v30.s[0]
	fmla	v11.4s, v3.4s, v27.s[0]
	fmla	v9.4s, v3.4s, v28.s[0]
	fmla	v16.4s, v3.4s, v29.s[0]
	fmla	v19.4s, v3.4s, v1.s[0]
	fmla	v20.4s, v3.4s, v25.s[0]
	fmla	v23.4s, v3.4s, v0.s[0]
	fmla	v24.4s, v3.4s, v7.s[0]
	ldr	q3, [x14, #4608]
	fmla	v15.4s, v2.4s, v30.s[1]
	fmla	v11.4s, v2.4s, v27.s[1]
	fmla	v9.4s, v2.4s, v28.s[1]
	fmla	v16.4s, v2.4s, v29.s[1]
	fmla	v19.4s, v2.4s, v1.s[1]
	fmla	v20.4s, v2.4s, v25.s[1]
	fmla	v23.4s, v2.4s, v0.s[1]
	fmla	v24.4s, v2.4s, v7.s[1]
	ldr	q2, [x14, #4624]
	fmla	v15.4s, v2.4s, v30.s[2]
	fmla	v11.4s, v2.4s, v27.s[2]
	fmla	v9.4s, v2.4s, v28.s[2]
	fmla	v16.4s, v2.4s, v29.s[2]
	fmla	v19.4s, v2.4s, v1.s[2]
	fmla	v20.4s, v2.4s, v25.s[2]
	fmla	v23.4s, v2.4s, v0.s[2]
	fmla	v24.4s, v2.4s, v7.s[2]
	stp	q7, q6, [sp, #96]               // 32-byte Folded Spill
	ldr	q5, [x14, #4864]
	stp	q3, q5, [sp, #144]              // 32-byte Folded Spill
	fmla	v14.4s, v6.4s, v30.s[0]
	str	q4, [sp, #128]                  // 16-byte Folded Spill
	fmla	v14.4s, v4.4s, v30.s[1]
	fmla	v14.4s, v3.4s, v30.s[2]
	fmla	v14.4s, v5.4s, v30.s[3]
	ldr	q2, [x14, #4880]
	fmla	v15.4s, v2.4s, v30.s[3]
	fmla	v10.4s, v6.4s, v27.s[0]
	fmla	v10.4s, v4.4s, v27.s[1]
	fmla	v10.4s, v3.4s, v27.s[2]
	fmla	v10.4s, v5.4s, v27.s[3]
	fmla	v11.4s, v2.4s, v27.s[3]
	ldr	q27, [x12, #80]
	fmla	v26.4s, v6.4s, v28.s[0]
	fmla	v26.4s, v4.4s, v28.s[1]
	fmla	v26.4s, v3.4s, v28.s[2]
	fmla	v26.4s, v5.4s, v28.s[3]
	fmla	v9.4s, v2.4s, v28.s[3]
	ldr	q28, [x12, #208]
	fmla	v17.4s, v6.4s, v29.s[0]
	fmla	v17.4s, v4.4s, v29.s[1]
	fmla	v17.4s, v3.4s, v29.s[2]
	fmla	v17.4s, v5.4s, v29.s[3]
	fmla	v16.4s, v2.4s, v29.s[3]
	ldr	q29, [x12, #336]
	fmla	v18.4s, v6.4s, v1.s[0]
	fmla	v18.4s, v4.4s, v1.s[1]
	fmla	v18.4s, v3.4s, v1.s[2]
	fmla	v18.4s, v5.4s, v1.s[3]
	fmla	v19.4s, v2.4s, v1.s[3]
	ldr	q1, [x12, #464]
	fmla	v21.4s, v6.4s, v25.s[0]
	fmla	v21.4s, v4.4s, v25.s[1]
	fmla	v21.4s, v3.4s, v25.s[2]
	fmla	v21.4s, v5.4s, v25.s[3]
	fmla	v20.4s, v2.4s, v25.s[3]
	ldr	q25, [x12, #592]
	fmla	v22.4s, v6.4s, v0.s[0]
	fmla	v22.4s, v4.4s, v0.s[1]
	fmla	v22.4s, v3.4s, v0.s[2]
	fmla	v22.4s, v5.4s, v0.s[3]
	fmla	v23.4s, v2.4s, v0.s[3]
	ldr	q0, [x12, #720]
	fmla	v24.4s, v2.4s, v7.s[3]
	ldr	q2, [x14, #5136]
	fmla	v15.4s, v2.4s, v27.s[0]
	fmla	v11.4s, v2.4s, v28.s[0]
	fmla	v9.4s, v2.4s, v29.s[0]
	fmla	v16.4s, v2.4s, v1.s[0]
	fmla	v19.4s, v2.4s, v25.s[0]
	fmla	v20.4s, v2.4s, v0.s[0]
	ldr	q30, [x12, #848]
	fmla	v23.4s, v2.4s, v30.s[0]
	ldr	q7, [x12, #976]
	fmla	v24.4s, v2.4s, v7.s[0]
	ldr	q2, [x14, #5392]
	fmla	v15.4s, v2.4s, v27.s[1]
	fmla	v11.4s, v2.4s, v28.s[1]
	fmla	v9.4s, v2.4s, v29.s[1]
	fmla	v16.4s, v2.4s, v1.s[1]
	fmla	v19.4s, v2.4s, v25.s[1]
	fmla	v20.4s, v2.4s, v0.s[1]
	fmla	v23.4s, v2.4s, v30.s[1]
	fmla	v24.4s, v2.4s, v7.s[1]
	ldr	q2, [x14, #5648]
	fmla	v15.4s, v2.4s, v27.s[2]
	fmla	v11.4s, v2.4s, v28.s[2]
	fmla	v9.4s, v2.4s, v29.s[2]
	fmla	v16.4s, v2.4s, v1.s[2]
	fmla	v19.4s, v2.4s, v25.s[2]
	fmla	v20.4s, v2.4s, v0.s[2]
	fmla	v23.4s, v2.4s, v30.s[2]
	fmla	v24.4s, v2.4s, v7.s[2]
	ldr	q6, [x14, #5120]
	stp	q7, q6, [sp, #16]               // 32-byte Folded Spill
	fmla	v14.4s, v6.4s, v27.s[0]
	ldr	q5, [x14, #5376]
	fmla	v14.4s, v5.4s, v27.s[1]
	ldr	q4, [x14, #5632]
	stp	q5, q4, [sp, #48]               // 32-byte Folded Spill
	fmla	v14.4s, v4.4s, v27.s[2]
	ldr	q3, [x14, #5888]
	str	q3, [sp, #80]                   // 16-byte Folded Spill
	fmla	v14.4s, v3.4s, v27.s[3]
	ldr	q2, [x14, #5904]
	fmla	v15.4s, v2.4s, v27.s[3]
	fmla	v10.4s, v6.4s, v28.s[0]
	fmla	v10.4s, v5.4s, v28.s[1]
	fmla	v10.4s, v4.4s, v28.s[2]
	fmla	v10.4s, v3.4s, v28.s[3]
	fmla	v11.4s, v2.4s, v28.s[3]
	fmla	v26.4s, v6.4s, v29.s[0]
	fmla	v26.4s, v5.4s, v29.s[1]
	fmla	v26.4s, v4.4s, v29.s[2]
	fmla	v26.4s, v3.4s, v29.s[3]
	fmla	v9.4s, v2.4s, v29.s[3]
	fmla	v17.4s, v6.4s, v1.s[0]
	fmla	v17.4s, v5.4s, v1.s[1]
	fmla	v17.4s, v4.4s, v1.s[2]
	fmla	v17.4s, v3.4s, v1.s[3]
	fmla	v16.4s, v2.4s, v1.s[3]
	fmla	v18.4s, v6.4s, v25.s[0]
	fmla	v18.4s, v5.4s, v25.s[1]
	fmla	v18.4s, v4.4s, v25.s[2]
	fmla	v18.4s, v3.4s, v25.s[3]
	fmla	v19.4s, v2.4s, v25.s[3]
	fmla	v21.4s, v6.4s, v0.s[0]
	fmla	v21.4s, v5.4s, v0.s[1]
	fmla	v21.4s, v4.4s, v0.s[2]
	fmla	v21.4s, v3.4s, v0.s[3]
	fmla	v20.4s, v2.4s, v0.s[3]
	fmla	v22.4s, v6.4s, v30.s[0]
	fmla	v22.4s, v5.4s, v30.s[1]
	fmla	v22.4s, v4.4s, v30.s[2]
	fmla	v22.4s, v3.4s, v30.s[3]
	fmla	v23.4s, v2.4s, v30.s[3]
	fmla	v24.4s, v2.4s, v7.s[3]
	ldr	q0, [x12, #96]
	ldr	q6, [x14, #6160]
	fmla	v15.4s, v6.4s, v0.s[0]
	ldr	q5, [x12, #224]
	fmla	v11.4s, v6.4s, v5.s[0]
	ldr	q4, [x12, #352]
	fmla	v9.4s, v6.4s, v4.s[0]
	ldr	q3, [x12, #480]
	fmla	v16.4s, v6.4s, v3.s[0]
	ldr	q2, [x12, #608]
	fmla	v19.4s, v6.4s, v2.s[0]
	ldr	q25, [x12, #736]
	fmla	v20.4s, v6.4s, v25.s[0]
	ldr	q1, [x12, #864]
	fmla	v23.4s, v6.4s, v1.s[0]
	ldr	q7, [x12, #992]
	fmla	v24.4s, v6.4s, v7.s[0]
	ldr	q6, [x14, #6416]
	fmla	v15.4s, v6.4s, v0.s[1]
	fmla	v11.4s, v6.4s, v5.s[1]
	fmla	v9.4s, v6.4s, v4.s[1]
	fmla	v16.4s, v6.4s, v3.s[1]
	fmla	v19.4s, v6.4s, v2.s[1]
	fmla	v20.4s, v6.4s, v25.s[1]
	fmla	v23.4s, v6.4s, v1.s[1]
	fmla	v24.4s, v6.4s, v7.s[1]
	ldr	q6, [x14, #6672]
	fmla	v15.4s, v6.4s, v0.s[2]
	fmla	v11.4s, v6.4s, v5.s[2]
	fmla	v9.4s, v6.4s, v4.s[2]
	fmla	v16.4s, v6.4s, v3.s[2]
	fmla	v19.4s, v6.4s, v2.s[2]
	fmla	v20.4s, v6.4s, v25.s[2]
	fmla	v23.4s, v6.4s, v1.s[2]
	fmla	v24.4s, v6.4s, v7.s[2]
	str	q7, [sp]                        // 16-byte Folded Spill
	ldr	q31, [x14, #6144]
	fmla	v14.4s, v31.4s, v0.s[0]
	ldr	q8, [x14, #6400]
	fmla	v14.4s, v8.4s, v0.s[1]
	ldr	q12, [x14, #6656]
	fmla	v14.4s, v12.4s, v0.s[2]
	ldr	q13, [x14, #6912]
	fmla	v14.4s, v13.4s, v0.s[3]
	ldr	q6, [x14, #6928]
	fmla	v15.4s, v6.4s, v0.s[3]
	fmla	v10.4s, v31.4s, v5.s[0]
	fmla	v10.4s, v8.4s, v5.s[1]
	fmla	v10.4s, v12.4s, v5.s[2]
	fmla	v10.4s, v13.4s, v5.s[3]
	fmla	v11.4s, v6.4s, v5.s[3]
	fmla	v26.4s, v31.4s, v4.s[0]
	fmla	v26.4s, v8.4s, v4.s[1]
	fmla	v26.4s, v12.4s, v4.s[2]
	fmla	v26.4s, v13.4s, v4.s[3]
	fmla	v9.4s, v6.4s, v4.s[3]
	fmla	v17.4s, v31.4s, v3.s[0]
	fmla	v17.4s, v8.4s, v3.s[1]
	fmla	v17.4s, v12.4s, v3.s[2]
	fmla	v17.4s, v13.4s, v3.s[3]
	fmla	v16.4s, v6.4s, v3.s[3]
	fmla	v18.4s, v31.4s, v2.s[0]
	fmla	v18.4s, v8.4s, v2.s[1]
	fmla	v18.4s, v12.4s, v2.s[2]
	fmla	v18.4s, v13.4s, v2.s[3]
	fmla	v19.4s, v6.4s, v2.s[3]
	fmla	v21.4s, v31.4s, v25.s[0]
	fmla	v21.4s, v8.4s, v25.s[1]
	fmla	v21.4s, v12.4s, v25.s[2]
	fmla	v21.4s, v13.4s, v25.s[3]
	fmla	v20.4s, v6.4s, v25.s[3]
	fmla	v22.4s, v31.4s, v1.s[0]
	fmla	v22.4s, v8.4s, v1.s[1]
	fmla	v22.4s, v12.4s, v1.s[2]
	fmla	v22.4s, v13.4s, v1.s[3]
	fmla	v23.4s, v6.4s, v1.s[3]
	fmla	v24.4s, v6.4s, v7.s[3]
	ldr	q1, [x12, #112]
	ldr	q25, [x14, #7184]
	fmla	v15.4s, v25.4s, v1.s[0]
	ldr	q2, [x12, #240]
	fmla	v11.4s, v25.4s, v2.s[0]
	ldr	q3, [x12, #368]
	fmla	v9.4s, v25.4s, v3.s[0]
	ldr	q4, [x12, #496]
	fmla	v16.4s, v25.4s, v4.s[0]
	ldr	q5, [x12, #624]
	fmla	v19.4s, v25.4s, v5.s[0]
	ldr	q6, [x12, #752]
	fmla	v20.4s, v25.4s, v6.s[0]
	ldr	q7, [x12, #880]
	fmla	v23.4s, v25.4s, v7.s[0]
	ldr	q30, [x12, #1008]
	fmla	v24.4s, v25.4s, v30.s[0]
	ldr	q25, [x14, #7440]
	fmla	v15.4s, v25.4s, v1.s[1]
	fmla	v11.4s, v25.4s, v2.s[1]
	fmla	v9.4s, v25.4s, v3.s[1]
	fmla	v16.4s, v25.4s, v4.s[1]
	fmla	v19.4s, v25.4s, v5.s[1]
	fmla	v20.4s, v25.4s, v6.s[1]
	fmla	v23.4s, v25.4s, v7.s[1]
	fmla	v24.4s, v25.4s, v30.s[1]
	ldr	q25, [x14, #7696]
	fmla	v15.4s, v25.4s, v1.s[2]
	fmla	v11.4s, v25.4s, v2.s[2]
	fmla	v9.4s, v25.4s, v3.s[2]
	fmla	v16.4s, v25.4s, v4.s[2]
	fmla	v19.4s, v25.4s, v5.s[2]
	fmla	v20.4s, v25.4s, v6.s[2]
	fmla	v23.4s, v25.4s, v7.s[2]
	fmla	v24.4s, v25.4s, v30.s[2]
	ldr	q29, [x14, #7168]
	fmla	v14.4s, v29.4s, v1.s[0]
	ldr	q28, [x14, #7424]
	fmla	v14.4s, v28.4s, v1.s[1]
	ldr	q27, [x14, #7680]
	fmla	v14.4s, v27.4s, v1.s[2]
	ldr	q25, [x14, #7936]
	fmla	v14.4s, v25.4s, v1.s[3]
	ldr	q0, [x14, #7952]
	fmla	v15.4s, v0.4s, v1.s[3]
	fmla	v10.4s, v29.4s, v2.s[0]
	fmla	v10.4s, v28.4s, v2.s[1]
	fmla	v10.4s, v27.4s, v2.s[2]
	fmla	v10.4s, v25.4s, v2.s[3]
	fmla	v11.4s, v0.4s, v2.s[3]
	fmla	v26.4s, v29.4s, v3.s[0]
	fmla	v26.4s, v28.4s, v3.s[1]
	fmla	v26.4s, v27.4s, v3.s[2]
	fmla	v26.4s, v25.4s, v3.s[3]
	fmla	v9.4s, v0.4s, v3.s[3]
	fmla	v17.4s, v29.4s, v4.s[0]
	fmla	v17.4s, v28.4s, v4.s[1]
	fmla	v17.4s, v27.4s, v4.s[2]
	fmla	v17.4s, v25.4s, v4.s[3]
	fmla	v16.4s, v0.4s, v4.s[3]
	fmla	v18.4s, v29.4s, v5.s[0]
	fmla	v18.4s, v28.4s, v5.s[1]
	fmla	v18.4s, v27.4s, v5.s[2]
	fmla	v18.4s, v25.4s, v5.s[3]
	fmla	v19.4s, v0.4s, v5.s[3]
	fmla	v21.4s, v29.4s, v6.s[0]
	fmla	v21.4s, v28.4s, v6.s[1]
	fmla	v21.4s, v27.4s, v6.s[2]
	fmla	v21.4s, v25.4s, v6.s[3]
	fmla	v20.4s, v0.4s, v6.s[3]
	fmla	v22.4s, v29.4s, v7.s[0]
	fmla	v22.4s, v28.4s, v7.s[1]
	fmla	v22.4s, v27.4s, v7.s[2]
	fmla	v22.4s, v25.4s, v7.s[3]
	fmla	v23.4s, v0.4s, v7.s[3]
	fmla	v24.4s, v0.4s, v30.s[3]
	add	x14, x13, x10
	ldp	q1, q0, [x14]
	fadd	v0.4s, v15.4s, v0.4s
	fadd	v1.4s, v14.4s, v1.4s
	ldp	q3, q2, [x14, #256]
	fadd	v2.4s, v11.4s, v2.4s
	fadd	v3.4s, v10.4s, v3.4s
	ldp	q5, q4, [x14, #512]
	fadd	v4.4s, v9.4s, v4.4s
	fadd	v5.4s, v26.4s, v5.4s
	ldp	q7, q6, [x14, #768]
	fadd	v6.4s, v16.4s, v6.4s
	fadd	v7.4s, v17.4s, v7.4s
	ldr	q16, [x14, #1040]
	fadd	v16.4s, v19.4s, v16.4s
	ldr	q17, [x14, #1024]
	fadd	v17.4s, v18.4s, v17.4s
	ldr	q18, [x14, #1296]
	fadd	v18.4s, v20.4s, v18.4s
	ldr	q19, [x14, #1280]
	fadd	v19.4s, v21.4s, v19.4s
	ldr	q20, [x14, #1552]
	fadd	v20.4s, v23.4s, v20.4s
	ldr	q21, [x14, #1536]
	fadd	v21.4s, v22.4s, v21.4s
	ldr	q22, [x14, #1808]
	fadd	v22.4s, v24.4s, v22.4s
	ldp	q26, q23, [sp, #208]            // 32-byte Folded Reload
	ldr	q24, [sp, #192]                 // 16-byte Folded Reload
	fmla	v26.4s, v24.4s, v23.s[1]
	movi	v24.2d, #0000000000000000
	fmax	v1.4s, v1.4s, v24.4s
	ldr	q9, [sp, #240]                  // 16-byte Folded Reload
	fmla	v26.4s, v9.4s, v23.s[2]
	mov	v9.16b, v23.16b
	fmax	v0.4s, v0.4s, v24.4s
	add	x15, x1, x8
	add	x15, x15, x10
	ldr	q23, [x14, #1792]
	stp	q1, q0, [x15]
	ldr	q0, [sp, #176]                  // 16-byte Folded Reload
	fmla	v26.4s, v0.4s, v9.s[3]
	fmax	v0.4s, v3.4s, v24.4s
	ldp	q3, q1, [sp, #96]               // 32-byte Folded Reload
	fmla	v26.4s, v1.4s, v3.s[0]
	fmax	v1.4s, v2.4s, v24.4s
	stp	q0, q1, [x15, #256]
	ldp	q0, q1, [sp, #128]              // 32-byte Folded Reload
	fmla	v26.4s, v0.4s, v3.s[1]
	fmax	v0.4s, v5.4s, v24.4s
	fmla	v26.4s, v1.4s, v3.s[2]
	fmax	v1.4s, v4.4s, v24.4s
	stp	q0, q1, [x15, #512]
	ldr	q0, [sp, #160]                  // 16-byte Folded Reload
	fmla	v26.4s, v0.4s, v3.s[3]
	fmax	v0.4s, v7.4s, v24.4s
	ldp	q3, q1, [sp, #16]               // 32-byte Folded Reload
	fmla	v26.4s, v1.4s, v3.s[0]
	fmax	v1.4s, v6.4s, v24.4s
	stp	q0, q1, [x15, #768]
	mov	v1.16b, v26.16b
	ldr	q0, [sp, #48]                   // 16-byte Folded Reload
	fmla	v1.4s, v0.4s, v3.s[1]
	fmax	v0.4s, v16.4s, v24.4s
	str	q0, [x15, #1040]
	ldr	q0, [sp, #64]                   // 16-byte Folded Reload
	fmla	v1.4s, v0.4s, v3.s[2]
	fmax	v0.4s, v17.4s, v24.4s
	str	q0, [x15, #1024]
	ldr	q0, [sp, #80]                   // 16-byte Folded Reload
	fmla	v1.4s, v0.4s, v3.s[3]
	fmax	v0.4s, v18.4s, v24.4s
	str	q0, [x15, #1296]
	ldr	q2, [sp]                        // 16-byte Folded Reload
	fmla	v1.4s, v31.4s, v2.s[0]
	fmax	v0.4s, v19.4s, v24.4s
	str	q0, [x15, #1280]
	fmla	v1.4s, v8.4s, v2.s[1]
	fmax	v0.4s, v20.4s, v24.4s
	str	q0, [x15, #1552]
	fmla	v1.4s, v12.4s, v2.s[2]
	fmax	v0.4s, v21.4s, v24.4s
	str	q0, [x15, #1536]
	fmla	v1.4s, v13.4s, v2.s[3]
	fmax	v0.4s, v22.4s, v24.4s
	str	q0, [x15, #1808]
	fmla	v1.4s, v29.4s, v30.s[0]
	fmla	v1.4s, v28.4s, v30.s[1]
	fmla	v1.4s, v27.4s, v30.s[2]
	fmla	v1.4s, v25.4s, v30.s[3]
	fadd	v0.4s, v1.4s, v23.4s
	fmax	v0.4s, v0.4s, v24.4s
	str	q0, [x15, #1792]
	add	x11, x11, #8
	add	x10, x10, #32
	cmp	x11, #63
	b.le	.LBB0_4
	b	.LBB0_1
.LBB0_5:
	mov	x2, xzr
	mov	w3, #16                         // =0x10
	mov	w4, #64                         // =0x40
	mov	w5, #64                         // =0x40
	mov	w6, #1                          // =0x1
	ldp	x20, x19, [sp, #352]            // 16-byte Folded Reload
	ldp	x30, x21, [sp, #336]            // 16-byte Folded Reload
	ldr	x29, [sp, #320]                 // 8-byte Folded Reload
	ldp	d9, d8, [sp, #304]              // 16-byte Folded Reload
	ldp	d11, d10, [sp, #288]            // 16-byte Folded Reload
	ldp	d13, d12, [sp, #272]            // 16-byte Folded Reload
	ldp	d15, d14, [sp, #256]            // 16-byte Folded Reload
	add	sp, sp, #368
	ret
.Lfunc_end0:
	.size	matmul_bias_relu_tiled_16x64x32, .Lfunc_end0-matmul_bias_relu_tiled_16x64x32
	.cfi_endproc
                                        // -- End function
	.globl	_mlir_ciface_matmul_bias_relu_tiled_16x64x32 // -- Begin function _mlir_ciface_matmul_bias_relu_tiled_16x64x32
	.p2align	4
	.type	_mlir_ciface_matmul_bias_relu_tiled_16x64x32,@function
_mlir_ciface_matmul_bias_relu_tiled_16x64x32: // @_mlir_ciface_matmul_bias_relu_tiled_16x64x32
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
	bl	matmul_bias_relu_tiled_16x64x32
	stp	x0, x1, [x19]
	stp	x2, x3, [x19, #16]
	stp	x4, x5, [x19, #32]
	str	x6, [x19, #48]
	ldp	x30, x19, [sp, #112]            // 16-byte Folded Reload
	add	sp, sp, #128
	ret
.Lfunc_end1:
	.size	_mlir_ciface_matmul_bias_relu_tiled_16x64x32, .Lfunc_end1-_mlir_ciface_matmul_bias_relu_tiled_16x64x32
	.cfi_endproc
                                        // -- End function
	.section	".note.GNU-stack","",@progbits
