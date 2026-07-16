	.file	"LLVMDialectModule"
	.text
	.globl	matmul_bias_relu_16x16x16       // -- Begin function matmul_bias_relu_16x16x16
	.p2align	4
	.type	matmul_bias_relu_16x16x16,@function
matmul_bias_relu_16x16x16:              // @matmul_bias_relu_16x16x16
	.cfi_startproc
// %bb.0:
	sub	sp, sp, #128
	stp	x29, x30, [sp, #32]             // 16-byte Folded Spill
	stp	x28, x27, [sp, #48]             // 16-byte Folded Spill
	stp	x26, x25, [sp, #64]             // 16-byte Folded Spill
	stp	x24, x23, [sp, #80]             // 16-byte Folded Spill
	stp	x22, x21, [sp, #96]             // 16-byte Folded Spill
	stp	x20, x19, [sp, #112]            // 16-byte Folded Spill
	.cfi_def_cfa_offset 128
	.cfi_offset w19, -8
	.cfi_offset w20, -16
	.cfi_offset w21, -24
	.cfi_offset w22, -32
	.cfi_offset w23, -40
	.cfi_offset w24, -48
	.cfi_offset w25, -56
	.cfi_offset w26, -64
	.cfi_offset w27, -72
	.cfi_offset w28, -80
	.cfi_offset w30, -88
	.cfi_offset w29, -96
	mov	x20, x6
	mov	x21, x5
	mov	x22, x2
	mov	x23, x1
	ldp	x25, x24, [sp, #216]
	ldp	x8, x9, [sp, #184]
	stp	x8, x9, [sp, #8]                // 16-byte Folded Spill
	ldp	x27, x29, [sp, #160]
	ldp	x19, x28, [sp, #128]
	mov	w0, #1088                       // =0x440
	bl	malloc
	mov	x8, xzr
	mov	x9, xzr
	str	x0, [sp, #24]                   // 8-byte Folded Spill
	add	x10, x0, #63
	and	x26, x10, #0xffffffffffffffc0
	b	.LBB0_2
	.p2align	5, , 16
.LBB0_1:                                //   in Loop: Header=BB0_2 Depth=1
	add	x9, x9, #1
	add	x8, x8, #64
.LBB0_2:                                // =>This Loop Header: Depth=1
                                        //     Child Loop BB0_4 Depth 2
	cmp	x9, #15
	b.gt	.LBB0_5
// %bb.3:                               // %.preheader4
                                        //   in Loop: Header=BB0_2 Depth=1
	mov	x10, xzr
	mov	x11, x8
	cmp	x10, #15
	b.gt	.LBB0_1
	.p2align	5, , 16
.LBB0_4:                                //   Parent Loop BB0_2 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	str	wzr, [x26, x11]
	add	x10, x10, #1
	add	x11, x11, #4
	cmp	x10, #15
	b.le	.LBB0_4
	b	.LBB0_1
.LBB0_5:                                // %.preheader3
	mov	x8, xzr
	add	x9, x23, x22, lsl #2
	add	x10, x19, x28, lsl #2
	b	.LBB0_7
	.p2align	5, , 16
.LBB0_6:                                //   in Loop: Header=BB0_7 Depth=1
	add	x8, x8, #1
.LBB0_7:                                // =>This Loop Header: Depth=1
                                        //     Child Loop BB0_10 Depth 2
                                        //       Child Loop BB0_12 Depth 3
	cmp	x8, #15
	b.gt	.LBB0_13
// %bb.8:                               // %.preheader2
                                        //   in Loop: Header=BB0_7 Depth=1
	mov	x11, xzr
	b	.LBB0_10
	.p2align	5, , 16
.LBB0_9:                                //   in Loop: Header=BB0_10 Depth=2
	add	x11, x11, #1
.LBB0_10:                               //   Parent Loop BB0_7 Depth=1
                                        // =>  This Loop Header: Depth=2
                                        //       Child Loop BB0_12 Depth 3
	cmp	x11, #15
	b.gt	.LBB0_6
// %bb.11:                              // %.preheader1
                                        //   in Loop: Header=BB0_10 Depth=2
	mov	x12, xzr
	mul	x13, x11, x29
	cmp	x12, #15
	b.gt	.LBB0_9
	.p2align	5, , 16
.LBB0_12:                               //   Parent Loop BB0_7 Depth=1
                                        //     Parent Loop BB0_10 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	mul	x14, x12, x20
	madd	x14, x8, x21, x14
	ldr	s0, [x9, x14, lsl #2]
	madd	x14, x12, x27, x13
	ldr	s1, [x10, x14, lsl #2]
	add	x14, x11, x8, lsl #4
	ldr	s2, [x26, x14, lsl #2]
	fmul	s0, s0, s1
	fadd	s0, s2, s0
	str	s0, [x26, x14, lsl #2]
	add	x12, x12, #1
	cmp	x12, #15
	b.le	.LBB0_12
	b	.LBB0_9
.LBB0_13:
	mov	w0, #1088                       // =0x440
	bl	malloc
	mov	x20, x0
	mov	x8, xzr
	mov	x9, xzr
	add	x10, x0, #63
	and	x21, x10, #0xffffffffffffffc0
	ldp	x11, x10, [sp, #8]              // 16-byte Folded Reload
	add	x10, x11, x10, lsl #2
	movi	d0, #0000000000000000
	b	.LBB0_15
	.p2align	5, , 16
.LBB0_14:                               //   in Loop: Header=BB0_15 Depth=1
	add	x9, x9, #1
	add	x8, x8, #64
.LBB0_15:                               // =>This Loop Header: Depth=1
                                        //     Child Loop BB0_17 Depth 2
	cmp	x9, #15
	b.gt	.LBB0_18
// %bb.16:                              // %.preheader
                                        //   in Loop: Header=BB0_15 Depth=1
	mov	x11, xzr
	cmp	x11, #15
	b.gt	.LBB0_14
	.p2align	5, , 16
.LBB0_17:                               //   Parent Loop BB0_15 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	add	x12, x26, x8
	ldr	s1, [x12, x11, lsl #2]
	mul	x12, x11, x24
	madd	x12, x9, x25, x12
	ldr	s2, [x10, x12, lsl #2]
	fadd	s1, s1, s2
	fmax	s1, s1, s0
	add	x12, x21, x8
	str	s1, [x12, x11, lsl #2]
	add	x11, x11, #1
	cmp	x11, #15
	b.le	.LBB0_17
	b	.LBB0_14
.LBB0_18:
	ldr	x0, [sp, #24]                   // 8-byte Folded Reload
	bl	free
	mov	x0, x20
	mov	x1, x21
	mov	x2, xzr
	mov	w3, #16                         // =0x10
	mov	w4, #16                         // =0x10
	mov	w5, #16                         // =0x10
	mov	w6, #1                          // =0x1
	ldp	x20, x19, [sp, #112]            // 16-byte Folded Reload
	ldp	x22, x21, [sp, #96]             // 16-byte Folded Reload
	ldp	x24, x23, [sp, #80]             // 16-byte Folded Reload
	ldp	x26, x25, [sp, #64]             // 16-byte Folded Reload
	ldp	x28, x27, [sp, #48]             // 16-byte Folded Reload
	ldp	x29, x30, [sp, #32]             // 16-byte Folded Reload
	add	sp, sp, #128
	ret
.Lfunc_end0:
	.size	matmul_bias_relu_16x16x16, .Lfunc_end0-matmul_bias_relu_16x16x16
	.cfi_endproc
                                        // -- End function
	.globl	_mlir_ciface_matmul_bias_relu_16x16x16 // -- Begin function _mlir_ciface_matmul_bias_relu_16x16x16
	.p2align	4
	.type	_mlir_ciface_matmul_bias_relu_16x16x16,@function
_mlir_ciface_matmul_bias_relu_16x16x16: // @_mlir_ciface_matmul_bias_relu_16x16x16
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
	bl	matmul_bias_relu_16x16x16
	stp	x0, x1, [x19]
	stp	x2, x3, [x19, #16]
	stp	x4, x5, [x19, #32]
	str	x6, [x19, #48]
	ldp	x30, x19, [sp, #112]            // 16-byte Folded Reload
	add	sp, sp, #128
	ret
.Lfunc_end1:
	.size	_mlir_ciface_matmul_bias_relu_16x16x16, .Lfunc_end1-_mlir_ciface_matmul_bias_relu_16x16x16
	.cfi_endproc
                                        // -- End function
	.section	".note.GNU-stack","",@progbits
