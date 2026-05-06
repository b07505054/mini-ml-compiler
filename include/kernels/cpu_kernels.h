#pragma once

#include "ir/tensor.h"

void matmul(const Tensor& A, const Tensor& B, Tensor& C);
void matmul_tiled(const Tensor& A, const Tensor& B, Tensor& C, int tile_size = 32);

void add(const Tensor& A, const Tensor& B, Tensor& C);
void relu(const Tensor& A, Tensor& B);
void attention(const Tensor& Q, const Tensor& K, const Tensor& V, Tensor& Out);

void fused_matmul_add_relu_baseline(const Tensor& A, const Tensor& B, const Tensor& Bias, Tensor& Out);
void fused_matmul_add_relu_optimized(const Tensor& A, const Tensor& B, const Tensor& Bias, Tensor& Out);
void causal_attention(const Tensor& Q, const Tensor& K, const Tensor& V, Tensor& Out);
// default fused op uses optimized kernel
void fused_matmul_add_relu(const Tensor& A, const Tensor& B, const Tensor& Bias, Tensor& Out);
void matmul_tiled_threaded(const Tensor& A, const Tensor& B, Tensor& C, int tile_size = 32, int num_threads = 4);

void fused_matmul_add_relu_threaded(const Tensor& A, const Tensor& B, const Tensor& Bias, Tensor& Out, int num_threads = 4);
class ThreadPool;

void matmul_tiled_threaded_pool(
    const Tensor& A,
    const Tensor& B,
    Tensor& C,
    ThreadPool& pool,
    int tile_size = 32,
    int num_tasks = 8
);

void fused_matmul_add_relu_threaded_pool(
    const Tensor& A,
    const Tensor& B,
    const Tensor& Bias,
    Tensor& Out,
    ThreadPool& pool,
    int num_tasks = 8
);
void decode_attention(const Tensor& Q, const Tensor& K_cache, const Tensor& V_cache, Tensor& Out);
void relu_avx2(const Tensor& A, Tensor& B);
void add_avx2(const Tensor& A, const Tensor& B, Tensor& C);
