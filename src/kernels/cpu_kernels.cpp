#include "kernels/cpu_kernels.h"

#include <algorithm>
#include <stdexcept>
#include <thread>

void matmul(const Tensor& A, const Tensor& B, Tensor& C) {
    int M = A.shape[0];
    int K = A.shape[1];
    int N = B.shape[1];

    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                sum += A.data[i * K + k] * B.data[k * N + j];
            }
            C.data[i * N + j] = sum;
        }
    }
}

void matmul_tiled(const Tensor& A, const Tensor& B, Tensor& C, int tile_size) {
    int M = A.shape[0];
    int K = A.shape[1];
    int N = B.shape[1];

    std::fill(C.data.begin(), C.data.end(), 0.0f);

    for (int ii = 0; ii < M; ii += tile_size) {
        for (int kk = 0; kk < K; kk += tile_size) {
            for (int jj = 0; jj < N; jj += tile_size) {

                int i_max = std::min(ii + tile_size, M);
                int k_max = std::min(kk + tile_size, K);
                int j_max = std::min(jj + tile_size, N);

                for (int i = ii; i < i_max; ++i) {
                    for (int k = kk; k < k_max; ++k) {
                        float a = A.data[i * K + k];

                        for (int j = jj; j < j_max; ++j) {
                            C.data[i * N + j] += a * B.data[k * N + j];
                        }
                    }
                }
            }
        }
    }
}

void matmul_tiled_threaded(const Tensor& A, const Tensor& B, Tensor& C, int tile_size, int num_threads) {
    int M = A.shape[0];
    int K = A.shape[1];
    int N = B.shape[1];

    std::fill(C.data.begin(), C.data.end(), 0.0f);

    auto worker = [&](int row_start, int row_end) {
        for (int ii = row_start; ii < row_end; ii += tile_size) {
            for (int kk = 0; kk < K; kk += tile_size) {
                for (int jj = 0; jj < N; jj += tile_size) {

                    int i_max = std::min(ii + tile_size, row_end);
                    int k_max = std::min(kk + tile_size, K);
                    int j_max = std::min(jj + tile_size, N);

                    for (int i = ii; i < i_max; ++i) {
                        for (int k = kk; k < k_max; ++k) {
                            float a = A.data[i * K + k];

                            for (int j = jj; j < j_max; ++j) {
                                C.data[i * N + j] += a * B.data[k * N + j];
                            }
                        }
                    }
                }
            }
        }
    };

    std::vector<std::thread> threads;

    int rows_per_thread = (M + num_threads - 1) / num_threads;

    for (int t = 0; t < num_threads; ++t) {
        int row_start = t * rows_per_thread;
        int row_end = std::min(row_start + rows_per_thread, M);

        if (row_start < row_end) {
            threads.emplace_back(worker, row_start, row_end);
        }
    }

    for (auto& th : threads) {
        th.join();
    }
}

void fused_matmul_add_relu_threaded(const Tensor& A, const Tensor& B, const Tensor& Bias, Tensor& Out, int num_threads) {
    matmul_tiled_threaded(A, B, Out, 32, num_threads);

    for (size_t i = 0; i < Out.data.size(); ++i) {
        Out.data[i] += Bias.data[i];
        Out.data[i] = std::max(0.0f, Out.data[i]);
    }
}

void add(const Tensor& A, const Tensor& B, Tensor& C) {
    if (A.data.size() != B.data.size() || A.data.size() != C.data.size()) {
        throw std::runtime_error("Add shape mismatch");
    }

    for (size_t i = 0; i < A.data.size(); ++i) {
        C.data[i] = A.data[i] + B.data[i];
    }
}

void relu(const Tensor& A, Tensor& B) {
    if (A.data.size() != B.data.size()) {
        throw std::runtime_error("ReLU shape mismatch");
    }

    for (size_t i = 0; i < A.data.size(); ++i) {
        B.data[i] = std::max(0.0f, A.data[i]);
    }
}

void fused_matmul_add_relu_baseline(const Tensor& A, const Tensor& B, const Tensor& Bias, Tensor& Out) {
    matmul(A, B, Out);

    for (size_t i = 0; i < Out.data.size(); ++i) {
        Out.data[i] += Bias.data[i];
        Out.data[i] = std::max(0.0f, Out.data[i]);
    }
}

void fused_matmul_add_relu_optimized(const Tensor& A, const Tensor& B, const Tensor& Bias, Tensor& Out) {
    matmul_tiled(A, B, Out, 32);

    for (size_t i = 0; i < Out.data.size(); ++i) {
        Out.data[i] += Bias.data[i];
        Out.data[i] = std::max(0.0f, Out.data[i]);
    }
}

void fused_matmul_add_relu(const Tensor& A, const Tensor& B, const Tensor& Bias, Tensor& Out) {
    fused_matmul_add_relu_optimized(A, B, Bias, Out);
}