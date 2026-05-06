#include "kernels/cpu_kernels.h"
#include "runtime/thread_pool.h"
#include <cmath>
#include <vector>
#include <iostream>
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

    ThreadPool pool(num_threads);

    int rows_per_thread = (M + num_threads - 1) / num_threads;

    for (int t = 0; t < num_threads; ++t) {
        int row_start = t * rows_per_thread;
        int row_end = std::min(row_start + rows_per_thread, M);

        if (row_start >= row_end) continue;

        pool.enqueue([&, row_start, row_end]() {

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

        });
    }

    pool.wait();
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

void matmul_tiled_threaded_pool(
    const Tensor& A,
    const Tensor& B,
    Tensor& C,
    ThreadPool& pool,
    int tile_size,
    int num_tasks
) {
    int M = A.shape[0];
    int K = A.shape[1];
    int N = B.shape[1];

    std::fill(C.data.begin(), C.data.end(), 0.0f);

    // Here num_tasks is interpreted as row-block size.
    // Example: 32 means each task computes 32 rows.
    int row_block = num_tasks;

    for (int ii = 0; ii < M; ii += row_block) {
        int i_start = ii;
        int i_end = std::min(ii + row_block, M);

        pool.enqueue([&, i_start, i_end]() {
            for (int kk = 0; kk < K; kk += tile_size) {
                for (int jj = 0; jj < N; jj += tile_size) {
                    int k_max = std::min(kk + tile_size, K);
                    int j_max = std::min(jj + tile_size, N);

                    for (int i = i_start; i < i_end; ++i) {
                        for (int k = kk; k < k_max; ++k) {
                            float a = A.data[i * K + k];

                            for (int j = jj; j < j_max; ++j) {
                                C.data[i * N + j] += a * B.data[k * N + j];
                            }
                        }
                    }
                }
            }
        });
    }

    pool.wait();
}

void fused_matmul_add_relu_threaded_pool(
    const Tensor& A,
    const Tensor& B,
    const Tensor& Bias,
    Tensor& Out,
    ThreadPool& pool,
    int num_tasks
) {
    matmul_tiled_threaded_pool(A, B, Out, pool, 32, num_tasks);

    for (size_t i = 0; i < Out.data.size(); ++i) {
        Out.data[i] += Bias.data[i];
        Out.data[i] = std::max(0.0f, Out.data[i]);
    }
}
void attention(const Tensor& Q, const Tensor& K, const Tensor& V, Tensor& Out) {
    int S = Q.shape[0];  // sequence length
    int D = Q.shape[1];  // hidden dimension

    std::vector<float> scores(S * S, 0.0f);

    float scale = 1.0f / std::sqrt(static_cast<float>(D));

    // scores = Q * K^T / sqrt(D)
    for (int i = 0; i < S; ++i) {
        for (int j = 0; j < S; ++j) {
            float sum = 0.0f;

            for (int d = 0; d < D; ++d) {
                sum += Q.data[i * D + d] * K.data[j * D + d];
            }

            scores[i * S + j] = sum * scale;
        }
    }

    // row-wise softmax
    for (int i = 0; i < S; ++i) {
        float max_score = scores[i * S];

        for (int j = 1; j < S; ++j) {
            max_score = std::max(max_score, scores[i * S + j]);
        }

        float denom = 0.0f;

        for (int j = 0; j < S; ++j) {
            scores[i * S + j] = std::exp(scores[i * S + j] - max_score);
            denom += scores[i * S + j];
        }

        for (int j = 0; j < S; ++j) {
            scores[i * S + j] /= denom;
        }
    }

    // Out = softmax(scores) * V
    for (int i = 0; i < S; ++i) {
        for (int d = 0; d < D; ++d) {
            float sum = 0.0f;

            for (int j = 0; j < S; ++j) {
                sum += scores[i * S + j] * V.data[j * D + d];
            }

            Out.data[i * D + d] = sum;
        }
    }
}

void causal_attention(const Tensor& Q, const Tensor& K, const Tensor& V, Tensor& Out) {
    int S = Q.shape[0];
    int D = Q.shape[1];

    std::vector<float> scores(S * S, 0.0f);

    float scale = 1.0f / std::sqrt(static_cast<float>(D));

    // scores = Q * K^T / sqrt(D), with causal mask
    for (int i = 0; i < S; ++i) {
        for (int j = 0; j < S; ++j) {
            if (j > i) {
                scores[i * S + j] = -1e9f;
                continue;
            }

            float sum = 0.0f;

            for (int d = 0; d < D; ++d) {
                sum += Q.data[i * D + d] * K.data[j * D + d];
            }

            scores[i * S + j] = sum * scale;
        }
    }

    // row-wise softmax
    for (int i = 0; i < S; ++i) {
        float max_score = scores[i * S];

        for (int j = 1; j < S; ++j) {
            max_score = std::max(max_score, scores[i * S + j]);
        }

        float denom = 0.0f;

        for (int j = 0; j < S; ++j) {
            scores[i * S + j] = std::exp(scores[i * S + j] - max_score);
            denom += scores[i * S + j];
        }

        for (int j = 0; j < S; ++j) {
            scores[i * S + j] /= denom;
        }
    }

    // Out = softmax(scores) * V
    for (int i = 0; i < S; ++i) {
        for (int d = 0; d < D; ++d) {
            float sum = 0.0f;

            for (int j = 0; j < S; ++j) {
                sum += scores[i * S + j] * V.data[j * D + d];
            }

            Out.data[i * D + d] = sum;
        }
    }
    std::cout << "[Kernel] CausalAttention finished\n";
}

void decode_attention(const Tensor& Q, const Tensor& K_cache, const Tensor& V_cache, Tensor& Out) {
    int T = K_cache.shape[0];
    int D = Q.shape[1];

    std::vector<float> scores(T, 0.0f);

    float scale = 1.0f / std::sqrt(static_cast<float>(D));

    for (int j = 0; j < T; ++j) {
        float sum = 0.0f;

        for (int d = 0; d < D; ++d) {
            sum += Q.data[d] * K_cache.data[j * D + d];
        }

        scores[j] = sum * scale;
    }

    float max_score = scores[0];
    for (int j = 1; j < T; ++j) {
        max_score = std::max(max_score, scores[j]);
    }

    float denom = 0.0f;
    for (int j = 0; j < T; ++j) {
        scores[j] = std::exp(scores[j] - max_score);
        denom += scores[j];
    }

    for (int j = 0; j < T; ++j) {
        scores[j] /= denom;
    }

    for (int d = 0; d < D; ++d) {
        float sum = 0.0f;

        for (int j = 0; j < T; ++j) {
            sum += scores[j] * V_cache.data[j * D + d];
        }

        Out.data[d] = sum;
    }
}