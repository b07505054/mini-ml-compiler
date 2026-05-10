#include "kernels/cpu_kernels.h"
#include "runtime/thread_pool.h"
#include <cmath>
#include <vector>
#include <iostream>
#include <algorithm>
#include <stdexcept>
#include <thread>
#include <immintrin.h>
#include <cstdint>
#ifdef __ARM_NEON
#include <arm_neon.h>
#endif
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

void relu_avx2(const Tensor& A, Tensor& B) {

    int N = static_cast<int>(A.data.size());

    B.data.resize(N);

    __m256 zero = _mm256_setzero_ps();

    int i = 0;

    for (; i + 8 <= N; i += 8) {

        __m256 x = _mm256_loadu_ps(&A.data[i]);

        __m256 y = _mm256_max_ps(x, zero);

        _mm256_storeu_ps(&B.data[i], y);
    }

    // tail
    for (; i < N; ++i) {
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

void add_avx2(const Tensor& A, const Tensor& B, Tensor& C) {
    int N = static_cast<int>(A.data.size());

    C.data.resize(N);

    int i = 0;

    for (; i + 8 <= N; i += 8) {
        __m256 a = _mm256_loadu_ps(&A.data[i]);
        __m256 b = _mm256_loadu_ps(&B.data[i]);
        __m256 c = _mm256_add_ps(a, b);

        _mm256_storeu_ps(&C.data[i], c);
    }

    for (; i < N; ++i) {
        C.data[i] = A.data[i] + B.data[i];
    }
}
void matmul_avx2(const Tensor& A, const Tensor& B, Tensor& C) {
    int M = A.shape[0];
    int K = A.shape[1];
    int N = B.shape[1];

    C.shape = {M, N};
    C.data.assign(M * N, 0.0f);

    int vecN = (N / 8) * 8;

    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < vecN; j += 8) {
            __m256 acc = _mm256_setzero_ps();

            for (int k = 0; k < K; ++k) {
                __m256 a = _mm256_set1_ps(A.data[i * K + k]);
                __m256 b = _mm256_loadu_ps(&B.data[k * N + j]);
                acc = _mm256_add_ps(_mm256_mul_ps(a, b), acc);
            }

            _mm256_storeu_ps(&C.data[i * N + j], acc);
        }

        // tail columns
        for (int j = vecN; j < N; ++j) {
            float sum = 0.0f;

            for (int k = 0; k < K; ++k) {
                sum += A.data[i * K + k] * B.data[k * N + j];
            }

            C.data[i * N + j] = sum;
        }
    }
}
QuantTensor quantize_tensor_symmetric(
    const Tensor& input,
    float scale
) {
    QuantTensor q(input.name + "_int8", input.shape, scale, 0);

    for (int i = 0; i < input.numel(); ++i) {
        int qvalue = static_cast<int>(std::round(input.data[i] / scale));

        qvalue = std::max(-128, std::min(127, qvalue));

        q.data[i] = static_cast<int8_t>(qvalue);
    }

    return q;
}

void matmul_int8(
    const QuantTensor& A,
    const QuantTensor& B,
    Tensor& C
) {
    int M = A.shape[0];
    int K = A.shape[1];
    int N = B.shape[1];

    C.shape = {M, N};
    C.data.assign(M * N, 0.0f);

    float output_scale = A.scale * B.scale;

    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            int32_t acc = 0;

            for (int k = 0; k < K; ++k) {
                int32_t a =
                    static_cast<int32_t>(A.data[i * K + k]) - A.zero_point;

                int32_t b =
                    static_cast<int32_t>(B.data[k * N + j]) - B.zero_point;

                acc += a * b;
            }

            C.data[i * N + j] =
                static_cast<float>(acc) * output_scale;
        }
    }
}
void add_neon(
    const Tensor& A,
    const Tensor& B,
    Tensor& C
) {

    C.shape = A.shape;
    C.data.resize(A.numel());

#ifdef __ARM_NEON

    int i = 0;

    for (; i + 4 <= A.numel(); i += 4) {

        float32x4_t va =
            vld1q_f32(&A.data[i]);

        float32x4_t vb =
            vld1q_f32(&B.data[i]);

        float32x4_t vc =
            vaddq_f32(va, vb);

        vst1q_f32(&C.data[i], vc);
    }

    for (; i < A.numel(); ++i) {
        C.data[i] = A.data[i] + B.data[i];
    }

#else

    // fallback on non-ARM systems

    add(A, B, C);

#endif
}

void layernorm(
    const Tensor& input,
    Tensor& output,
    float eps
) {
    output.shape = input.shape;
    output.data.resize(input.data.size());

    int rows = input.shape[0];
    int cols = input.shape[1];

    for (int i = 0; i < rows; ++i) {
        float mean = 0.0f;

        for (int j = 0; j < cols; ++j) {
            mean += input.data[i * cols + j];
        }

        mean /= cols;

        float var = 0.0f;

        for (int j = 0; j < cols; ++j) {
            float diff = input.data[i * cols + j] - mean;
            var += diff * diff;
        }

        var /= cols;

        float inv_std = 1.0f / std::sqrt(var + eps);

        for (int j = 0; j < cols; ++j) {
            float x = input.data[i * cols + j];
            output.data[i * cols + j] =
                (x - mean) * inv_std;
        }
    }
}

void fused_attention(
    const Tensor& Q,
    const Tensor& K,
    const Tensor& V,
    Tensor& output
) {
    int seq = Q.shape[0];
    int dim = Q.shape[1];

    output.shape = {seq, dim};
    output.data.assign(seq * dim, 0.0f);

    std::vector<float> scores(seq);

    for (int i = 0; i < seq; ++i) {

        float max_score = -1e9f;

        for (int j = 0; j < seq; ++j) {

            float score = 0.0f;

            for (int d = 0; d < dim; ++d) {
                score +=
                    Q.data[i * dim + d] *
                    K.data[j * dim + d];
            }

            score /= std::sqrt(static_cast<float>(dim));

            scores[j] = score;

            if (score > max_score) {
                max_score = score;
            }
        }

        float sum = 0.0f;

        for (int j = 0; j < seq; ++j) {
            scores[j] = std::exp(scores[j] - max_score);
            sum += scores[j];
        }

        for (int j = 0; j < seq; ++j) {
            scores[j] /= sum;
        }

        for (int d = 0; d < dim; ++d) {

            float value = 0.0f;

            for (int j = 0; j < seq; ++j) {
                value +=
                    scores[j] *
                    V.data[j * dim + d];
            }

            output.data[i * dim + d] = value;
        }
    }
}