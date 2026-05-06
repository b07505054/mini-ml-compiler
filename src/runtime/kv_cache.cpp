#include "runtime/kv_cache.h"

#include <stdexcept>

KVCache::KVCache(int max_seq_len, int hidden_dim)
    : max_seq_len(max_seq_len),
      hidden_dim(hidden_dim),
      current_len(0),
      K_cache(max_seq_len * hidden_dim, 0.0f),
      V_cache(max_seq_len * hidden_dim, 0.0f) {}

void KVCache::append(const Tensor& K_new, const Tensor& V_new) {
    if (K_new.shape.size() != 2 || V_new.shape.size() != 2) {
        throw std::runtime_error("KVCache expects 2D K/V tensors");
    }

    int new_tokens = K_new.shape[0];

    if (K_new.shape[1] != hidden_dim || V_new.shape[1] != hidden_dim) {
        throw std::runtime_error("KVCache hidden_dim mismatch");
    }

    if (current_len + new_tokens > max_seq_len) {
        throw std::runtime_error("KVCache capacity exceeded");
    }

    for (int i = 0; i < new_tokens; ++i) {
        for (int d = 0; d < hidden_dim; ++d) {
            K_cache[(current_len + i) * hidden_dim + d] =
                K_new.data[i * hidden_dim + d];

            V_cache[(current_len + i) * hidden_dim + d] =
                V_new.data[i * hidden_dim + d];
        }
    }

    current_len += new_tokens;
}

Tensor KVCache::get_cached_K() const {
    Tensor K("cached_K", {current_len, hidden_dim});
    K.data.assign(K_cache.begin(), K_cache.begin() + current_len * hidden_dim);
    return K;
}

Tensor KVCache::get_cached_V() const {
    Tensor V("cached_V", {current_len, hidden_dim});
    V.data.assign(V_cache.begin(), V_cache.begin() + current_len * hidden_dim);
    return V;
}

int KVCache::size() const {
    return current_len;
}