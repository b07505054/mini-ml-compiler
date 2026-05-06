#pragma once

#include "ir/tensor.h"

#include <vector>

class KVCache {
public:
    KVCache(int max_seq_len, int hidden_dim);

    void append(const Tensor& K_new, const Tensor& V_new);

    Tensor get_cached_K() const;
    Tensor get_cached_V() const;

    int size() const;

private:
    int max_seq_len;
    int hidden_dim;
    int current_len;

    std::vector<float> K_cache;
    std::vector<float> V_cache;
};