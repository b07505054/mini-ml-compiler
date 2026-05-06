#include "runtime/kv_cache.h"
#include "ir/tensor.h"
#include "kernels/cpu_kernels.h"

#include <iostream>

void print_tensor(const Tensor& t) {
    for (float x : t.data) {
        std::cout << x << " ";
    }
    std::cout << "\n";
}

int main() {
    const int D = 4;
    const int MAX_SEQ = 4;

    KVCache cache(MAX_SEQ, D);

    Tensor K0("K0", {1, D});
    Tensor V0("V0", {1, D});

    K0.data = {1, 0, 0, 0};
    V0.data = {10, 0, 0, 0};

    cache.append(K0, V0);

    Tensor Q0("Q0", {1, D});
    Tensor Out0("Out0", {1, D});

    Q0.data = {1, 0, 0, 0};

    decode_attention(Q0, cache.get_cached_K(), cache.get_cached_V(), Out0);

    std::cout << "Step 0 output:\n";
    print_tensor(Out0);

    Tensor K1("K1", {1, D});
    Tensor V1("V1", {1, D});

    K1.data = {0, 1, 0, 0};
    V1.data = {0, 20, 0, 0};

    cache.append(K1, V1);

    Tensor Q1("Q1", {1, D});
    Tensor Out1("Out1", {1, D});

    Q1.data = {0, 1, 0, 0};

    decode_attention(Q1, cache.get_cached_K(), cache.get_cached_V(), Out1);

    std::cout << "Step 1 output using KV cache:\n";
    print_tensor(Out1);

    std::cout << "KV cache size: " << cache.size() << "\n";

    return 0;
}