#pragma once

#include <functional>

struct Dim3 {
    int x, y, z;

    Dim3(int x = 1, int y = 1, int z = 1) : x(x), y(y), z(z) {}
};

class GPUSimulator {
public:
    using KernelFn = std::function<void(int bx, int by, int tx, int ty)>;

    static void launch(
        Dim3 gridDim,
        Dim3 blockDim,
        KernelFn kernel
    );
};