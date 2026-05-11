#pragma once

#include <vector>
#include <string>

enum class DType {
    Float32
};

struct Tensor {
    std::string name;
    std::vector<int> shape;
    DType dtype;
    std::vector<float> data;
    int first_use = -1;
    int last_use = -1;
    int memory_offset = -1;
    float* runtime_data = nullptr;
    bool persistent = false;
    Tensor() = default;

    Tensor(const std::string& name, std::vector<int> shape)
        : name(name), shape(std::move(shape)), dtype(DType::Float32) {
        if (this->shape.empty()) {
            return;
        }

        int total = 1;
        for (int dim : this->shape) {
            total *= dim;
        }
        data.resize(total, 0.0f);
    }

    int numel() const {
        if (shape.empty()) return 0;

        int total = 1;
        for (int dim : shape) {
            total *= dim;
        }
        return total;
    }
};