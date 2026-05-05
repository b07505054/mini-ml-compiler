#include "analysis/graph_verifier.h"

#include <iostream>

bool GraphVerifier::verify(const Graph& graph) const {
    bool ok = true;

    for (const auto& node : graph.nodes) {
        if (node.outputs.empty()) {
            std::cerr << "[Verifier] Node has no output: " << node.name << "\n";
            ok = false;
        }

        for (int input_id : node.inputs) {
            if (input_id < 0 || input_id >= static_cast<int>(graph.tensors.size())) {
                std::cerr << "[Verifier] Invalid input tensor id in node: " << node.name << "\n";
                ok = false;
            }
        }

        for (int output_id : node.outputs) {
            if (output_id < 0 || output_id >= static_cast<int>(graph.tensors.size())) {
                std::cerr << "[Verifier] Invalid output tensor id in node: " << node.name << "\n";
                ok = false;
            }
        }

        if (node.op == OpType::MatMul) {
            const auto& A = graph.get_tensor(node.inputs[0]);
            const auto& B = graph.get_tensor(node.inputs[1]);
            const auto& C = graph.get_tensor(node.outputs[0]);

            if (A.shape.size() != 2 || B.shape.size() != 2 || C.shape.size() != 2) {
                std::cerr << "[Verifier] MatMul expects 2D tensors\n";
                ok = false;
            } else if (A.shape[1] != B.shape[0]) {
                std::cerr << "[Verifier] MatMul inner dimension mismatch\n";
                ok = false;
            } else if (C.shape[0] != A.shape[0] || C.shape[1] != B.shape[1]) {
                std::cerr << "[Verifier] MatMul output shape mismatch\n";
                ok = false;
            }
        }

        if (node.op == OpType::Add) {
            const auto& A = graph.get_tensor(node.inputs[0]);
            const auto& B = graph.get_tensor(node.inputs[1]);
            const auto& C = graph.get_tensor(node.outputs[0]);

            if (A.shape != B.shape || A.shape != C.shape) {
                std::cerr << "[Verifier] Add shape mismatch\n";
                ok = false;
            }
        }

        if (node.op == OpType::ReLU) {
            const auto& A = graph.get_tensor(node.inputs[0]);
            const auto& B = graph.get_tensor(node.outputs[0]);

            if (A.shape != B.shape) {
                std::cerr << "[Verifier] ReLU shape mismatch\n";
                ok = false;
            }
        }

        if (node.op == OpType::FusedMatMulAddReLU) {
            const auto& A = graph.get_tensor(node.inputs[0]);
            const auto& B = graph.get_tensor(node.inputs[1]);
            const auto& Bias = graph.get_tensor(node.inputs[2]);
            const auto& Out = graph.get_tensor(node.outputs[0]);

            if (A.shape.size() != 2 || B.shape.size() != 2 || Out.shape.size() != 2) {
                std::cerr << "[Verifier] Fused op expects 2D tensors\n";
                ok = false;
            } else if (A.shape[1] != B.shape[0]) {
                std::cerr << "[Verifier] Fused MatMul inner dimension mismatch\n";
                ok = false;
            } else if (Out.shape[0] != A.shape[0] || Out.shape[1] != B.shape[1]) {
                std::cerr << "[Verifier] Fused output shape mismatch\n";
                ok = false;
            } else if (Bias.shape != Out.shape) {
                std::cerr << "[Verifier] Fused bias shape mismatch\n";
                ok = false;
            }
        }
    }

    if (ok) {
        std::cout << "[Verifier] Graph verification passed\n";
    }

    return ok;
}