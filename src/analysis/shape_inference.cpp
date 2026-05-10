#include "analysis/shape_inference.h"

#include <iostream>

void ShapeInference::run(Graph& graph) {
    std::cout << "[ShapeInference] Running shape inference\n";

    for (const auto& node : graph.nodes) {
        if (node.op == OpType::MatMul) {
            auto& A = graph.get_tensor(node.inputs[0]);
            auto& B = graph.get_tensor(node.inputs[1]);
            auto& C = graph.get_tensor(node.outputs[0]);

            int M = A.shape[0];
            int N = B.shape[1];

            C.shape = {M, N};
        }

        if (node.op == OpType::Add) {
            auto& A = graph.get_tensor(node.inputs[0]);
            auto& C = graph.get_tensor(node.outputs[0]);

            C.shape = A.shape;
        }

        if (node.op == OpType::ReLU) {
            auto& A = graph.get_tensor(node.inputs[0]);
            auto& B = graph.get_tensor(node.outputs[0]);

            B.shape = A.shape;
        }

        if (node.op == OpType::FusedMatMulAddReLU) {
            auto& A = graph.get_tensor(node.inputs[0]);
            auto& B = graph.get_tensor(node.inputs[1]);
            auto& Out = graph.get_tensor(node.outputs[0]);

            int M = A.shape[0];
            int N = B.shape[1];

            Out.shape = {M, N};
        }

        if (node.op == OpType::Attention) {
            auto& Q = graph.get_tensor(node.inputs[0]);
            auto& Out = graph.get_tensor(node.outputs[0]);

            Out.shape = Q.shape;
        }
        
        if (node.op == OpType::CausalAttention) {
            auto& Q = graph.get_tensor(node.inputs[0]);
            auto& Out = graph.get_tensor(node.outputs[0]);

            Out.shape = Q.shape;
        }

        if (node.op == OpType::FusedAttention) {
            auto& Q = graph.get_tensor(node.inputs[0]);
            auto& Out = graph.get_tensor(node.outputs[0]);

            Out.shape = Q.shape;
        }
    }
}