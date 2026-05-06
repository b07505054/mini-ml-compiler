#include "runtime/op_registry.h"
#include "kernels/cpu_kernels.h"

OpRegistry create_default_registry() {
    OpRegistry registry;

    registry.register_op(OpType::Input, [](Graph&, const Node&) {
        // Input node does not execute computation.
    });

    registry.register_op(OpType::MatMul, [](Graph& graph, const Node& node) {
        auto& A = graph.get_tensor(node.inputs[0]);
        auto& B = graph.get_tensor(node.inputs[1]);
        auto& C = graph.get_tensor(node.outputs[0]);
        matmul(A, B, C);
    });

    registry.register_op(OpType::Add, [](Graph& graph, const Node& node) {
        auto& A = graph.get_tensor(node.inputs[0]);
        auto& B = graph.get_tensor(node.inputs[1]);
        auto& C = graph.get_tensor(node.outputs[0]);
        add(A, B, C);
    });

    registry.register_op(OpType::ReLU, [](Graph& graph, const Node& node) {
        auto& A = graph.get_tensor(node.inputs[0]);
        auto& B = graph.get_tensor(node.outputs[0]);
        relu(A, B);
    });

    registry.register_op(OpType::FusedMatMulAddReLU, [](Graph& graph, const Node& node) {
        auto& A = graph.get_tensor(node.inputs[0]);
        auto& B = graph.get_tensor(node.inputs[1]);
        auto& Bias = graph.get_tensor(node.inputs[2]);
        auto& Out = graph.get_tensor(node.outputs[0]);
        fused_matmul_add_relu(A, B, Bias, Out);
    });

    registry.register_op(OpType::Attention, [](Graph& graph, const Node& node) {
        auto& Q = graph.get_tensor(node.inputs[0]);
        auto& K = graph.get_tensor(node.inputs[1]);
        auto& V = graph.get_tensor(node.inputs[2]);
        auto& Out = graph.get_tensor(node.outputs[0]);

        attention(Q, K, V, Out);
    });
    
    registry.register_op(OpType::CausalAttention, [](Graph& graph, const Node& node) {
        auto& Q = graph.get_tensor(node.inputs[0]);
        auto& K = graph.get_tensor(node.inputs[1]);
        auto& V = graph.get_tensor(node.inputs[2]);
        auto& Out = graph.get_tensor(node.outputs[0]);

        causal_attention(Q, K, V, Out);
    });
    return registry;
}