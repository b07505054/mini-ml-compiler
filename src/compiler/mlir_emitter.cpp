#include "compiler/mlir_emitter.h"

#include <sstream>

static std::string shape_to_mlir(const Tensor& tensor) {
    std::stringstream ss;

    for (size_t i = 0; i < tensor.shape.size(); ++i) {
        ss << tensor.shape[i];

        if (i + 1 < tensor.shape.size()) {
            ss << "x";
        }
    }

    ss << "xf32";

    return ss.str();
}

std::string MLIREmitter::emit(const Graph& graph) const {
    std::stringstream ss;

    ss << "module {\n";
    ss << "  func.func @main(";

    bool first = true;

    for (const auto& tensor : graph.tensors) {
        if (!tensor.persistent) {
            if (!first) {
                ss << ", ";
            }

            ss << "%" << tensor.name
               << ": memref<"
               << shape_to_mlir(tensor)
               << ">";

            first = false;
        }
    }

    ss << ") {\n";

    for (const auto& node : graph.nodes) {
        if (node.op == OpType::MatMul) {
            const auto& A = graph.get_tensor(node.inputs[0]);
            const auto& B = graph.get_tensor(node.inputs[1]);
            const auto& C = graph.get_tensor(node.outputs[0]);

            ss << "    linalg.matmul ins("
               << "%" << A.name
               << ", "
               << "%" << B.name
               << " : memref<"
               << shape_to_mlir(A)
               << ">, memref<"
               << shape_to_mlir(B)
               << ">) outs("
               << "%" << C.name
               << " : memref<"
               << shape_to_mlir(C)
               << ">)\n";
        }
    }

    ss << "    return\n";
    ss << "  }\n";
    ss << "}\n";

    return ss.str();
}