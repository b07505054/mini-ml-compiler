#include "ir/graph.h"
#include "analysis/shape_inference.h"
#include "compiler/mlir_emitter.h"

#include <fstream>
#include <iostream>

int main() {
    Graph graph;

    int A = graph.add_tensor(Tensor("A", {128, 128}));
    int B = graph.add_tensor(Tensor("B", {128, 128}));
    int C = graph.add_tensor(Tensor("C", {128, 128}));

    graph.add_node(
        Node(
            "matmul",
            OpType::MatMul,
            {A, B},
            {C}
        )
    );

    ShapeInference infer;
    infer.run(graph);

    MLIREmitter emitter;

    std::string mlir = emitter.emit(graph);

    std::ofstream out("generated_graph.mlir");
    out << mlir;

    std::cout << "=== Generated MLIR ===\n";
    std::cout << mlir << "\n";

    std::cout << "Saved to generated_graph.mlir\n";

    return 0;
}