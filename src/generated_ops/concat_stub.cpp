#include "generated_ops/concat_stub.h"

#include <iostream>

void concat_kernel_stub(
    const Graph& graph,
    const Node& node
) {
    std::cout << "[ConcatStub] Runtime kernel is not implemented yet.\n";
    std::cout << "[ConcatStub] Node: " << node.name << "\n";

    // TODO:
    // - Read input tensors from graph.get_tensor(node.inputs[i]).
    // - Allocate and write output tensors through graph.get_tensor(node.outputs[i]).
    // - Implement operator semantics.
}

void concat_shape_inference_stub(
    Graph& graph,
    const Node& node
) {
    std::cout << "[ConcatStub] Shape inference is not implemented yet.\n";

    // TODO:
    // - Infer output shapes from input tensor shapes.
    // - Example:
    //   auto& input = graph.get_tensor(node.inputs[0]);
    //   auto& output = graph.get_tensor(node.outputs[0]);
    //   output.shape = input.shape;
}

bool concat_verify_stub(
    const Graph& graph,
    const Node& node
) {
    std::cout << "[ConcatStub] Verifier is not implemented yet.\n";

    // TODO:
    // - Check input count.
    // - Check output count.
    // - Check supported tensor ranks and shape constraints.
    // - Return false if unsupported.

    return true;
}
