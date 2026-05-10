import sys
from pathlib import Path

def snake_case(name: str) -> str:
    out = []
    for i, c in enumerate(name):
        if c.isupper() and i > 0:
            out.append("_")
        out.append(c.lower())
    return "".join(out)

def main():
    if len(sys.argv) != 2:
        print("Usage: python tools/generate_operator_stub.py <OpName>")
        sys.exit(1)

    op_name = sys.argv[1]
    op_snake = snake_case(op_name)

    include_dir = Path("include/generated_ops")
    src_dir = Path("src/generated_ops")

    include_dir.mkdir(parents=True, exist_ok=True)
    src_dir.mkdir(parents=True, exist_ok=True)

    header_path = include_dir / f"{op_snake}_stub.h"
    cpp_path = src_dir / f"{op_snake}_stub.cpp"

    header = f"""#pragma once

#include "ir/graph.h"
#include "ir/node.h"
#include "ir/tensor.h"

// Auto-generated operator onboarding stub for {op_name}.
//
// TODO:
// - Define runtime kernel semantics.
// - Add OpType::{op_name} to include/ir/node.h.
// - Register the operator in src/runtime/op_registry.cpp.
// - Add shape inference logic in src/analysis/shape_inference.cpp.
// - Add graph verification rules in src/analysis/graph_verifier.cpp.
// - Add unit/demo test under apps/.

void {op_snake}_kernel_stub(
    const Graph& graph,
    const Node& node
);

void {op_snake}_shape_inference_stub(
    Graph& graph,
    const Node& node
);

bool {op_snake}_verify_stub(
    const Graph& graph,
    const Node& node
);
"""

    cpp = f"""#include "generated_ops/{op_snake}_stub.h"

#include <iostream>

void {op_snake}_kernel_stub(
    const Graph& graph,
    const Node& node
) {{
    std::cout << "[{op_name}Stub] Runtime kernel is not implemented yet.\\n";
    std::cout << "[{op_name}Stub] Node: " << node.name << "\\n";

    // TODO:
    // - Read input tensors from graph.get_tensor(node.inputs[i]).
    // - Allocate and write output tensors through graph.get_tensor(node.outputs[i]).
    // - Implement operator semantics.
}}

void {op_snake}_shape_inference_stub(
    Graph& graph,
    const Node& node
) {{
    std::cout << "[{op_name}Stub] Shape inference is not implemented yet.\\n";

    // TODO:
    // - Infer output shapes from input tensor shapes.
    // - Example:
    //   auto& input = graph.get_tensor(node.inputs[0]);
    //   auto& output = graph.get_tensor(node.outputs[0]);
    //   output.shape = input.shape;
}}

bool {op_snake}_verify_stub(
    const Graph& graph,
    const Node& node
) {{
    std::cout << "[{op_name}Stub] Verifier is not implemented yet.\\n";

    // TODO:
    // - Check input count.
    // - Check output count.
    // - Check supported tensor ranks and shape constraints.
    // - Return false if unsupported.

    return true;
}}
"""

    header_path.write_text(header, encoding="utf-8")
    cpp_path.write_text(cpp, encoding="utf-8")

    print("Generated operator onboarding stubs:")
    print(f"  {header_path}")
    print(f"  {cpp_path}")

if __name__ == "__main__":
    main()