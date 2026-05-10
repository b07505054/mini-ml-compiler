import onnx
from collections import Counter
from pathlib import Path

MODEL_PATH = "models/bert_tiny.onnx"
REPORT_PATH = "reports/bert_tiny_integration_report.md"

SUPPORTED_OPS = {
    "MatMul",
    "Add",
    "Gemm",
    "Softmax",
    "LayerNormalization",
}

PARTIAL_SUPPORT_OPS = {
    "Reshape",
    "Transpose",
    "Mul",
    "Div",
}

OP_PRIORITY = {
    "LayerNormalization": "Completed",
    "Slice": "High",
    "Concat": "High",
    "Shape": "Medium",
    "Unsqueeze": "Medium",
    "Where": "Medium",
    "Gather": "Medium",
    "Erf": "Medium",
    "Expand": "Low",
    "Cast": "Low",
}

def main():
    Path("reports").mkdir(exist_ok=True)

    model = onnx.load(MODEL_PATH)
    graph = model.graph

    op_counts = Counter(node.op_type for node in graph.node)

    supported = {}
    partial = {}
    unsupported = {}

    for op, count in op_counts.items():
        if op in SUPPORTED_OPS:
            supported[op] = count
        elif op in PARTIAL_SUPPORT_OPS:
            partial[op] = count
        else:
            unsupported[op] = count

    total_nodes = sum(op_counts.values())
    supported_nodes = sum(supported.values())
    partial_nodes = sum(partial.values())
    unsupported_nodes = sum(unsupported.values())

    coverage = 100.0 * (supported_nodes + partial_nodes) / total_nodes

    lines = []

    lines.append("# BERT Tiny ONNX Integration Report\n")
    lines.append("## Model Summary\n")
    lines.append(f"- Model path: `{MODEL_PATH}`")
    lines.append(f"- Inputs: {len(graph.input)}")
    lines.append(f"- Outputs: {len(graph.output)}")
    lines.append(f"- Total nodes: {total_nodes}")
    lines.append(f"- Initializers: {len(graph.initializer)}")
    lines.append(f"- Approx operator coverage: **{coverage:.2f}%**\n")

    lines.append("## Supported Operators\n")
    for op, count in sorted(supported.items()):
        lines.append(f"- `{op}`: {count}")
    lines.append("")

    lines.append("## Partially Supported Operators\n")
    for op, count in sorted(partial.items()):
        lines.append(f"- `{op}`: {count}")
    lines.append("")

    lines.append("## Unsupported Operators\n")
    for op, count in sorted(unsupported.items(), key=lambda x: x[1], reverse=True):
        priority = OP_PRIORITY.get(op, "Unassigned")
        lines.append(f"- `{op}`: {count} | Priority: {priority}")
    lines.append("")

    lines.append("## Top Integration Gaps\n")
    for op, count in sorted(unsupported.items(), key=lambda x: x[1], reverse=True)[:8]:
        priority = OP_PRIORITY.get(op, "Unassigned")
        lines.append(f"- `{op}` appears {count} times and is marked as `{priority}` priority.")
    lines.append("")

    lines.append("## Recommended Operator Onboarding Plan\n")
    lines.append("1. Add `Slice` and `Concat` because they are high-frequency graph-structure operators.")
    lines.append("2. Add `Shape`, `Unsqueeze`, and `Gather` to improve dynamic-shape graph support.")
    lines.append("3. Add `Where` and `Erf` to support more Transformer activation and masking patterns.")
    lines.append("4. Continue validating each operator with unit-level correctness tests and ONNX graph coverage reports.")
    lines.append("")

    lines.append("## Engineering Notes\n")
    lines.append("- `LayerNormalization` has been onboarded into the runtime as a CPU kernel.")
    lines.append("- `MatMul`, `Add`, `Gemm`, and `Softmax` map to existing runtime operators.")
    lines.append("- This report is intended to guide incremental model integration and runtime operator coverage expansion.")

    Path(REPORT_PATH).write_text("\n".join(lines), encoding="utf-8")

    print(f"Generated {REPORT_PATH}")
    print(f"Approx operator coverage: {coverage:.2f}%")
    print(f"Unsupported operators: {unsupported_nodes}")

if __name__ == "__main__":
    main()