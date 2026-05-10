import onnx
from collections import Counter

MODEL_PATH = "models/bert_tiny.onnx"

SUPPORTED_OPS = {
    "MatMul",
    "Add",
    "Relu",
    "ReLU",
    "Gemm",
    "Softmax",
}

PARTIAL_SUPPORT_OPS = {
    "Reshape",
    "Transpose",
    "Mul",
    "Div",
}

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

print("=== ONNX Operator Support Report ===")
print(f"Model: {MODEL_PATH}")
print(f"Total nodes: {total_nodes}")
print(f"Supported nodes: {supported_nodes}")
print(f"Partially supported nodes: {partial_nodes}")
print(f"Unsupported nodes: {unsupported_nodes}")

print("\nSupported ops:")
for op, count in sorted(supported.items()):
    print(f"  {op}: {count}")

print("\nPartially supported ops:")
for op, count in sorted(partial.items()):
    print(f"  {op}: {count}")

print("\nUnsupported ops:")
for op, count in sorted(unsupported.items()):
    print(f"  {op}: {count}")

print("\nTop unsupported ops:")
for op, count in sorted(unsupported.items(), key=lambda x: x[1], reverse=True)[:10]:
    print(f"  {op}: {count}")

coverage = 100.0 * (supported_nodes + partial_nodes) / total_nodes

print(f"\nApprox operator coverage: {coverage:.2f}%")