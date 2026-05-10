import onnx
from collections import Counter

MODEL_PATH = "models/bert_tiny.onnx"

model = onnx.load(MODEL_PATH)
graph = model.graph

op_counts = Counter(node.op_type for node in graph.node)

print("=== ONNX Graph Inspection ===")
print(f"Model: {MODEL_PATH}")
print(f"Inputs: {len(graph.input)}")
print(f"Outputs: {len(graph.output)}")
print(f"Nodes: {len(graph.node)}")
print(f"Initializers: {len(graph.initializer)}")

print("\nOperator counts:")
for op, count in op_counts.most_common():
    print(f"  {op}: {count}")

print("\nInputs:")
for inp in graph.input:
    dims = []
    for d in inp.type.tensor_type.shape.dim:
        dims.append(d.dim_param if d.dim_param else d.dim_value)
    print(f"  {inp.name}: {dims}")

print("\nOutputs:")
for out in graph.output:
    dims = []
    for d in out.type.tensor_type.shape.dim:
        dims.append(d.dim_param if d.dim_param else d.dim_value)
    print(f"  {out.name}: {dims}")