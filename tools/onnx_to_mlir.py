import onnx
import numpy as np
from onnx import numpy_helper


def shape_to_str(shape):
    return ",".join(str(x) for x in shape)


def get_attr(node, name, default):
    for attr in node.attribute:
        if attr.name == name:
            return attr.i
    return default


def main():
    model = onnx.load("models/tiny_mlp.onnx")
    graph = model.graph

    initializers = {
        init.name: numpy_helper.to_array(init).astype(np.float32)
        for init in graph.initializer
    }

    used_initializers = set()

    with open("models/tiny_mlp.mlir", "w") as f:
        f.write("# mini-ml-compiler IR\n")

        for inp in graph.input:
            if inp.name in initializers:
                continue

            dims = []
            tensor_type = inp.type.tensor_type
            for d in tensor_type.shape.dim:
                dims.append(d.dim_value if d.dim_value > 0 else 1)

            f.write(f"TENSOR {inp.name} {shape_to_str(dims)}\n")

        for node in graph.node:
            if node.op_type == "Gemm":
                A, B, Bias = node.input
                Out = node.output[0]

                weight = initializers[B]
                bias = initializers[Bias]

                trans_b = get_attr(node, "transB", 0)

                if trans_b == 1:
                    weight = weight.T

                if bias.ndim == 1:
                    bias = bias.reshape(1, bias.shape[0])

                weight_name = B + ".runtime"
                bias_name = Bias + ".runtime"

                for name, arr in [(weight_name, weight), (bias_name, bias)]:
                    flat = arr.flatten()
                    values = ",".join(str(float(x)) for x in flat)
                    f.write(f"CONST {name} {shape_to_str(list(arr.shape))} {values}\n")

                f.write(f"NODE MatMul {A} {weight_name} -> {Out}.matmul\n")
                f.write(f"NODE Add {Out}.matmul {bias_name} -> {Out}\n")

                used_initializers.add(B)
                used_initializers.add(Bias)

            elif node.op_type == "Relu":
                inputs = " ".join(node.input)
                outputs = " ".join(node.output)
                f.write(f"NODE ReLU {inputs} -> {outputs}\n")

            else:
                print(f"Unsupported op: {node.op_type}")

    print("Converted models/tiny_mlp.onnx -> models/tiny_mlp.mlir")


if __name__ == "__main__":
    main()