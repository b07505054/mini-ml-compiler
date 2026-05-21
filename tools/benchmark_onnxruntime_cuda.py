import json
import time
from pathlib import Path

import numpy as np
import onnx
import onnx.helper as helper
import onnx.numpy_helper as numpy_helper
import onnxruntime as ort


MODEL_PATH = Path("models/matmul_add_relu.onnx")
TRACE_PATH = Path("trace/onnxruntime_cuda_benchmark.json")


def build_model():
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    input_tensor = helper.make_tensor_value_info(
        "input",
        onnx.TensorProto.FLOAT,
        [1, 2048],
    )

    output_tensor = helper.make_tensor_value_info(
        "output",
        onnx.TensorProto.FLOAT,
        [1, 2048],
    )

    weight = np.random.randn(2048, 2048).astype(np.float32)
    bias = np.random.randn(1, 2048).astype(np.float32)

    weight_init = numpy_helper.from_array(weight, name="weight")
    bias_init = numpy_helper.from_array(bias, name="bias")

    matmul = helper.make_node(
        "MatMul",
        ["input", "weight"],
        ["matmul_out"],
        name="matmul",
    )

    add = helper.make_node(
        "Add",
        ["matmul_out", "bias"],
        ["add_out"],
        name="add",
    )

    relu = helper.make_node(
        "Relu",
        ["add_out"],
        ["output"],
        name="relu",
    )

    graph = helper.make_graph(
        [matmul, add, relu],
        "MatMulAddReluGraph",
        [input_tensor],
        [output_tensor],
        [weight_init, bias_init],
    )

    model = helper.make_model(
        graph,
        producer_name="mini-ml-compiler",
    )

    model.ir_version = 10

    for opset in model.opset_import:
        opset.version = 17

    onnx.save(model, MODEL_PATH)

    print(f"Saved ONNX model to {MODEL_PATH}")


def make_session(provider):
    return ort.InferenceSession(
        str(MODEL_PATH),
        providers=[provider, "CPUExecutionProvider"],
    )


def benchmark(provider, runs=1000, warmup=50):
    print(f"\n=== Benchmark {provider} ===")

    session = make_session(provider)

    actual_provider = session.get_providers()[0]

    x = np.random.randn(1, 2048).astype(np.float32)

    for _ in range(warmup):
        session.run(None, {"input": x})

    latencies = []

    for _ in range(runs):
        t0 = time.perf_counter()
        out = session.run(None, {"input": x})
        t1 = time.perf_counter()

        latencies.append((t1 - t0) * 1000.0)

    latencies = np.array(latencies)

    result = {
        "requested_provider": provider,
        "actual_provider": actual_provider,
        "runs": runs,
        "warmup": warmup,
        "avg_ms": float(latencies.mean()),
        "p50_ms": float(np.percentile(latencies, 50)),
        "p95_ms": float(np.percentile(latencies, 95)),
        "p99_ms": float(np.percentile(latencies, 99)),
        "output_checksum": float(np.sum(out[0])),
    }

    print(json.dumps(result, indent=2))

    return result


def main():
    TRACE_PATH.parent.mkdir(parents=True, exist_ok=True)

    build_model()

    available = ort.get_available_providers()

    print("Available providers:", available)

    requested = [
        "CPUExecutionProvider",
        "CUDAExecutionProvider",
        "TensorrtExecutionProvider",
    ]

    results = []

    for provider in requested:
        if provider not in available:
            print(f"Skipping {provider}: not available")
            continue

        try:
            results.append(benchmark(provider))
        except Exception as e:
            print(f"Failed {provider}: {e}")

    with open(TRACE_PATH, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved benchmark report to {TRACE_PATH}")


if __name__ == "__main__":
    main()