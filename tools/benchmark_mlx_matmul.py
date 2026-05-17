import time
import numpy as np
import mlx.core as mx


def benchmark_numpy(size: int, runs: int):
    a = np.ones((size, size), dtype=np.float32)
    b = np.ones((size, size), dtype=np.float32)

    # Warmup
    _ = a @ b

    t1 = time.perf_counter()

    for _ in range(runs):
        c = a @ b

    t2 = time.perf_counter()

    return (t2 - t1) * 1000.0 / runs, c


def benchmark_mlx(size: int, runs: int):
    a = mx.ones((size, size), dtype=mx.float32)
    b = mx.ones((size, size), dtype=mx.float32)

    # Warmup
    c = a @ b
    mx.eval(c)

    t1 = time.perf_counter()

    for _ in range(runs):
        c = a @ b
        mx.eval(c)

    t2 = time.perf_counter()

    return (t2 - t1) * 1000.0 / runs, c


def main():
    size = 2048
    runs = 20

    print("=== MLX MatMul Benchmark ===")
    print(f"Matrix size: {size}x{size}")
    print(f"Runs: {runs}")
    print(f"MLX default device: {mx.default_device()}")

    numpy_ms, numpy_out = benchmark_numpy(size, runs)
    mlx_ms, mlx_out = benchmark_mlx(size, runs)

    print(f"NumPy avg latency: {numpy_ms:.4f} ms")
    print(f"MLX avg latency: {mlx_ms:.4f} ms")
    print(f"Speedup vs NumPy: {numpy_ms / mlx_ms:.2f}x")

    print("Output check:")
    print("NumPy[0,0]:", float(numpy_out[0, 0]))
    print("MLX[0,0]:", float(np.array(mlx_out)[0, 0]))


if __name__ == "__main__":
    main()