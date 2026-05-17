import os
import time
import numpy as np
import torch
import coremltools as ct


class TinyMLP(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(4, 8)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(8, 2)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))


os.makedirs("models", exist_ok=True)

model = TinyMLP().eval()

example_input = torch.randn(1, 4)

print("Tracing TinyMLP...")
traced = torch.jit.trace(model, example_input)

print("Converting TinyMLP to CoreML...")
mlmodel = ct.convert(
    traced,
    inputs=[
        ct.TensorType(
            name="input",
            shape=example_input.shape,
        )
    ],
    outputs=[
        ct.TensorType(name="output")
    ],
    minimum_deployment_target=ct.target.macOS13,
)

output_path = "models/tiny_mlp.mlpackage"

print(f"Saving CoreML model to {output_path}...")
mlmodel.save(output_path)

print("Running CoreML prediction...")

x = np.random.randn(1, 4).astype(np.float32)

t1 = time.perf_counter()
out = mlmodel.predict({"input": x})
t2 = time.perf_counter()

print("CoreML output:")
print(out)

print(f"CoreML latency: {(t2 - t1) * 1000:.4f} ms")

print("CoreML TinyMLP conversion + execution complete.")