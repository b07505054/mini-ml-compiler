import os
import time
import numpy as np
import torch
import coremltools as ct
from transformers import AutoModel

MODEL_NAME = "hf-internal-testing/tiny-random-bert"

os.makedirs("models", exist_ok=True)

print(f"Loading {MODEL_NAME}...")

bert = AutoModel.from_pretrained(MODEL_NAME)
bert.eval()


class EncoderOnlyWrapper(torch.nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, hidden_states):
        outputs = self.encoder(
            hidden_states
        )

        return outputs.last_hidden_state


wrapper = EncoderOnlyWrapper(
    bert.encoder
)

wrapper.eval()

hidden_size = bert.config.hidden_size
seq_len = 8

example_input = torch.randn(
    1,
    seq_len,
    hidden_size
)

print("Tracing encoder-only model...")

traced = torch.jit.trace(
    wrapper,
    example_input
)

print("Converting encoder-only model to CoreML...")

mlmodel = ct.convert(
    traced,
    inputs=[
        ct.TensorType(
            name="hidden_states",
            shape=example_input.shape,
        )
    ],
    outputs=[
        ct.TensorType(
            name="last_hidden_state"
        )
    ],
    minimum_deployment_target=ct.target.macOS13,
)

output_path = (
    "models/bert_encoder_only.mlpackage"
)

print(f"Saving to {output_path}...")

mlmodel.save(output_path)

print("Running CoreML prediction...")

x = np.random.randn(
    1,
    seq_len,
    hidden_size
).astype(np.float32)

t1 = time.perf_counter()

out = mlmodel.predict({
    "hidden_states": x
})

t2 = time.perf_counter()

print("Output keys:")
print(out.keys())

print("Output shape:")
print(
    out["last_hidden_state"].shape
)

print(
    f"CoreML latency: "
    f"{(t2 - t1) * 1000:.4f} ms"
)

print(
    "Encoder-only CoreML conversion complete."
)