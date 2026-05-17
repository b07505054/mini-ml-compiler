import os
import time
import numpy as np
import torch
import coremltools as ct
from transformers import AutoModel

MODEL_NAME = "hf-internal-testing/tiny-random-bert"

os.makedirs("models", exist_ok=True)

print(f"Loading {MODEL_NAME}...")

model = AutoModel.from_pretrained(MODEL_NAME)
model.eval()


class BertEncoderFloatWrapper(torch.nn.Module):
    def __init__(self, bert):
        super().__init__()
        self.bert = bert

    def forward(self, inputs_embeds):
        outputs = self.bert(
            inputs_embeds=inputs_embeds,
            attention_mask=None
        )
        return outputs.last_hidden_state


wrapper = BertEncoderFloatWrapper(model)
wrapper.eval()

hidden_size = model.config.hidden_size
seq_len = 8

example_input = torch.randn(
    1,
    seq_len,
    hidden_size,
    dtype=torch.float32
)

print("Tracing BERT encoder with float inputs_embeds...")

traced = torch.jit.trace(
    wrapper,
    example_input
)

print("Converting BERT encoder float-input path to CoreML...")

mlmodel = ct.convert(
    traced,
    inputs=[
        ct.TensorType(
            name="inputs_embeds",
            shape=example_input.shape,
        )
    ],
    outputs=[
        ct.TensorType(name="last_hidden_state")
    ],
    minimum_deployment_target=ct.target.macOS13,
)

output_path = "models/bert_encoder_float_coreml.mlpackage"

print(f"Saving CoreML model to {output_path}...")
mlmodel.save(output_path)

print("Running CoreML prediction...")

x = np.random.randn(
    1,
    seq_len,
    hidden_size
).astype(np.float32)

t1 = time.perf_counter()
out = mlmodel.predict({"inputs_embeds": x})
t2 = time.perf_counter()

print("CoreML output keys:")
print(out.keys())

print("Output shape:")
print(out["last_hidden_state"].shape)

print(f"CoreML latency: {(t2 - t1) * 1000:.4f} ms")

print("BERT encoder float-input CoreML conversion complete.")