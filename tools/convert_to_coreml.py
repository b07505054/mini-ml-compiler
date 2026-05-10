import torch
import coremltools as ct
from transformers import AutoModel

MODEL_NAME = "hf-internal-testing/tiny-random-bert"

print(f"Loading {MODEL_NAME}...")

model = AutoModel.from_pretrained(MODEL_NAME)
model.eval()

input_ids = torch.tensor([[1, 2, 3, 4, 5, 6, 0, 0]], dtype=torch.long)
attention_mask = torch.tensor([[1, 1, 1, 1, 1, 1, 0, 0]], dtype=torch.long)

class BertWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        return outputs.last_hidden_state

wrapper = BertWrapper(model)
wrapper.eval()

print("Tracing PyTorch model...")

traced = torch.jit.trace(
    wrapper,
    (input_ids, attention_mask)
)

print("Converting TorchScript model to CoreML...")

mlmodel = ct.convert(
    traced,
    inputs=[
        ct.TensorType(name="input_ids", shape=input_ids.shape, dtype=int),
        ct.TensorType(name="attention_mask", shape=attention_mask.shape, dtype=int),
    ],
    outputs=[
        ct.TensorType(name="last_hidden_state")
    ],
    minimum_deployment_target=ct.target.iOS17,
)

output_path = "models/bert_tiny_coreml.mlpackage"

print(f"Saving CoreML model to {output_path}...")

mlmodel.save(output_path)

print("CoreML conversion complete.")