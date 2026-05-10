import torch
from transformers import AutoModel

MODEL_NAME = "hf-internal-testing/tiny-random-bert"

print(f"Loading {MODEL_NAME}...")

model = AutoModel.from_pretrained(MODEL_NAME)
model.eval()

input_ids = torch.tensor([[1, 2, 3, 4, 5, 6, 0, 0]], dtype=torch.long)
attention_mask = torch.tensor([[1, 1, 1, 1, 1, 1, 0, 0]], dtype=torch.long)

with torch.no_grad():
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)

print("PyTorch output shape:")
print(outputs.last_hidden_state.shape)

print("Exporting ONNX...")

torch.onnx.export(
    model,
    (input_ids, attention_mask),
    "models/bert_tiny.onnx",
    input_names=["input_ids", "attention_mask"],
    output_names=["last_hidden_state"],
    dynamic_axes={
        "input_ids": {0: "batch", 1: "seq"},
        "attention_mask": {0: "batch", 1: "seq"},
        "last_hidden_state": {0: "batch", 1: "seq"}
    },
    opset_version=17
)

print("Exported models/bert_tiny.onnx")