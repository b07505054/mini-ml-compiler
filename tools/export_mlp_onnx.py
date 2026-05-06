import torch
import torch.nn as nn


class TinyMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 4)
        self.relu = nn.ReLU()

        with torch.no_grad():
            self.fc1.weight.copy_(torch.tensor([
                [-0.4,  0.1, -0.3,  0.5],
                [-0.1, -0.3, -0.2, -0.4],
                [ 0.1,  0.0, -0.4, -0.4],
                [-0.2,  0.4,  0.1,  0.2],
            ]))

            self.fc1.bias.copy_(torch.tensor([
                0.1,
                -0.1,
                -0.1,
                0.1
            ]))

    def forward(self, x):
        return self.relu(self.fc1(x))


def main():
    model = TinyMLP()
    model.eval()

    x = torch.tensor([[1.0, 1.0, 1.0, 1.0]])

    with torch.no_grad():
        y = model(x)

    print("PyTorch output:")
    print(y)

    torch.onnx.export(
        model,
        x,
        "models/tiny_mlp.onnx",
        input_names=["input"],
        output_names=["output"],
        opset_version=17
    )

    print("Exported models/tiny_mlp.onnx")


if __name__ == "__main__":
    main()