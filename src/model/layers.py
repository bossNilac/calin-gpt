import torch
from torch import nn


class LayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-5):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        var = (x - mean).pow(2).mean(-1, keepdim=True)
        normalised_x = (x - mean) / torch.sqrt(var + self.eps)
        output = self.gamma * normalised_x + self.beta
        return output

class MLP(nn.Module):

    def __init__(self, d_model, mlp_ratio=4):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_model*mlp_ratio)
        self.fc2 = nn.Linear(d_model*mlp_ratio,d_model)
        self.activation = nn.GELU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        return x


def test_layernorm():
    x = torch.randn(2, 8, 32, requires_grad=True)

    custom = LayerNorm(32)
    reference = nn.LayerNorm(32)

    y = custom(x)
    expected = reference(x)

    print("Shape:", y.shape)
    print("Matches PyTorch:", torch.allclose(y, expected, atol=1e-5))

    y.sum().backward()

    print("Gamma gradient:", custom.gamma.grad is not None)
    print("Beta gradient:", custom.beta.grad is not None)
    print("Input gradient:", x.grad is not None)

def test_mlp():
    mlp = MLP(128)
    x = torch.randn(2, 16, 128, requires_grad=True)

    y = mlp(x)

    print("Input shape:", x.shape)
    print("Output shape:", y.shape)
    print("Shapes match:", y.shape == x.shape)

    y.sum().backward()

    print("Input gradient:", x.grad is not None)
    print("fc1 weight gradient:", mlp.fc1.weight.grad is not None)
    print("fc2 weight gradient:", mlp.fc2.weight.grad is not None)
    print("All gradients finite:", all(
        p.grad is not None and torch.isfinite(p.grad).all().item()
        for p in mlp.parameters()
    ))
if __name__ == '__main__':
    test_layernorm()
    test_mlp()