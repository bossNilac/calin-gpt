import torch
from torch import nn

from src.model.layers import LayerNorm, MLP
from week3.day6.main import MultiheadAttention


class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.layer_norm1 = LayerNorm(d_model)
        self.multihead_attn = MultiheadAttention(d_model, n_heads)
        self.layer_norm2 = LayerNorm(d_model)
        self.mlp = MLP(d_model)

    def forward(self, x):
        normalised = self.layer_norm1(x)
        x_1 = x + self.multihead_attn(normalised, normalised, normalised)
        x_2 = x_1 + self.mlp(self.layer_norm2(x_1))

        return x_2

def test_gradients():
    block = TransformerBlock(128, 4)
    x = torch.randn(2, 16, 128, requires_grad=True)

    y = block(x)

    print("Input shape:", x.shape)
    print("Output shape:", y.shape)
    print("Shapes match:", y.shape == x.shape)

    y.sum().backward()

    print("Input gradient:", x.grad is not None)
    print("Attention gradients:", all(p.grad is not None for p in block.multihead_attn.parameters()))
    print("MLP gradients:", all(p.grad is not None for p in block.mlp.parameters()))
    print("LayerNorm gradients:", all(
        p.grad is not None
        for ln in [block.layer_norm1, block.layer_norm2]
        for p in ln.parameters()
    ))

def test_causality():
    block = TransformerBlock(128, 4)
    block.eval()

    x = torch.randn(2, 16, 128)
    x_modified = x.clone()

    x_modified[:, 10, :] = torch.randn_like(x_modified[:, 10, :])

    with torch.no_grad():
        y1 = block(x)
        y2 = block(x_modified)

    print("Earlier positions unchanged:",
          torch.allclose(y1[:, :10, :], y2[:, :10, :], atol=1e-5))

    print("Changed position affected:",
          not torch.allclose(y1[:, 10, :], y2[:, 10, :]))

if __name__ == '__main__':
    test_gradients()
    test_causality()

