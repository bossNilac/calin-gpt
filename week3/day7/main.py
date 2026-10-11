import torch
from week3.day6.main import MultiheadAttention

torch.manual_seed(42)

for h in (1, 4, 8):
    model = MultiheadAttention(64, h)
    x = torch.randn(2, 10, 64)

    y = model(x, x, x)
    assert y.shape == x.shape

    y.sum().backward()
    assert all(p.grad is not None for p in model.parameters())

    x2 = x.clone()
    x2[:, -1] += 10
    y2 = model(x2, x2, x2)

    assert torch.allclose(y[:, :-1], y2[:, :-1], atol=1e-6)
    print(f"{h} heads: passed")