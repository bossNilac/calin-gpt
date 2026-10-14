
import torch

from src.optim.adamw import AdamW


p1 = torch.nn.Parameter(torch.tensor([1.0, -2.0, 3.0]))
p2 = torch.nn.Parameter(p1.detach().clone())

custom = AdamW([p1], lr=0.001, weight_decay=0.01)
pytorch = torch.optim.AdamW([p2], lr=0.001, weight_decay=0.01)

for step in range(10):
    grad = torch.tensor([0.1, -0.2, 0.3])

    p1.grad = grad.clone()
    p2.grad = grad.clone()

    custom.step()
    pytorch.step()

    print(
        f"Step {step + 1}: "
        f"Max difference = {(p1 - p2).abs().max().item():.10f}"
    )

print("Custom parameters:", p1.detach())
print("PyTorch parameters:", p2.detach())
print("Match:", torch.allclose(p1, p2, atol=1e-6))

assert torch.allclose(p1, p2, atol=1e-6)
