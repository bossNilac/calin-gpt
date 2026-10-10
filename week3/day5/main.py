from math import sqrt

import torch

class CausalSelfAttentionHead(torch.nn.Module):
    def __init__(self, C, D):
        super().__init__()
        self.Q = torch.nn.Linear(C, D, bias=False)
        self.K = torch.nn.Linear(C, D, bias=False)
        self.V = torch.nn.Linear(C, D, bias=False)

    def forward(self, x):
        q = self.Q(x)
        k = self.K(x)
        v = self.V(x)

        D = q.shape[-1]
        k_transpose = k.transpose(-2, -1)

        T = q.shape[-2]
        mask = torch.tril(torch.ones(T, T, device=q.device))
        S = torch.matmul(q, k_transpose, ).div(sqrt(D))
        S = S.masked_fill(mask == 0, float("-inf"))

        return torch.softmax(S, dim=-1).matmul(v)

if __name__ == "__main__":
    torch.manual_seed(42)

    model = CausalSelfAttentionHead(16, 8)
    x = torch.randn(2, 5, 16)

    output = model(x)

    print("Input shape:", x.shape)
    print("Output shape:", output.shape)
    print("Output:", output)

    loss = output.sum()
    loss.backward()

    print("Q gradient:", model.Q.weight.grad.norm().item())
    print("K gradient:", model.K.weight.grad.norm().item())
    print("V gradient:", model.V.weight.grad.norm().item())

    x2 = x.clone()
    x2[:, -1, :] = torch.randn_like(x2[:, -1, :])

    output2 = model(x2)
    print("Causal masking works:",
          torch.allclose(output[:, :-1], output2[:, :-1]))