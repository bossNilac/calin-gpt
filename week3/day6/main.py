from math import sqrt

import torch
import torch.nn as nn


def scaled_dot_product_attention(q, k, v):
    D = q.shape[-1]
    k_transpose = k.transpose(-2, -1)

    T = q.shape[-2]
    mask = torch.tril(torch.ones(T, T, device=q.device))
    S = torch.matmul(q, k_transpose, ).div(sqrt(D))
    S = S.masked_fill(mask == 0, float("-inf"))

    return torch.softmax(S, dim=-1).matmul(v)


class MultiheadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(MultiheadAttention, self).__init__()
        assert embed_dim % num_heads == 0, "Embedding dimension must be divisible by number of heads"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.linear_q = nn.Linear(embed_dim, embed_dim)
        self.linear_k = nn.Linear(embed_dim, embed_dim)
        self.linear_v = nn.Linear(embed_dim, embed_dim)

        self.final_linear = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, query, key, value):
        batch_size = query.size(0)

        query = self.linear_q(query).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        key = self.linear_k(key).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        value = self.linear_v(value).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        attn_output = scaled_dot_product_attention(query, key, value)

        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.embed_dim)
        output = self.final_linear(attn_output)

        return output


if __name__ == "__main__":
    torch.manual_seed(42)

    model = MultiheadAttention(64, 4)
    x = torch.randn(2, 10, 64)

    output = model(x, x, x)

    assert output.shape == (2, 10, 64)

    output.sum().backward()
    assert all(p.grad is not None for p in model.parameters())

    x2 = x.clone()
    x2[:, -1] = torch.randn_like(x2[:, -1])

    output2 = model(x2, x2, x2)
    assert torch.allclose(output[:, :-1], output2[:, :-1], atol=1e-6)

    print("All tests passed!")