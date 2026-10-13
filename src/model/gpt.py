import torch
from torch import nn

from src.model.block import TransformerBlock
from src.model.config import GPTConfig
from src.model.layers import LayerNorm


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.token_emb=nn.Embedding(config.vocab_size, config.d_model)
        self.pos_emb=nn.Embedding(config.context_length, config.d_model)
        self.blocks = nn.ModuleList([
            TransformerBlock(config.d_model, config.n_heads)
            for _ in range(config.n_layers)
        ])
        self.final_norm = LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        position_ids = torch.arange(T, device=idx.device)

        token_embeddings = self.token_emb(idx)
        position_embeddings = self.pos_emb(position_ids)

        out =  token_embeddings + position_embeddings

        for block in self.blocks:
            out = block(out)

        out = self.final_norm(out)
        logits = self.lm_head(out)

        if targets is None:
            return logits

        b, t, v = logits.shape
        logits_flat = logits.reshape(b * t, v)
        targets_flat = targets.reshape(b * t)

        loss = torch.nn.functional.cross_entropy(logits_flat, targets_flat)

        return logits, loss

if __name__ == "__main__":
    config = GPTConfig(vocab_size=300)
    model = GPT(config)
    x = torch.randint(0, 300, (2, 16))
    y = torch.randint(0, 300, (2, 16))

    logits, loss = model(x, y)

    print("Logits shape:", logits.shape)
    print("Loss:", loss.item())
    print("Loss is finite:", torch.isfinite(loss).item())
    print("Number of blocks:", len(model.blocks))
    print("Trainable parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))

    loss.backward()

    print("Embedding gradients:", model.token_emb.weight.grad is not None)
    print("LM head gradients:", model.lm_head.weight.grad is not None)
    print("All gradients finite:", all(
        p.grad is not None and torch.isfinite(p.grad).all().item()
        for p in model.parameters() if p.requires_grad
    ))