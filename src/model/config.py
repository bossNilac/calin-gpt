from dataclasses import dataclass

@dataclass
class GPTConfig:
    vocab_size: int
    context_length: int = 64
    n_layers: int = 2
    n_heads: int = 4
    d_model: int = 128