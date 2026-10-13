import time
from pathlib import Path

import torch

from Week1.day7.main import x_y_batches, split_dataset
from src.model.config import GPTConfig
from src.model.gpt import GPT
from week2.day4.main import encode_bpe
from week2.day5.main import load_tokenizer

root = Path(__file__).resolve().parent.parent
text_path = root / "week2/day7/text.txt"
tokenizer_path = root / "week2/day7/tokenizer.json"

if __name__ == "__main__":
    print("Project root:", root)
    print("Text exists:", text_path.exists())
    print("Tokenizer exists:", tokenizer_path.exists())

    with open(text_path, encoding="utf-8") as f:
        data = f.read()

    print("Text characters:", len(data))

    vocab, merges = load_tokenizer(str(tokenizer_path))
    token_ids = encode_bpe(data, merges)

    print("Vocabulary size:", len(vocab))
    print("Total tokens:", len(token_ids))

    train_data, _, validation_data = split_dataset(token_ids, 0.8, 0, 0.2)

    B, T = 2, 16
    x, y = x_y_batches(train_data, B, T)

    print("Batch X shape:", x.shape)
    print("Batch Y shape:", y.shape)
    print("First input tokens:", x[0].tolist())
    print("First target tokens:", y[0].tolist())

    config = GPTConfig(vocab_size=len(vocab))
    model = GPT(config)

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)

    parameters = sum(p.numel() for p in model.parameters())

    print("\n--- MODEL ---")
    print("Parameters:", f"{parameters:,}")
    print("Layers:", config.n_layers)
    print("Embedding dimension:", config.d_model)
    print("Attention heads:", config.n_heads)

    print("\n--- TRAINING ---")
    start = time.time()

    for step in range(1001):
        logits, loss = model(x, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 100 == 0:
            elapsed = time.time() - start

            print(
                f"Step {step:4d}/1000 | "
                f"Loss: {loss.item():.6f} | "
                f"Elapsed: {elapsed:.1f}s"
            )

    print("\n--- RESULTS ---")
    print("Final loss:", loss.item())
    print("Total training time:", round(time.time() - start, 2), "seconds")
    print("Overfitting successful:", loss.item() < 0.1)