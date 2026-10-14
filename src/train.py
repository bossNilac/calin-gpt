import time
from dataclasses import asdict
from pathlib import Path

import torch

from Week1.day7.main import x_y_batches, split_dataset
from src.model.config import GPTConfig
from src.model.gpt import GPT
from src.optim.adamw import AdamW
from src.optim.scheduler import get_lr
from src.training.checkpoint import load_checkpoint
from src.training.evaluate import estimate_loss
from training.checkpoint import save_checkpoint
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

    vocab, merges = load_tokenizer(str(tokenizer_path))
    token_ids = encode_bpe(data, merges)
    train_data, _, validation_data = split_dataset(token_ids, 0.8, 0, 0.2)

    train_config = {
        "batch_size": 2,
        "context_length": 16,
        "max_steps": 200,
        "max_lr": 0.001,
        "min_lr": 0.0001,
        "warmup_steps": 20,
        "grad_clip": 1.0,
        "eval_interval": 25,
        "save_interval": 50,
        "eval_batches": 10,
        "weight_decay": 0.01,
    }

    B = train_config["batch_size"]
    T = train_config["context_length"]
    max_steps = train_config["max_steps"]
    max_lr = train_config["max_lr"]
    min_lr = train_config["min_lr"]
    warmup_steps = train_config["warmup_steps"]
    grad_clip = train_config["grad_clip"]
    eval_interval = train_config["eval_interval"]
    save_interval = train_config["save_interval"]
    eval_batches = train_config["eval_batches"]

    checkpoint_path = root / "src" / "checkpoints"
    checkpoint_file = checkpoint_path / "checkpoint.pt"

    config = GPTConfig(vocab_size=len(vocab))
    model = GPT(config)

    optimizer = AdamW(
        model.parameters(),
        lr=max_lr,
        weight_decay=train_config["weight_decay"]
    )

    start_step = 0

    if checkpoint_file.exists():
        start_step = load_checkpoint(str(checkpoint_path), model, optimizer)
        print(f"Resumed from step {start_step}")
    else:
        print("Starting new training run")

    parameters = sum(p.numel() for p in model.parameters())

    print("\n--- DATA ---")
    print("Text characters:", len(data))
    print("Vocabulary size:", len(vocab))
    print("Total tokens:", len(token_ids))

    print("\n--- MODEL ---")
    print("Parameters:", f"{parameters:,}")
    print("Layers:", config.n_layers)
    print("Embedding dimension:", config.d_model)
    print("Attention heads:", config.n_heads)

    print("\n--- TRAINING ---")
    start_time = time.time()
    tokens_processed = start_step * B * T

    for step in range(start_step, max_steps):
        model.train()
        x, y = x_y_batches(train_data, B, T)

        lr = get_lr(step, max_lr, min_lr, warmup_steps, max_steps)
        optimizer.lr = lr

        logits, loss = model(x, y)

        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite loss at step {step + 1}")

        optimizer.zero_grad()
        loss.backward()

        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=grad_clip,
            error_if_nonfinite=True
        )

        optimizer.step()
        tokens_processed += B * T

        completed_steps = step + 1



        if completed_steps % eval_interval == 0:
            val_loss = estimate_loss(
                model, validation_data, B, T, eval_batches
            )

            elapsed = time.time() - start_time
            session_tokens = (completed_steps - start_step) * B * T
            throughput = session_tokens / elapsed if elapsed > 0 else 0

            print(
                f"Step {completed_steps}/{max_steps} | "
                f"Train: {loss.item():.4f} | "
                f"Val: {val_loss:.4f} | "
                f"LR: {lr:.6f} | "
                f"Grad: {grad_norm.item():.2f} | "
                f"Tokens: {tokens_processed:,} | "
                f"Tok/s: {throughput:.0f}"
            )

        if completed_steps % save_interval == 0:
            save_checkpoint(
                str(checkpoint_path),
                model,
                optimizer,
                completed_steps,
                asdict(config),
                train_config
            )
            print(f"Checkpoint saved at step {completed_steps}")

            # if completed_steps == 100:
            #     print("Simulating interruption at step 100")
            #     break

    print("\n--- RESULTS ---")
    print("Completed steps:", completed_steps)
    print("Total tokens processed:", tokens_processed)
    print("Session training time:", round(time.time() - start_time, 2), "seconds")
