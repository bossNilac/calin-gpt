import torch
import math

from src.model.config import GPTConfig
from src.model.gpt  import GPT
from Week1.day7.main import x_y_batches


def estimate_loss(model_, validation_data_, batch_size, context_length, eval_batches=20):
    was_training = model_.training
    model_.eval()
    all_loss = torch.zeros(eval_batches)
    with torch.no_grad():
        for i in range(eval_batches):
            x_,y_ = x_y_batches(validation_data_, batch_size, context_length)
            logits_, loss_ = model_(x_, y_)
            all_loss[i] = loss_.item()

    model_.train(was_training)
    return all_loss.mean().item()

if __name__ == "__main__":
    config = GPTConfig(vocab_size=300)
    model = GPT(config)

    B, T = 2, 16

    validation_data = torch.randint(0, 300, (500,)).tolist()

    model.train()

    before = [p.detach().clone() for p in model.parameters()]
    val_loss = estimate_loss(model, validation_data, B, T)

    print("Validation loss:", val_loss)
    print("Finite:", math.isfinite(val_loss))
    print("Training mode restored:", model.training)
    print("Parameters unchanged:", all(
        torch.equal(a, b)
        for a, b in zip(before, model.parameters())
    ))