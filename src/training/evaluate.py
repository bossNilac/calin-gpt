import torch

from src.data.dataset import get_batch


def estimate_loss(model_, validation_data_, batch_size, context_length, eval_batches=20):
    was_training = model_.training
    model_.eval()
    device = next(model_.parameters()).device
    losses = []
    try:
        with torch.no_grad():
            for _ in range(eval_batches):
                x, y = get_batch(validation_data_, batch_size, context_length, device)
                _, loss = model_(x, y)
                losses.append(loss.item())
    finally:
        model_.train(was_training)
    return sum(losses) / len(losses)
