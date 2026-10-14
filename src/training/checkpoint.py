import os
import random

import torch


def save_checkpoint(path, model, optimizer, step, model_config, train_config):
    os.makedirs(path, exist_ok=True)

    output = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'step': step,
        'model_config': model_config,
        'train_config': train_config,
        'python_rng': random.getstate(),
        'torch_rng': torch.get_rng_state(),
    }

    torch.save(output, os.path.join(path, 'checkpoint.pt'))


def load_checkpoint(path, model, optimizer):
    checkpoint = torch.load(
        os.path.join(path, "checkpoint.pt"),
        map_location="cpu",
        weights_only=True
    )

    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    random.setstate(checkpoint["python_rng"])
    torch.set_rng_state(checkpoint["torch_rng"])

    return checkpoint["step"]