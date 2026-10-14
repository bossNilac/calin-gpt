import os
import random

import torch


def save_checkpoint(path, model, optimizer, step, model_config, train_config, identity=None):
    os.makedirs(path, exist_ok=True)
    output = {
        'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
        'step': step, 'model_config': model_config, 'train_config': train_config,
        'python_rng': random.getstate(), 'torch_rng': torch.get_rng_state(),
        'identity': identity,
    }
    if next(model.parameters()).device.type == 'cuda':
        output['cuda_rng'] = torch.cuda.get_rng_state_all()
    temporary = os.path.join(path, 'checkpoint.tmp')
    torch.save(output, temporary)
    os.replace(temporary, os.path.join(path, 'checkpoint.pt'))


def load_checkpoint(path, model, optimizer, device='cpu', identity=None,
                    model_config=None, train_config=None):
    checkpoint = torch.load(os.path.join(path, 'checkpoint.pt'), map_location=device, weights_only=True)
    if identity is not None and checkpoint.get('identity') != identity:
        raise ValueError('Checkpoint dataset/tokenizer identity mismatch')
    if model_config is not None and checkpoint['model_config'] != model_config:
        raise ValueError('Checkpoint model configuration mismatch')
    if train_config is not None:
        ignored = {'resume', 'eval_interval', 'save_interval'}
        previous = {k: v for k, v in checkpoint['train_config'].items() if k not in ignored}
        current = {k: v for k, v in train_config.items() if k not in ignored}
        if previous != current:
            raise ValueError('Resume training settings differ; preserve the schedule and batch settings')
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    random.setstate(checkpoint['python_rng'])
    torch.set_rng_state(checkpoint['torch_rng'].cpu())
    if torch.device(device).type == 'cuda' and 'cuda_rng' in checkpoint:
        torch.cuda.set_rng_state_all([state.cpu() for state in checkpoint['cuda_rng']])
    return checkpoint['step']
