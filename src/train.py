import time
from contextlib import nullcontext
from dataclasses import asdict

import torch

from src.data.dataset import get_batch, load_data
from src.model.config import GPTConfig
from src.model.gpt import GPT, parameter_counts
from src.optim.adamw import AdamW
from src.optim.scheduler import get_lr
from src.runtime import ROOT, fingerprint, seed_all, select_device, settings_from_cli
from src.training.checkpoint import load_checkpoint, save_checkpoint
from src.training.evaluate import estimate_loss


def train(settings, stop_after=None):
    device = select_device(settings)
    seed_all(settings['data']['seed'])
    train_data, val_data, metadata = load_data(settings)
    config = GPTConfig(**settings['model'])
    training = settings['training']
    precision = settings['runtime']['mixed_precision']
    if precision not in ('none', 'bf16'):
        raise ValueError('Use none or bf16; FP16 scaling is not supported by the custom AdamW')
    if precision == 'bf16' and (device.type != 'cuda' or not torch.cuda.is_bf16_supported()):
        raise RuntimeError('BF16 requires a supported CUDA GPU; use mixed_precision: none')
    model = GPT(config).to(device)
    optimizer = AdamW(model.parameters(), lr=training['max_lr'], weight_decay=training['weight_decay'])
    identity = {'tokenizer': metadata['tokenizer_sha256'],
                'dataset': fingerprint(ROOT / settings['data']['processed_dir'] / 'metadata.json')}
    checkpoint_dir = ROOT / settings['checkpoint_dir']
    start = 0
    if training['resume'] and (checkpoint_dir / 'checkpoint.pt').exists():
        start = load_checkpoint(checkpoint_dir, model, optimizer, device, identity,
                                asdict(config), training)
        print('Resumed completed optimizer steps:', start)
    print('Parameters:', parameter_counts(model))
    forward_model = torch.compile(model) if settings['runtime']['compile'] else model
    accumulation = training['accumulation_steps']
    if accumulation < 1:
        raise ValueError('accumulation_steps must be positive')
    batch_size, context = training['batch_size'], config.context_length
    maximum = training['max_steps']
    end = min(maximum, stop_after) if stop_after is not None else maximum
    completed = start
    begin = time.monotonic()
    for step in range(start, end):
        model.train()
        optimizer.zero_grad()
        lr = get_lr(step, training['max_lr'], training['min_lr'], training['warmup_steps'], maximum)
        optimizer.lr = lr
        total_loss = 0.0
        for _ in range(accumulation):
            x, y = get_batch(train_data, batch_size, context, device)
            amp = torch.autocast('cuda', dtype=torch.bfloat16) if precision == 'bf16' else nullcontext()
            with amp:
                _, loss = forward_model(x, y)
            if not torch.isfinite(loss):
                raise RuntimeError(f'Non-finite loss at optimizer step {step + 1}')
            total_loss += loss.item() / accumulation
            (loss / accumulation).backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), training['grad_clip'],
                                                  error_if_nonfinite=True)
        optimizer.step()
        completed = step + 1
        if completed % training['eval_interval'] == 0:
            val = estimate_loss(model, val_data, batch_size, context, training['eval_batches'])
            if not torch.isfinite(torch.tensor(val)):
                raise RuntimeError('Non-finite validation loss')
            print(f'Step {completed}/{maximum} | Train {total_loss:.4f} | Val {val:.4f} | '
                  f'LR {lr:.6f} | Grad {grad_norm.item():.2f}')
        if completed % training['save_interval'] == 0 or completed == end:
            save_checkpoint(checkpoint_dir, model, optimizer, completed, asdict(config), training, identity)
    print(f'Completed steps: {completed}; session: {time.monotonic() - begin:.2f}s')
    if device.type == 'cuda':
        print(f'Peak CUDA allocated memory: {torch.cuda.max_memory_allocated() / 2**20:.1f} MiB')
    return model, optimizer, completed


if __name__ == '__main__':
    settings = settings_from_cli()
    if settings['runtime']['device'] == 'cpu':
        from src.data.prepare import prepare
        if not (ROOT / settings['data']['processed_dir'] / 'metadata.json').exists():
            prepare(settings)
    train(settings)
