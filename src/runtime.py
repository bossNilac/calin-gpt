import argparse
import json
import random
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent


def settings_from_cli():
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--cpu', action='store_true')
    mode.add_argument('--gpu', action='store_true')
    parser.add_argument('--prompt', type=str, default=None)
    parser.add_argument('--temperature', type=float, default=None)
    parser.add_argument('--max-new-tokens', type=int, default=None)
    args = parser.parse_args()
    name = 'train_final' if args.gpu else 'debug'
    settings = json.loads((ROOT / 'configs' / f'{name}.json').read_text())
    settings['runtime']['device'] = 'cuda' if args.gpu else 'cpu'
    if args.prompt is not None:
        settings['generation']['prompt'] = args.prompt
    if args.temperature is not None:
        settings['generation']['temperature'] = args.temperature
    if args.max_new_tokens is not None:
        settings['generation']['max_new_tokens'] = args.max_new_tokens

    if not args.gpu:
        settings['runtime'].update(mixed_precision='none', compile=False)
    return settings


def select_device(settings):
    device = torch.device(settings['runtime']['device'])
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('GPU mode requires CUDA. Install a compatible PyTorch build and NVIDIA driver.')
    if device.type == 'cpu':
        torch.set_num_threads(settings['runtime'].get('threads', 1))
    return device


def seed_all(seed):
    random.seed(seed)
    torch.manual_seed(seed)


def fingerprint(path):
    import hashlib
    digest = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()
