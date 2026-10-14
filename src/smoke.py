import copy
import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import torch

from src.data.dataset import get_batch, load_data
from src.data.prepare import prepare
from src.generate import generate
from src.optim.scheduler import get_lr
from src.runtime import ROOT, select_device, settings_from_cli
from src.tokeniser.bpe import decode_bpe, encode_bpe, read_tokenizer
from src.train import train


def smoke(device='cpu'):
    settings = json.loads((ROOT / 'configs/debug.json').read_text())
    settings['runtime']['device'] = device
    select_device(settings)
    with tempfile.TemporaryDirectory(prefix='calin-smoke-') as temporary:
        output = Path(temporary)
        settings['data']['processed_dir'] = str(output / 'data')
        settings['data']['tokenizer_path'] = str(output / 'data/tokenizer.json')
        preparation = copy.deepcopy(settings)
        preparation['runtime']['device'] = 'cpu'
        with patch('src.data.acquire.fetch', side_effect=AssertionError('CPU smoke attempted network access')):
            prepare(preparation)
        vocab, merges, _ = read_tokenizer(output / 'data/tokenizer.json')
        samples = ['', 'Hello, world!\nA clear explanation.', 'café, 你好, 🙂\t  spaces']
        encoded = [encode_bpe(text, merges) for text in samples]
        for text, ids in zip(samples, encoded):
            assert decode_bpe(ids, vocab) == text
        del vocab, merges
        vocab, merges, _ = read_tokenizer(output / 'data/tokenizer.json')
        assert encoded == [encode_bpe(text, merges) for text in samples]
        train_data, val_data, _ = load_data(settings)
        x, y = get_batch(train_data, 2, 24, device)
        assert x.shape == y.shape == (2, 24)
        assert torch.equal(x[:, 1:], y[:, :-1])
        assert x.min() >= 0 and y.max() < len(vocab)
        assert len(val_data) > 24
        assert get_lr(0, 0.001, 0.0001, 2, 12) == 0
        assert get_lr(2, 0.001, 0.0001, 2, 12) == 0.001
        assert get_lr(12, 0.001, 0.0001, 2, 12) == 0.0001
        settings['checkpoint_dir'] = str(output / 'continuous')
        continuous, continuous_opt, steps = train(settings)
        settings['checkpoint_dir'] = str(output / 'interrupted')
        partial, _, partial_steps = train(settings, stop_after=6)
        assert partial_steps == 6
        assert any(not torch.equal(a, b) for a, b in zip(partial.parameters(), continuous.parameters()))
        resumed, resumed_opt, resumed_steps = train(settings)
        assert resumed_steps == steps == 12
        for a, b in zip(continuous.parameters(), resumed.parameters()):
            assert torch.equal(a, b), 'Interrupted model differs from uninterrupted model'
            assert torch.isfinite(b).all()
        for name in ('m', 'v'):
            assert all(torch.equal(a, b) for a, b in zip(getattr(continuous_opt, name), getattr(resumed_opt, name)))
        assert resumed_opt.steps == continuous_opt.steps
        assert all(step == 12 for step in resumed_opt.steps)
        text = generate(settings)
        assert text.startswith(settings['generation']['prompt'])
        print('Generation:', repr(text))
    print(f'PASS: {device} pipeline, BPE reload, batches, scheduler, accumulation, validation, '
          'checkpoints, exact resume, and generation')


def gpu_smoke(settings):
    select_device(settings)
    settings['training'].update(max_steps=2, warmup_steps=0, eval_interval=1,
                                save_interval=1, eval_batches=1, resume=True)
    settings['generation']['max_new_tokens'] = 12
    settings['runtime'].update(mixed_precision='none', compile=False)
    with tempfile.TemporaryDirectory(prefix='calin-gpu-smoke-') as temporary:
        settings['checkpoint_dir'] = temporary
        train(settings, stop_after=1)
        _, _, completed = train(settings)
        assert completed == 2
        print('Generation:', repr(generate(settings)))
    print('PASS: final model GPU smoke, real binary data, checkpoint reload, resume, generation')


if __name__ == '__main__':
    settings = settings_from_cli()
    if settings['runtime']['device'] == 'cuda':
        gpu_smoke(settings)
    else:
        smoke()

