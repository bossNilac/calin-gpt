import argparse
import json
import math
import os
import re

import torch
import torch.nn.functional as F

from src.model.config import GPTConfig
from src.model.gpt import GPT
from src.runtime import ROOT, fingerprint, seed_all, select_device
from src.tokeniser.bpe import encode_bpe, read_tokenizer


@torch.no_grad()
def evaluate(model, ids, vocab, context_length, limit=100000):
    model.eval()
    device = next(model.parameters()).device
    count = min(limit, len(ids) - 1)
    if count < 1:
        raise ValueError('Test text must contain at least two tokens')
    stride = max(1, context_length // 2)
    total_nll = 0.0
    scored = 0
    end = min(context_length, count)
    while scored < count:
        start = max(0, end - context_length)
        x = torch.tensor([ids[start:end]], dtype=torch.long, device=device)
        targets = torch.tensor(ids[scored + 1:end + 1], dtype=torch.long, device=device)
        logits = model(x)[0, scored - start:]
        total_nll += F.cross_entropy(logits.float(), targets, reduction='sum').item()
        scored = end
        end = min(end + stride, count)
    byte_count = sum(len(vocab[token]) for token in ids[1:count + 1])
    loss = total_nll / count
    metrics = {'loss': loss, 'perplexity': math.exp(loss),
               'bpb': total_nll / (math.log(2) * byte_count),
               'tokens': count, 'bytes': byte_count}
    if not all(math.isfinite(metrics[key]) for key in ('loss', 'perplexity', 'bpb')):
        raise RuntimeError('Non-finite benchmark metrics')
    return metrics


def save_results(path, section):
    if path.exists():
        previous = path.read_text(encoding='utf-8')
        if re.search(r'^## Benchmark Results$', previous, re.M):
            report = re.sub(r'^## Benchmark Results\n.*?(?=^## |\Z)',
                            lambda _: section + '\n', previous, flags=re.M | re.S)
        else:
            report = previous.rstrip() + '\n\n' + section
    else:
        report = '# Calin-GPT benchmark\n\n' + section
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report, encoding='utf-8')


def benchmark(settings):
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    torch.use_deterministic_algorithms(True)
    seed_all(settings['data']['seed'])
    device = select_device(settings)
    checkpoint_path = ROOT / settings['checkpoint_dir'] / 'checkpoint.pt'
    tokenizer_path = ROOT / settings['data']['tokenizer_path']
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    tokenizer_hash = fingerprint(tokenizer_path)
    if (checkpoint.get('identity') or {}).get('tokenizer') != tokenizer_hash:
        raise ValueError('Checkpoint/tokenizer identity mismatch')
    vocab, merges, _ = read_tokenizer(tokenizer_path)
    config = GPTConfig(**checkpoint['model_config'])
    if config.vocab_size != len(vocab):
        raise ValueError('Checkpoint vocabulary mismatch')
    model = GPT(config)
    model.load_state_dict(checkpoint['model'])
    model = model.to(device)
    del checkpoint['model']
    checkpoint.pop('optimizer', None)

    from datasets import load_dataset
    source = settings['data']['hf']['wikitext']
    dataset = load_dataset('Salesforce/wikitext', name='wikitext-103-raw-v1',
                           split='test', revision=source['revision'], streaming=True,
                           cache_dir=str(ROOT / settings['data']['cache_dir'] / 'hf'))
    text = '\n\n'.join(row['text'] for row in dataset)
    ids = encode_bpe(text, merges)
    metrics = evaluate(model, ids, vocab, config.context_length)
    section = f'''## Benchmark Results

| Item | Value |
| --- | --- |
| Dataset | Salesforce/wikitext / wikitext-103-raw-v1 / official test split |
| Dataset revision | `{source['revision']}` |
| Checkpoint | `{settings['checkpoint_dir']}/checkpoint.pt` |
| Completed optimizer steps | {checkpoint['step']} |
| Parameters | {sum(p.numel() for p in model.parameters()):,} |
| Checkpoint SHA-256 | `{fingerprint(checkpoint_path)}` |
| Tokenizer SHA-256 | `{tokenizer_hash}` |
| Scored tokens | {metrics['tokens']:,} |
| Scored UTF-8 bytes | {metrics['bytes']:,} |
| Mean cross-entropy (nats/token) | {metrics['loss']:.8f} |
| Token perplexity | {metrics['perplexity']:.8f} |
| Bits per byte | {metrics['bpb']:.8f} |
| Device / PyTorch | {device} / {torch.__version__} |

Method: join all official test rows in dataset order with two newlines, preserving
raw text and empty rows. Encode once using the saved custom byte-level BPE, with
no added EOD/BOS tokens or cleaning. Score the first up to 100,000 next-token
targets; the first token is context only. Use a {config.context_length}-token input
window and stride {max(1, config.context_length // 2)}. Score the first window fully,
then only the new targets in each overlapping window, including the final partial
window. Each target is scored exactly once; positions restart at zero per window.
Context grows within each window and never exceeds the checkpoint's context size.
Sum next-token negative log likelihoods and divide by scored tokens for loss;
perplexity = exp(loss). BPB = total negative log likelihood / (ln(2) × scored
bytes), where bytes are the lengths of the scored BPE tokens, excluding the first
context token. Evaluation uses full precision, model.eval(), torch.no_grad(), a
fixed seed, and deterministic PyTorch algorithms. This is a capped test-prefix
result with custom tokenization, not a full-test word-level WikiText perplexity.

Command: `python -m src.benchmark --{'gpu' if device.type == 'cuda' else 'cpu'}`.
'''
    save_results(ROOT / 'results/benchmark.md', section)
    print(section)
    return metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--cpu', action='store_true')
    mode.add_argument('--gpu', action='store_true')
    args = parser.parse_args()
    settings = json.loads((ROOT / 'configs/train_final.json').read_text())
    settings['runtime']['device'] = 'cuda' if args.gpu else 'cpu'
    benchmark(settings)
