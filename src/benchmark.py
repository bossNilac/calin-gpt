import argparse
import hashlib
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

GPT2_ID = 'openai-community/gpt2'
GPT2_REVISION = '607a30d783dfa663caf39e06633721c8d4cfcd7e'


@torch.no_grad()
def evaluate(model, ids, vocab, context_length, bos_id, name='model'):
    model.eval()
    device = next(model.parameters()).device
    byte_count = sum(len(vocab[token]) for token in ids)
    count = len(ids)
    if not count or not byte_count:
        raise ValueError('Test text must not be empty')
    ids = [bos_id] + ids
    stride = max(1, context_length // 2)
    total_nll = 0.0
    scored = 0
    end = min(context_length, count)
    while scored < count:
        start = max(0, end - context_length)
        x = torch.tensor([ids[start:end]], dtype=torch.long, device=device)
        targets = torch.tensor(ids[scored + 1:end + 1], dtype=torch.long, device=device)
        output = model(x)
        logits = output if isinstance(output, torch.Tensor) else output.logits
        logits = logits[0, scored - start:]
        total_nll += F.cross_entropy(logits.float(), targets, reduction='sum').item()
        scored = end
        end = min(end + stride, count)
        if scored == count or scored % (stride * 100) == 0:
            print(f'{name}: {scored:,}/{count:,} tokens scored', flush=True)
    loss = total_nll / count
    metrics = {'loss': loss, 'perplexity': math.exp(loss),
               'bpb': total_nll / (math.log(2) * byte_count),
               'tokens': count, 'bytes': byte_count,
               'parameters': sum(p.numel() for p in model.parameters())}
    if not all(math.isfinite(metrics[key]) for key in ('loss', 'perplexity', 'bpb')):
        raise RuntimeError('Non-finite benchmark metrics')
    return metrics


def check_encoding(text, ids, vocab):
    if b''.join(vocab[token] for token in ids) != text.encode('utf-8'):
        raise ValueError('Tokenization did not preserve the exact test bytes')


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
    try:
        import transformers
        from transformers import GPT2LMHeadModel, GPT2Tokenizer
    except ImportError as error:
        raise RuntimeError('Install the reference-model dependency: '
                           'python -m pip install -r requirements-benchmark.txt') from error
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    torch.use_deterministic_algorithms(True)
    seed_all(settings['data']['seed'])
    device = select_device(settings)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    checkpoint_path = ROOT / settings['checkpoint_dir'] / 'checkpoint.pt'
    tokenizer_path = ROOT / settings['data']['tokenizer_path']
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    tokenizer_hash = fingerprint(tokenizer_path)
    if (checkpoint.get('identity') or {}).get('tokenizer') != tokenizer_hash:
        raise ValueError('Checkpoint/tokenizer identity mismatch')
    vocab, merges, eod = read_tokenizer(tokenizer_path)
    config = GPTConfig(**checkpoint['model_config'])
    if config.vocab_size != len(vocab) or eod is None:
        raise ValueError('Checkpoint vocabulary mismatch or missing EOD token')
    model = GPT(config)
    model.load_state_dict(checkpoint['model'])
    model = model.to(device)
    step = checkpoint['step']
    del checkpoint

    from datasets import load_dataset
    source = settings['data']['hf']['wikitext']
    cache = ROOT / settings['data']['cache_dir'] / 'hf'
    dataset = load_dataset('Salesforce/wikitext', name='wikitext-103-raw-v1',
                           split='test', revision=source['revision'], streaming=True,
                           cache_dir=str(cache))
    test_rows = [row['text'] for row in dataset]
    text = '\n\n'.join(test_rows)
    ids = encode_bpe(text, merges)
    check_encoding(text, ids, vocab)
    context = min(512, config.context_length)
    calin = evaluate(model, ids, vocab, context, eod, 'Calin-GPT')
    del model, ids, vocab, merges
    if device.type == 'cuda':
        torch.cuda.empty_cache()

    print('Loading GPT-2 reference (cached after the first download)...', flush=True)
    reference_options = dict(revision=GPT2_REVISION, cache_dir=str(cache))
    tokenizer = GPT2Tokenizer.from_pretrained(GPT2_ID, **reference_options)
    reference = GPT2LMHeadModel.from_pretrained(
        GPT2_ID, use_safetensors=True, attn_implementation='eager', **reference_options)
    reference.config.use_cache = False
    reference = reference.to(device)
    if reference.config.n_positions < context:
        raise ValueError('Reference model cannot support the comparison context')
    ids = tokenizer.encode(text, add_special_tokens=False, verbose=False)
    vocab = {index: bytes(tokenizer.byte_decoder[c] for c in token)
             for token, index in tokenizer.get_vocab().items()}
    check_encoding(text, ids, vocab)
    gpt2 = evaluate(reference, ids, vocab, context, tokenizer.eos_token_id, 'GPT-2')
    if calin['bytes'] != gpt2['bytes']:
        raise ValueError('Models were not evaluated on the same bytes')
    rows = []
    for name, result in [('Calin-GPT', calin), ('GPT-2', gpt2)]:
        rows.append(f"| {name} | {result['parameters']:,} | {result['tokens']:,} | "
                    f"{result['loss']:.6f} | {result['perplexity']:.6f} | **{result['bpb']:.6f}** |")
    section = f'''## Benchmark Results

Full official WikiText-103 test split: Calin-GPT versus pretrained GPT-2.
**Bits per byte (BPB) is the comparison metric; lower is better.**

| Model | Parameters | Scored tokens | Loss (nats/token) | Token perplexity | BPB ↓ |
| --- | ---: | ---: | ---: | ---: | ---: |
{chr(10).join(rows)}

Calin-GPT / GPT-2 BPB ratio: **{calin['bpb'] / gpt2['bpb']:.4f}×**.
Token loss and perplexity use each model's own tokenizer and should not be ranked
across models. BPB normalizes their total negative log likelihood by the same bytes.

| Run details | Value |
| --- | --- |
| Dataset | Salesforce/wikitext / wikitext-103-raw-v1 / official test split |
| Dataset revision | `{source['revision']}` |
| Official test rows evaluated | {len(test_rows):,} |
| Evaluated UTF-8 bytes (both models) | {calin['bytes']:,} |
| Test text SHA-256 | `{hashlib.sha256(text.encode('utf-8')).hexdigest()}` |
| Calin-GPT checkpoint | `{settings['checkpoint_dir']}/checkpoint.pt` |
| Completed optimizer steps | {step} |
| Checkpoint SHA-256 | `{fingerprint(checkpoint_path)}` |
| Custom tokenizer SHA-256 | `{tokenizer_hash}` |
| Reference model | `{GPT2_ID}` |
| Reference model/tokenizer revision | `{GPT2_REVISION}` |
| Context / stride (both models) | {context} / {max(1, context // 2)} tokens |
| Device / PyTorch / Transformers | {device} / {torch.__version__} / {transformers.__version__} |

Method: join every official test row in dataset order with two newlines, preserving
raw text and empty rows. Evaluate the entire resulting text, with no token cap,
cleaning, or retraining. Each model uses its original tokenizer; exact UTF-8
reconstruction is checked. Prepend that model's EOD/EOS as a context-only start
marker so every text token and byte is scored, including the first. No EOD/EOS
markers are inserted between test rows. Use overlapping input windows, scoring the first
window fully and only new targets thereafter, including the final partial window.
Each target is scored exactly once; positions restart at zero per window. Both
models use the same token context/stride policy, but tokenizers give different
byte spans. GPT-2 is deliberately limited to Calin-GPT's context for this comparison.

Loss = summed next-token negative log likelihood / scored tokens;
perplexity = exp(loss); BPB = summed negative log likelihood / (ln(2) × UTF-8 bytes).
Both models run in full precision with model.eval(), torch.no_grad(), a fixed seed,
deterministic PyTorch algorithms, TF32 disabled, and no compilation or KV cache.
This is a reproducible comparison on a standard test dataset, not a reproduction
of published word-level WikiText perplexity. Training/test overlap across the
models' pretraining sources has not been exhaustively checked.

Protocol reference: [strided sliding-window evaluation](https://huggingface.co/docs/transformers/perplexity).
Command: `python -m src.benchmark --{'gpu' if device.type == 'cuda' else 'cpu'}`.
'''
    save_results(ROOT / 'results/benchmark.md', section)
    print(section)
    return {'calin': calin, 'gpt2': gpt2}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--cpu', action='store_true')
    mode.add_argument('--gpu', action='store_true')
    args = parser.parse_args()
    settings = json.loads((ROOT / 'configs/train_final.json').read_text())
    settings['runtime']['device'] = 'cuda' if args.gpu else 'cpu'
    benchmark(settings)
