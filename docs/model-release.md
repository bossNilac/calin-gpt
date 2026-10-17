# Trying the Released Model

[README](../README.md) | [Training guide](training.md) |
[Benchmark and reflection](../results/benchmark.md)

The [20,000-step model release](https://github.com/bossNilac/calin-gpt/releases/tag/calin-gpt-20k)
contains the trained checkpoint, tokenizer, config, and checksums. You do not need
to prepare a dataset or train anything to generate text.

## Download

Use the code from the release tag `calin-gpt-20k`, or a compatible newer checkout.
Install a suitable PyTorch build and the repo's requirements in a virtual environment.
Python 3.11 is a good starting point. Keep the weekly code folders too, because
the model and tokenizer still import from them.

| File | Download | Where it goes in the repo |
| --- | --- | --- |
| Checkpoint, about 211 MB | [checkpoint.pt](https://github.com/bossNilac/calin-gpt/releases/download/calin-gpt-20k/checkpoint.pt) | `checkpoints/final/checkpoint.pt` |
| Tokenizer | [tokenizer.json](https://github.com/bossNilac/calin-gpt/releases/download/calin-gpt-20k/tokenizer.json) | `data/processed/tokenizer.json` |
| Settings | [train_final.json](https://github.com/bossNilac/calin-gpt/releases/download/calin-gpt-20k/train_final.json) | `configs/train_final.json` (already included in the tagged code) |
| Checksums | [SHA256SUMS](https://github.com/bossNilac/calin-gpt/releases/download/calin-gpt-20k/SHA256SUMS) | Use to check the downloads |

The checkpoint is the same file used in the benchmark, including its optimizer
state. It has 17,542,912 model parameters and 20,000 completed training updates.
The file hashes are also listed in [the benchmark report](../results/benchmark.md#file-hashes).

From the repo root, these PowerShell commands download the model files. They
replace files at those paths, so keep any other experiment in a separate folder:

```powershell
New-Item -ItemType Directory -Force checkpoints/final, data/processed | Out-Null
Invoke-WebRequest https://github.com/bossNilac/calin-gpt/releases/download/calin-gpt-20k/checkpoint.pt -OutFile checkpoints/final/checkpoint.pt
Invoke-WebRequest https://github.com/bossNilac/calin-gpt/releases/download/calin-gpt-20k/tokenizer.json -OutFile data/processed/tokenizer.json
```

To check downloads on Windows, use `Get-FileHash -Algorithm SHA256 <file>` and
compare with `SHA256SUMS`. On Linux, put the three downloaded files beside
`SHA256SUMS` and run `sha256sum -c SHA256SUMS`.

## Generate on CPU

The usual `--cpu` command selects the tiny debug files. To use this trained model
on CPU without changing your debug config, run this from the repo root:

```bash
python -c "import json; from src.generate import generate; s=json.load(open('configs/train_final.json')); s['runtime']['device']='cpu'; print(generate(s))"
```

It uses the prompt, length, and temperature in `configs/train_final.json`.
Change its `generation` section to try other prompts. This only runs inference;
it does not read training data or update weights.

## Generate on GPU

With a compatible NVIDIA driver and PyTorch/CUDA build:

```bash
python -m src.generate --gpu --prompt "Once upon a time" --temperature 0.8 --max-new-tokens 100
```

The tokenizer must be the one from this release. A different file will fail the
checkpoint's hash check. Generation reads the model shape from the checkpoint.

This is a small base language model, not a chat assistant. It can continue a
prompt, but the benchmark does not establish reliable facts, reasoning, or
instruction following. See [the results and reflection](../results/benchmark.md).
