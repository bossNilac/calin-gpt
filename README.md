# Calin-GPT

This is my attempt at building a small GPT from scratch and understanding how it
works. I started with daily exercises, then put the pieces together under `src/`.
The notes and exercises are still here so other people can follow the same path.

I used PyTorch for tensors, autograd, embeddings, and linear layers. The BPE
tokenizer, attention, LayerNorm, Transformer blocks, AdamW, and learning-rate
schedule are implemented in this repo.

I developed and prepared data on a laptop, then trained the model on my gaming
PC's GTX 1080. The goal was to learn and build something reproducible, not to
match the size or abilities of GPT-2.

## Where It Is Now

The training run finished **20,000 optimizer steps**. The last saved checkpoint
confirms that count and matches the current model and training config.

- **17,542,912 parameters**
- **4,096-token vocabulary**
- **512-token context**
- **38 million training tokens**, plus **1,983,311 validation tokens**
- Final training and validation loss around **2.9**, based on my recorded values
- Final saved learning rate around **0.00001**

The checkpoint does not store losses, so the loss values above are approximate.
The exact saved learning rate, dataset counts, and file hashes are in
[the training record](results/benchmark.md).

Training is finished. No separate benchmark results are recorded.
The generation showcase lets me inspect samples; it is not a benchmark score.
BF16 and `torch.compile` are optional code paths, but I have not documented tests
showing they work. I also still need to publish the tokenizer and checkpoint so
someone else can try the model without training it.

## Follow the Learning Journey

1. [Week one](Week1/week1_curriculum_exercises.md): character prediction, batches,
   embeddings, loss, training, and generation.
2. [Week two](week2/week2_curriculum.md): UTF-8 bytes and a custom BPE tokenizer.
3. [Week three](week3/week3_curriculum.md): causal and multi-head attention.
4. [Development history](docs/development.md): putting together the full GPT,
   adding a training system, and preparing for GPU training.
5. [Architecture](docs/architecture.md) and [training guide](docs/training.md):
   how the current code works and how to run it.

Try building each stage before reading my version. Keep notes about tensor
shapes, bugs, and what you learned. The weekly plans are the original exercises,
not a checklist of results. Older notes describe the model I had at that point.

## What's in the Repo

```text
Week1/                    character-model exercises and notes
week2/                    BPE tokenizer exercises
week3/                    attention exercises and checks
configs/                  debug/final configs and a small-run template
data/fixtures/english.txt  small local dataset for CPU checks
docs/                     architecture, training, development history
results/benchmark.md      training record and benchmark status
src/
  configs/debug.json      old config; current commands do not load it
  data/                   downloads, preparation, binary data loading
  model/                  GPT, blocks, LayerNorm, MLP, model config
  optim/                  AdamW and learning-rate schedule
  tokeniser/              connects the week-two tokenizer to the pipeline
  training/               validation and checkpoints
  tests/                  regression tests and AdamW comparison
  runtime.py              settings, devices, seeds, file hashes
  train.py                training loop
  generate.py             text generation from a checkpoint
  model_showcase.py       ten prompt/temperature tests
  diagnostics.py          environment info and parameter counts
  smoke.py                small CPU/GPU pipeline checks
requirements.txt          dependencies
```

Keep the weekly folders: `src/` still imports code from them. Folder case matters
on Linux. Datasets, checkpoints, environments, and raw showcase output stay out of
Git. `results/benchmark.md` is the exception because it is a written report.

## Try It on CPU

Run commands from the repo root in a virtual environment. Python 3.11 is a good
starting point for the pinned data dependencies. Install a PyTorch build that
supports your hardware, then:

```bash
python -m pip install -r requirements.txt
python -m src.diagnostics --cpu
python -m unittest discover -s src/tests -v
```

The tests use local data and temporary folders. They include a tiny training run
and a check that resuming gives the same CPU weights as uninterrupted training.
To run that pipeline check on its own, use `python -m src.smoke --cpu`.

You can also run the debug stages separately:

```bash
python -m src.data.prepare --cpu
python -m src.train --cpu
python -m src.generate --cpu --prompt "Once upon a time" --max-new-tokens 40
```

`--cpu` loads `configs/debug.json`; `--gpu` loads `configs/train_final.json`.
No flag defaults to CPU. There is no `--config` option.
See the [training guide](docs/training.md) for real data, GPU setup, and resume.

## Trying the Trained Model

Generation only needs the repo, settings, matching tokenizer, and checkpoint.
It does not need the training dataset. The
[inference instructions](docs/training.md#inference-handoff) explain the paths
for CPU and GPU use. The trained files exist locally, but public download links
have not been added yet.

## Compare with GPT-2

The benchmark evaluates my trained checkpoint and pretrained GPT-2 on the entire
WikiText-103 raw test split. It uses the same 512-token window and 256-token stride
for both models and reports a comparison table in [results/benchmark.md](results/benchmark.md).
Lower **bits per byte (BPB)** is better. Token perplexity is also shown, but the
models have different tokenizers, so those perplexities aren't directly comparable.

On the gaming PC, with the final checkpoint and tokenizer in their usual paths:

```bash
python -m pip install -r requirements-benchmark.txt
python -m src.benchmark --gpu
```

The optional dependency is only for loading GPT-2; the training setup stays the
same. The first run downloads GPT-2 weights (roughly half a gigabyte) and the test
split, then caches them. Both models are evaluated one at a time to reduce GPU
memory use. The full test takes longer than the old 100,000-token check; progress
is printed as it runs. No training happens.

The output records model revisions, checkpoint step, hashes, byte/token counts,
and the exact procedure. Both tokenizers must reconstruct the same test bytes.
A context-only start marker lets both models score all text bytes. These settings
make a comparison between these two runs, not a claim to reproduce published
word-level WikiText scores. The full trained-model comparison still needs running
on the gaming PC; adding the script doesn't establish a score.

## Data and References

I trained on FineWeb-Edu, WikiText-103 raw, and Project Gutenberg books.
See [the training record](results/benchmark.md) for the actual mixture and the
WikiText split check. Check source terms before sharing data:

- [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu)
- [WikiText](https://huggingface.co/datasets/Salesforce/wikitext)
- [Project Gutenberg license](https://www.gutenberg.org/policy/license.html)

References that helped me:
[Zero to Hero](https://karpathy.ai/zero-to-hero.html),
[nanoGPT](https://github.com/karpathy/nanoGPT),
[build-nanogpt](https://github.com/karpathy/build-nanogpt),
[the Transformer paper](https://arxiv.org/abs/1706.03762),
[the GPT-2 paper](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf),
and [PyTorch docs](https://pytorch.org/docs/stable/).
