# How the Project Grew

[README](../README.md) | [Architecture](architecture.md) | [Training](training.md)

## The First Three Weeks

I started with a daily plan.
[Week one](../Week1/week1_curriculum_exercises.md) covered character models,
token IDs, batches, embeddings, loss, and training.
[Week two](../week2/week2_curriculum.md) added my byte-level BPE tokenizer.
[Week three](../week3/week3_curriculum.md) focused on causal and multi-head attention.

I kept those plans and notes because they show how I got here. Results in an old
note belong to that exercise, not to the GPT I trained later.

## Burst 1: Full GPT Implementation

After week three, I wanted to spend more time connecting the pieces and coding.

- `3cd7799`: LayerNorm and an MLP with GELU.
- `c2b5c6a`: a Transformer block with normalization and residual connections.
- `ae6bd6e` and `ec1f31e`: token/position embeddings, stacked blocks, final
  normalization, the vocabulary head, configuration, and cross-entropy.

Read `src/model/layers.py`, `src/model/block.py`, then `src/model/gpt.py`.
The [architecture notes](architecture.md) explain the shapes.

One important check was trying to overfit a tiny batch. In `d61001d`, the training
script picks one batch and reuses it for 1,001 updates, then checks whether loss
is below 0.1. You can see it with `git show d61001d:src/train.py`. The check is in
Git, but I do not have a saved result here proving it passed. The current trainer
samples new windows instead.

## Burst 2: Training System

- `d61001d`: the first training script.
- `8326393`: my AdamW, validation loss, and warmup/cosine learning-rate schedule.
- `2e651c8`: saving and loading checkpoints.

The system now also has gradient clipping, console logging, seeds, JSON settings,
and resume. Checkpoints save the optimizer and random-number states as well as
the weights. The CPU smoke test compares a resumed run with an uninterrupted one.
Seeds help repeat a run, but do not promise identical results on different hardware.

To follow this part, read `src/optim/`, `src/training/`, and `src/train.py`.

## Burst 3: Training Readiness and Portability

I developed and prepared data on a laptop without a useful training GPU. The next
step was making the code and files easy to move to my gaming PC.

- `594fec8`: downloads, BPE preparation, document splits, binary data, and metadata.
- `9e5d4e0`: device handling, gradient accumulation, file checks, and portable resume.
- `91fad5f`: generation, diagnostics, and pipeline smoke checks.

Later changes added the Gutenberg marker fix, generation arguments, the showcase,
dependency pins, and updated training settings. The earlier plan used 20 million
tokens and context 256; the run used the newer 40-million-token target and context 512.

## The Completed Run

I trained on my GTX 1080. The last saved checkpoint confirms all **20,000 updates**
and **17,542,912 parameters**. My final training and validation losses were around
2.9; exact losses are not stored in the checkpoint.
See [the training record](../results/benchmark.md) for the verified settings,
dataset sizes, learning rate, and file hashes.

Finishing training does not tell me how good the model is. No separate benchmark
results are recorded; that evaluation would be optional follow-up work. I have not
documented BF16/compile checks or added public tokenizer/checkpoint downloads.
Understanding and being able to repeat the work matter more to me than making
the model bigger.

## Following Along

Build a small piece, check the shapes, and write down what happened.
Use `git log --oneline` and `git show <commit>:<path>` to look at earlier versions.

Some old exercises load files from the working directory, and the week-two
benchmark expects a corpus that is not included. The curricula explain those
details. Current `src/` data/config/checkpoint paths use the repo root instead.
The [training guide](training.md#machine-handoff) explains the machine setup.
