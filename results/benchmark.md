# Training Results and Benchmark Notes

[README](../README.md) | [Training guide](../docs/training.md)

I finished the **20,000-step training run** on my GTX 1080.
The last saved checkpoint, `checkpoints/final/checkpoint.pt`, confirms all 20,000
optimizer updates. Its settings match `configs/train_final.json`.

The checkpoint, tokenizer, dataset metadata, document list, and binary files were
checked locally on 2026-10-16. They are not included in Git; their hashes are below.

## Training Results

| Item | Value | Where it comes from |
| --- | --- | --- |
| Completed steps | 20,000 of 20,000 | Checkpoint step and all AdamW step counters |
| GPU | GTX 1080 | My run details; the checkpoint does not store the GPU name |
| Parameters | 17,542,912 | Counted from saved model tensors and checked against the code |
| Vocabulary / context | 4,096 / 512 | Checkpoint model config; vocabulary also matches the tokenizer |
| Layers / heads / embedding size / MLP ratio | 8 / 6 / 384 / 4 | Checkpoint model config |
| Micro-batch / accumulation / effective batch | 4 / 8 / 32 sequences | Checkpoint training config |
| Training loss | About 2.9 | My reported final value; loss is not saved in the checkpoint |
| Validation loss | About 2.9 | My reported final value; loss is not saved in the checkpoint |
| Final learning rate | `1.0000001862625769e-05` | Saved optimizer state and scheduler calculation |
| Maximum / minimum LR; warmup | `0.0003` / `0.00001`; 400 steps | Checkpoint training config |
| Gradient clip / weight decay | 1.0 / 0.1 | Checkpoint training config |
| Validation interval / batches | Every 200 steps / 20 batches | Checkpoint training config |

The final update uses zero-based scheduler step 19,999.
`get_lr(19999, 0.0003, 0.00001, 400, 20000)` matches the saved LR.
I use that saved value as the final LR.

The training loss is the average over the last update's micro-batches. Validation
loss is the average over sampled windows from my validation data. These are
training metrics, not a separate benchmark. I need the original log for more exact
losses, so I have not calculated perplexity from the rounded values.

The current config uses full precision and disables compilation.
The checkpoint does not store runtime settings or environment versions.
I have not recorded BF16/compile tests, GPU speed, memory use, or total run time.

## Training Data

The checkpoint's tokenizer and dataset hashes match the local files.
The binary files and document list also match the hashes in `metadata.json`.

| Source | Training tokens | Validation tokens | Training share |
| --- | ---: | ---: | ---: |
| FineWeb-Edu `sample-10BT` | 17,100,000 | 900,000 | 45% |
| WikiText-103 raw | 13,300,000 | 700,000 | 35% |
| Gutenberg | 7,600,000 | 383,311 | 20% |
| Total | 38,000,000 | 1,983,311 | 100% |

The target was 40 million tokens. The actual total is **39,983,311**, including
end-of-document tokens. Gutenberg validation was 16,689 tokens short.
The training mixture is exactly 45/35/20. These are dataset sizes; training samples
windows repeatedly, so they are not counts of unique tokens seen across all updates.

Dataset revisions saved in metadata:

- FineWeb-Edu: `87f09149ef4734204d70ed1d046ddc9ca3f2b8f9`.
- WikiText: `b08601e04326c79dfdd32d625aee71d232d685c3`.

## WikiText Test Split

[The downloader](../src/data/acquire.py) requests
`load_dataset(..., name='wikitext-103-raw-v1', split='train', streaming=True)`.
A mocked call checked those arguments without downloading anything.
The code makes my own train/validation split from that upstream training data.
The saved document list has 2,662 WikiText records with the expected revision.

So the training pipeline does not request the WikiText-103 test split.
One limit to this check: cached documents do not save the upstream split name,
and I have not compared every cached document with the official test set.
I also have not checked for overlapping text from FineWeb or Gutenberg.

## Benchmark Results

No separate benchmark results are recorded. The validation loss around
2.9 is **not** a WikiText-103 test result. No test-set evaluation was run during
this documentation check.

The showcase is a set of generation samples to inspect, not a measured score.
It does not prove factual knowledge, instruction following, or coding ability.

If I choose to run a separate benchmark, I should record the checkpoint/tokenizer hashes,
dataset revision and split, tokenization, context/window setup, number of scored
tokens, how loss is averaged, and the command used.

## File Hashes

| Artifact | SHA-256 |
| --- | --- |
| `checkpoints/final/checkpoint.pt` | `12b4b965227b17d12ee96e2ac512f0ea6c033f4840af01c9b5d69f4957733861` |
| `data/processed/tokenizer.json` | `4c9b0aebd663c3912258e2de18ea29ccfa1680b7c69346e61821234e665630f3` |
| `data/processed/metadata.json` | `5b37897b44202e1e293cd99e206e7ffdfcd723391ec6dad59a211518f4737bde` |
| `data/processed/documents.json` | `b78b61c35e2a0efe37138931703e0ef2e0d61dd4b567bc07c8ada366e96da131` |
| `data/processed/train.bin` | `f7e1ebc9ab949c58665c5f7a241a0f6b6e6e5cfa09c432d91e3a6c47c53e23f3` |
| `data/processed/val.bin` | `2a89bf5ac4359f909aa97f33be7a677ec0257755445e02f711aa7f451a6ad862` |
