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

Full official WikiText-103 test split: Calin-GPT versus pretrained GPT-2.
**Bits per byte (BPB) is the comparison metric; lower is better.**

| Model | Parameters | Scored tokens | Loss (nats/token) | Token perplexity | BPB ↓ |
| --- | ---: | ---: | ---: | ---: | ---: |
| Calin-GPT | 17,542,912 | 400,984 | 3.283374 | 26.665590 | **1.465187** |
| GPT-2 | 124,439,808 | 287,644 | 3.343994 | 28.332053 | **1.070450** |

Calin-GPT / GPT-2 BPB ratio: **1.3688×**.
Token loss and perplexity use each model's own tokenizer and should not be ranked
across models. BPB normalizes their total negative log likelihood by the same bytes.

| Run details | Value |
| --- | --- |
| Dataset | Salesforce/wikitext / wikitext-103-raw-v1 / official test split |
| Dataset revision | `b08601e04326c79dfdd32d625aee71d232d685c3` |
| Official test rows evaluated | 4,358 |
| Evaluated UTF-8 bytes (both models) | 1,296,370 |
| Test text SHA-256 | `696cca6b65a171b0a358a4be6732cdfdf2dd6164a32e20fd70e3c13fc4dfae83` |
| Calin-GPT checkpoint | `checkpoints/final/checkpoint.pt` |
| Completed optimizer steps | 20000 |
| Checkpoint SHA-256 | `12b4b965227b17d12ee96e2ac512f0ea6c033f4840af01c9b5d69f4957733861` |
| Custom tokenizer SHA-256 | `4c9b0aebd663c3912258e2de18ea29ccfa1680b7c69346e61821234e665630f3` |
| Reference model | `openai-community/gpt2` |
| Reference model/tokenizer revision | `607a30d783dfa663caf39e06633721c8d4cfcd7e` |
| Context / stride (both models) | 512 / 256 tokens |
| Device / PyTorch / Transformers | cuda / 2.3.1+cu118 / 4.57.1 |

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
Command: `python -m src.benchmark --gpu`.

## File Hashes

| Artifact | SHA-256 |
| --- | --- |
| `checkpoints/final/checkpoint.pt` | `12b4b965227b17d12ee96e2ac512f0ea6c033f4840af01c9b5d69f4957733861` |
| `data/processed/tokenizer.json` | `4c9b0aebd663c3912258e2de18ea29ccfa1680b7c69346e61821234e665630f3` |
| `data/processed/metadata.json` | `5b37897b44202e1e293cd99e206e7ffdfcd723391ec6dad59a211518f4737bde` |
| `data/processed/documents.json` | `b78b61c35e2a0efe37138931703e0ef2e0d61dd4b567bc07c8ada366e96da131` |
| `data/processed/train.bin` | `f7e1ebc9ab949c58665c5f7a241a0f6b6e6e5cfa09c432d91e3a6c47c53e23f3` |
| `data/processed/val.bin` | `2a89bf5ac4359f909aa97f33be7a677ec0257755445e02f711aa7f451a6ad862` |

## Reflection

For me, the main achievement is going from small exercises to a trained model
that I can compare with GPT-2 on the same test text. GPT-2 still does better:
Calin-GPT's BPB is about 37% higher, even though its token perplexity looks lower.
The tokenizers are different, so BPB is the useful comparison here.

My model has about seven times fewer parameters, but that alone does not prove
it is more efficient. The models also had different training data and budgets.
I am happy to have built and trained this on a GTX 1080, while recognizing that
one test set does not tell me how well it reasons, follows instructions, or gets
facts right. This feels like a solid learning milestone, not a claim to beat GPT-2.
