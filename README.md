# Calin-GPT

This is my attempt at building a small GPT from scratch and understanding how the pieces work.

For the first three weeks, I followed a fairly structured plan: learn the basics, write small implementations, and test them. That covered tokenization, embeddings, attention, and the transformer. The exercises and notes from those weeks are still here.

In week four, I changed the pace. I felt like I was spending too much time on the slow, day-by-day approach, and I wanted to actually code more. So I did an accelerated sprint to connect the parts, finish the training pipeline, and get the project ready to move from my CPU laptop to my gaming PC.

The code in `src/` is the version I use now. It still imports some of my earlier implementations instead of rewriting everything. I use PyTorch for tensors and autograd, but the BPE tokenizer, GPT layers, AdamW optimizer, and learning-rate schedule are my own implementations.

## Where it is now

The full pipeline works on a tiny local English dataset: tokenizer training, binary preprocessing, model training, validation, saving a checkpoint, resuming, and generating text. The tests also compare an interrupted CPU run with an uninterrupted one; the final weights and optimizer state match exactly.

The real dataset pipeline is written, and small samples from all three sources have been checked. I haven't prepared the full 20 million tokens or trained the final model yet. CUDA, GPU memory use, BF16, and compilation still need testing on the gaming PC. The tiny CPU model is just a correctness check, so its generated text isn't useful English.

## What's in the repo

```text
Week1/                    first exercises and notes
week2/                    byte-level BPE tokenizer
week3/                    attention and transformer exercises
src/model/                GPT, transformer blocks, LayerNorm, MLP
src/optim/                custom AdamW and learning-rate scheduler
src/data/                 downloading, preprocessing, binary dataset loading
src/tokeniser/            integration of my week-two BPE code
src/training/             validation and checkpoints
src/train.py              training loop
src/generate.py           text generation
src/diagnostics.py        environment and parameter counts
src/smoke.py               complete pipeline checks
src/tests/                regression tests
configs/                  debug, small-run, and final settings
data/fixtures/english.txt  small local dataset for CPU testing
```

The original study plans are in [week one](Week1/week1_curriculum_exercises.md), [week two](week2/week2_curriculum.md), and [week three](week3/week3_curriculum.md). They show how I started; the later work doesn't follow the same daily format.

## Setup

Use Python 3.10+ in a virtual environment and install a PyTorch build that works with your hardware. Then install the remaining requirements:

```bash
python -m pip install -r requirements.txt
python -m src.diagnostics --cpu
```

If PyTorch already works, there's no reason to replace it. Local CPU testing needs only PyTorch and NumPy; the `datasets` package is used for real data acquisition.

For a GTX 1080, check that the PyTorch/CUDA build still supports Pascal (compute capability 6.1). A recent CUDA wheel may not include it.

## Try it on CPU

```bash
python -m src.smoke --cpu
python -m unittest discover -s src/tests -v
```

These run offline. The smoke test uses the 6.3 KB local fixture, a 288-token vocabulary, one 32-dimensional transformer layer, context length 24, and 12 optimizer updates. It checks tokenizer serialization, batches, forward/backward, the optimizer, validation, checkpoints, exact resume, and generation in a temporary directory.

You can also run the stages separately:

```bash
python -m src.data.prepare --cpu
python -m src.train --cpu
python -m src.generate --cpu
```

`train --cpu` prepares the local data if it doesn't exist yet. Omitting the flag defaults to CPU/debug mode. Mixed precision and compilation are disabled in this mode. The workflow flags are just `--cpu` and `--gpu`; other settings live in JSON.

## Prepare the real dataset

```bash
python -m src.data.prepare --gpu
```

This is the command that downloads real data. Here, `--gpu` selects the real workflow; preprocessing itself runs on CPU and doesn't require CUDA. It never starts training.

The target is about **20 million BPE tokens**, including validation and document separators:

| Source | Target share | Target tokens |
|---|---:|---:|
| [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) | 45% | 9 million |
| [WikiText-103 raw](https://huggingface.co/datasets/Salesforce/wikitext) | 35% | 7 million |
| [Project Gutenberg](https://www.gutenberg.org/) | 20% | 4 million |

FineWeb-Edu and WikiText are streamed from pinned revisions. FineWeb uses a bounded prefix of `sample-10BT`, not the entire subset. Gutenberg uses an explicit book list in JSON and a [listed mirror](https://www.gutenberg.org/MIRRORS.ALL), with a two-second wait between uncached downloads.

The pipeline caches raw documents in `data/cache/`, removes obvious formatting problems and Gutenberg wrappers, deduplicates by normalized document hash, and splits documents before tokenization. WikiText rows are grouped into articles, and Gutenberg books stay together when splitting. This catches exact and whitespace/case duplicates, not similar passages.

My BPE tokenizer trains once on up to 300,000 training-only characters, targeting 4,096 IDs including an end-of-document token. Its artifact stores the vocabulary, ordered merges, special tokens, and version. Existing legacy token IDs aren't changed when loading old artifacts. New preparation needs the configured vocabulary and an EOD token; use a new artifact path when changing vocabulary.

The output is:

```text
data/processed/train.bin
data/processed/val.bin
data/processed/tokenizer.json
data/processed/metadata.json
data/processed/documents.json
```

Training reads the binary streams through NumPy memory mapping and samples random windows. The files use little-endian uint16, or uint32 for a larger vocabulary. Metadata records token counts and checksums; the document manifest records source IDs/URLs, revisions or book checksums, and split assignments.

The source proportions are measured in tokens. Final documents can be truncated to fit the budgets, and an exhausted source reports its shortfall without substituting another source. Download errors stop preparation. Streaming can read extra bytes for Parquet row groups, and the original BPE training code can take a while. I haven't checked the full collection's yield or every book URL yet.

Keep the cache if you want to reproduce preparation, especially since Gutenberg files can change. Training doesn't need the raw text or another tokenizer training run.

## Move to the gaming PC

Clone or pull the repo, set up Python and a compatible PyTorch/CUDA build, and copy the entire `data/processed/` directory. Copy `checkpoints/final/` too if you're continuing a run.

```bash
python -m src.diagnostics --gpu
python -m src.smoke --gpu
```

GPU mode fails clearly if CUDA isn't available. The GPU smoke test uses the real binary data and final model for two optimizer updates, reloads a checkpoint after the first update, generates briefly, and reports peak allocated CUDA memory. It uses temporary checkpoints and doesn't prepare data.

Once that works, start training:

```bash
python -m src.train --gpu
```

The provisional final model has **17,444,608 parameters**: vocabulary 4,096, context 256, dimension 384, eight layers, six heads, and MLP ratio four. The output head is not tied to the embeddings.

| Component | Parameters |
|---|---:|
| Token embeddings | 1,572,864 |
| Position embeddings | 98,304 |
| Attention | 4,730,880 |
| MLP | 9,452,544 |
| LayerNorm | 13,056 |
| LM head | 1,576,960 |

CPU commands select `configs/debug.json`; GPU commands select `configs/train_final.json`. `configs/train_small.json` is a shorter experiment template: copy the settings you want into the final config and choose a separate checkpoint directory. The old `src/configs/debug.json` is left over from the earlier training script.

Adjust micro-batch size and gradient accumulation in JSON before starting. The default effective batch is 2 × 16 = 32 sequences. The loss is divided across micro-batches, and clipping, scheduling, and AdamW updates happen once per accumulation cycle.

Full precision is the default. FP16 isn't supported with the current custom optimizer's interface. BF16 is optional on supported CUDA GPUs, which excludes the GTX 1080. `runtime.compile` is also optional and defaults to false. I still need to test these options on actual hardware.

## Resume and generate

Run the same training command again to resume from the last saved optimizer update:

```bash
python -m src.train --gpu
python -m src.generate --gpu
```

`training.resume` defaults to true. For a fresh experiment, choose a new `checkpoint_dir` or set it to false. Checkpoints save the model, custom optimizer, completed updates, model/training settings, RNG states, and dataset/tokenizer identities. Loading maps tensors to the selected device. Production resume checks the artifact identities; the low-level loader still accepts the old checkpoint structure.

Keep the schedule and batch settings unchanged when resuming, including `max_steps`. Checkpoints are saved at configured intervals and at the end, so an interruption can lose work since the last save. Moving from CPU to GPU is supported, but identical numerical results across devices aren't expected.

The prompt, generation length, and temperature are in the JSON `generation` section. Generation loads the matching model/tokenizer, uses the most recent context window, and stops at EOD. Temperature zero uses greedy selection. There is no KV cache or beam search.

## Data licenses

- FineWeb-Edu uses ODC-BY. The underlying web pages still have their own rights; the dataset license doesn't give unrestricted permission to republish them.
- WikiText's dataset card specifies CC BY-SA 4.0. Attribution and applicable share-alike requirements matter when redistributing adapted data.
- Gutenberg's public-domain checks mainly follow US law. Check the rules where you use the books, and follow its [license and trademark conditions](https://www.gutenberg.org/policy/license.html) when redistributing Gutenberg files. Original cached books keep their notices.

Caches, processed data, and checkpoints are git-ignored. Tokenizing text or removing wrappers doesn't remove source obligations. The downloader uses collection files rather than crawling website pages; see Gutenberg's [robot policy](https://www.gutenberg.org/policy/robot_access.html) and [mirror guidance](https://www.gutenberg.org/help/mirroring.html).

## References I used

Karpathy's [Neural Networks: Zero to Hero](https://karpathy.ai/zero-to-hero.html), [nanoGPT](https://github.com/karpathy/nanoGPT), and [build-nanogpt](https://github.com/karpathy/build-nanogpt) helped me understand the overall approach. I also used the [Transformer paper](https://arxiv.org/abs/1706.03762), the [GPT-2 paper](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf), and [PyTorch documentation](https://pytorch.org/docs/stable/).

This is still a student project. The next step is to test it on the GPU and actually train it, then see what it can write.
