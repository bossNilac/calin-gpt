# Running the Project

[README](../README.md) | [Architecture](architecture.md) |
[Development history](development.md) | [Training record](../results/benchmark.md)

The 20,000-step run is complete. These instructions explain how to repeat the
workflow, start another experiment, or load the trained model.

## Setup and Configs

Run commands from the repo root with your virtual environment active.
Python 3.11 is a good starting point. Install a PyTorch build for your hardware,
then `python -m pip install -r requirements.txt`.

The requirements are `torch>=2.2`, `numpy==1.26.4`, `pyarrow==17.0.0`, and
`datasets==2.21.0`. CPU checks use PyTorch and NumPy; real downloads also use Datasets.
These requirements do not select a CUDA build for you.

[The shared CLI](../src/runtime.py) chooses these paths:

| Mode | Config | Data | Checkpoints |
| --- | --- | --- | --- |
| Default or `--cpu` | `configs/debug.json` | `data/debug/` | `checkpoints/debug/` |
| `--gpu` | `configs/train_final.json` | `data/processed/` | `checkpoints/final/` |

There is no `--config` option. `configs/train_small.json` is a template: copy
the settings you want into the selected config and use a separate checkpoint folder.
The old `src/configs/debug.json` is not used by current commands.

Generation accepts `--prompt`, `--temperature`, and `--max-new-tokens`.
Other settings are in JSON. CPU mode disables mixed precision and compilation.

## CPU Checks

```bash
python -m src.diagnostics --cpu
python -m unittest discover -s src/tests -v
```

The tests cover tokenizer saving/loading, cleaning, duplicate removal, cached data,
file checks, resume settings, and a small CPU training pipeline.
The data-source test uses local documents, not live downloads.
The AdamW test compares my optimizer with PyTorch's version.

You can run `python -m src.smoke --cpu` on its own. It uses the local fixture,
a 288-token vocabulary, context 24, one layer of size 32, and 12 optimizer steps.
It compares uninterrupted training with a resumed run, including weights and AdamW
state. This checks the pipeline, not language quality.

The separate CPU commands are in the README.
`src.train --cpu` prepares debug data automatically if its metadata file is missing.

## Dataset Preparation

```bash
python -m src.data.prepare --gpu
```

Here `--gpu` means the real-data workflow. Preparation itself runs on CPU, does
not require CUDA, and does not start training.

| Source | Share | Total token target |
| --- | ---: | ---: |
| FineWeb-Edu `sample-10BT` | 45% | 18,000,000 |
| WikiText `wikitext-103-raw-v1` | 35% | 14,000,000 |
| Gutenberg book list from the config | 20% | 8,000,000 |

Five percent of each target is reserved for validation. The completed dataset has
38,000,000 training tokens and 1,983,311 validation tokens; Gutenberg validation
fell short by 16,689 tokens. The code reports shortfalls without filling them from
another source. [The training record](../results/benchmark.md) has the exact counts.

Hugging Face downloads use pinned revisions and only the upstream `train` split.
FineWeb is streamed until enough data is collected, not downloaded in full.
Gutenberg uses a fixed book list and waits two seconds before uncached downloads.
Its wrapper parser handles line-wrapped markers. Download or wrapper errors stop preparation.

The code cleans whitespace, rejects unsuitable text, and removes documents with
the same whitespace-normalized, case-insensitive hash. It does not find near duplicates.
WikiText rows are grouped into articles; Gutenberg books stay together when
splitting. Documents are shuffled and split in small batches before tokenization.

If the tokenizer file is missing, BPE trains on training documents from the first
batch of each source. The combined cap is 1,000,000 characters, divided by source
share, but those batches may contain less. The target is 4,096 IDs including EOD.

An existing tokenizer is reused, even if you change the character cap. Use a new
tokenizer path to retrain it. A wrong vocabulary size or missing EOD causes an
error. Changing a source's settings requires a new cache directory.

Default output files:

```text
data/processed/train.bin
data/processed/val.bin
data/processed/tokenizer.json
data/processed/metadata.json
data/processed/documents.json
```

The tokenizer path can be changed separately. Binary files use little-endian
uint16, or uint32 for a larger vocabulary. Documents end with EOD and may be cut
short to fit the token budget. Encoded documents are held in memory before writing,
so streaming downloads do not make the whole process use constant memory.

Training reads the binaries through NumPy memory mapping after checking hashes,
lengths, vocabulary size, and token IDs. It does not need raw text.
Keep the cache to repeat preparation: cached documents are reused, but binary
outputs are rebuilt. BPE training scans tokens per merge and can be slow;
I have not recorded a full preparation time.

## Training Settings

[`configs/train_final.json`](../configs/train_final.json) matches the completed
checkpoint's model and training settings:

| Setting | Value |
| --- | ---: |
| Vocabulary / context | 4,096 / 512 |
| Layers / heads / embedding size / MLP ratio | 8 / 6 / 384 / 4 |
| Micro-batch / accumulation | 4 / 8 |
| Sequences / prediction positions per update | 32 / 16,384 |
| Total steps / warmup steps | 20,000 / 400 |
| Maximum / minimum learning rate | 0.0003 / 0.00001 |
| Gradient clip / weight decay | 1.0 / 0.1 |
| Evaluate every / batches | 200 steps / 20 |
| Save every | 500 steps, plus the end |

The model has **17,542,912 parameters**, checked against both the code and saved
weights. My final training and validation losses were around 2.9. The saved
learning rate is `1.0000001862625769e-05`, close to the minimum.

One optimizer step averages gradients over eight micro-batches, clips them, and
applies AdamW. The schedule warms up, then decays with cosine. Step zero uses LR zero.
Weight decay applies to all parameters, including biases and LayerNorm.
Validation samples windows without gradients and restores the model's earlier mode.

## CUDA and Portability

### Machine Handoff

I used a laptop for development/data preparation and a GTX 1080 for training.
For another run, the workflow is:

```text
prepare code, tokenizer, data, and configs on the laptop
    -> run CPU checks
    -> pull the same repo and copy data to the GPU PC
    -> set up its environment
    -> run a short GPU check
    -> adjust settings for a new experiment
    -> train and save a checkpoint
```

Recreate the virtual environment on the other machine instead of copying it.
Current data/config/checkpoint paths use the repo root, not a fixed machine path.
Old exercises use working-directory paths; the showcase does too.

Copy all of `data/processed/`, and `checkpoints/final/` if resuming.
Install a compatible PyTorch/CUDA build. For older GPUs like the GTX 1080,
make sure that build supports the GPU architecture.

```bash
python -m src.diagnostics --gpu
python -m src.smoke --gpu
python -m src.train --gpu
```

GPU smoke needs prepared real data. It runs two updates with a checkpoint reload
after the first, generates briefly, and uses temporary checkpoints.
It leaves the real checkpoint alone and forces full precision with compilation off.

BF16 and `torch.compile` hooks exist, but I have not recorded successful tests for
them. BF16 requires a supported CUDA GPU. FP16 is not supported by this trainer.
There are no recorded GPU speed, memory, or total training-time measurements here.

Choose micro-batch size and accumulation before a new run. Changing them after
saving a checkpoint will fail the resume checks.

## Resume and Generate

With `training.resume=true`, training loads `checkpoint.pt` if it exists.
It saves weights, AdamW state, completed steps, configs, file identities, and RNG
states, replacing the previous checkpoint through a temporary file.

Resume checks the model, dataset/tokenizer, and training settings. Only `resume`,
`eval_interval`, and `save_interval` can change. Keep the schedule, batch settings,
`max_steps`, and `eval_batches` the same. At the completed 20,000-step checkpoint,
running the same training command does not perform more updates.

For a fresh experiment, use a new checkpoint folder. Setting resume false in the
old folder starts over and overwrites its checkpoint at the next save.
Interrupting training loses steps since the last save. Moving tensors between
CPU and CUDA is supported, but results need not be identical across devices.

```bash
python -m src.generate --gpu --prompt "Once upon a time" --temperature 0.8 --max-new-tokens 200
python -m src.model_showcase
```

Generation reads the model shape from the checkpoint. It needs a nonempty prompt
and matching tokenizer, but not the training data. Temperature zero is greedy;
positive values sample. Each call resets the seeds.

The showcase always uses GPU generation, with ten prompt/temperature cases and
up to 120 new tokens each. It writes `results/model_showcase.txt` from the working
directory. It records errors and continues, so check the individual exit codes.
It does not record file hashes or produce a benchmark score.

## Inference Handoff

The trained checkpoint, tokenizer, config, and checksums are available in the
[20,000-step release](https://github.com/bossNilac/calin-gpt/releases/tag/calin-gpt-20k).
The [download and inference guide](model-release.md) gives direct links and commands.
Another person only needs the repo, compatible dependencies, config, tokenizer,
and checkpoint to generate text.

For the default GPU command, use:

```text
data/processed/tokenizer.json
checkpoints/final/checkpoint.pt
```

For the same model on CPU, set `data.tokenizer_path` and `checkpoint_dir` in
`configs/debug.json` to those paths, then run `python -m src.generate --cpu`.
Otherwise CPU mode selects the tiny debug model's files. The saved model shape
comes from the checkpoint, not the debug config.

Inference needs no raw cache, binary dataset, or dataset metadata.
The release includes the config and file hashes alongside the model files.
