# Week 2 — Byte-Level BPE Tokenization

## Goal

Build a byte-level Byte Pair Encoding tokenizer from scratch and integrate it into the language-model pipeline from Week 1.

By the end of the week, the full pipeline should work:

```text
raw text
→ UTF-8 bytes
→ byte token IDs
→ learned BPE merges
→ compressed token IDs
→ language model
→ generated token IDs
→ BPE decoding
→ text
```

The focus is understanding how tokenization works internally rather than using an existing tokenizer library.

---

## Learning Objectives

By the end of Week 2, you should be able to explain:

- characters vs Unicode code points vs bytes vs token IDs
- how UTF-8 represents text
- why 256 byte values can represent arbitrary UTF-8 input
- why word-level vocabularies have out-of-vocabulary problems
- how Byte Pair Encoding works
- how adjacent pair frequencies are counted
- why pair frequencies must be recalculated after merges
- how learned tokens can represent multiple bytes
- why merge order matters
- the difference between vocabulary entries and merge rules
- how unseen text is encoded
- how BPE decoding reconstructs the original byte sequence
- vocabulary-size and sequence-length tradeoffs
- how tokenizer choice affects the downstream language model

---

# Day 1 — Text Representation

Study:

- Unicode
- UTF-8
- characters
- code points
- bytes
- byte-level token IDs

Implement:

```text
text → UTF-8 bytes
bytes → initial token IDs
token IDs → bytes
bytes → text
```

Initial vocabulary:

```text
0 ... 255
```

where each token represents exactly one byte.

Main invariant:

```python
decode(encode(text)) == text
```

Test ASCII, whitespace, accented characters, non-English text, and emojis.

---

# Day 2 — BPE Fundamentals

Study the BPE merge process manually.

Given a token sequence:

```text
a b a b a b
```

count adjacent pairs:

```text
(a,b)
(b,a)
(a,b)
(b,a)
(a,b)
```

Select the most frequent pair, assign it a new token ID, and replace all non-overlapping occurrences.

Example:

```text
(a,b) → 256

a b a b a b
→
256 256 256
```

Implement:

- adjacent pair extraction/counting
- most frequent pair selection
- non-overlapping pair merging

Understand why the token sequence must be rescanned after every merge.

---

# Day 3 — BPE Training

Build the complete BPE training loop.

```text
initial byte tokens
→ count pairs
→ select best pair
→ create new token
→ merge
→ recount
→ repeat
```

Maintain three separate structures:

### Current token sequence

```text
[97, 98, 97, 98]
→
[256, 256]
```

### Merge rules

```text
(97,98) → 256
(256,256) → 257
```

### Vocabulary

```text
97  → b"a"
98  → b"b"
256 → b"ab"
257 → b"abab"
```

For every new token:

```python
vocab[new_id] = vocab[left] + vocab[right]
```

Stop when the target vocabulary size is reached or no adjacent pairs remain.

Also improve the naive implementation by avoiding unnecessary sorting and intermediate pair lists.

---

# Day 4 — Encoding New Text

Use the learned merge rules to tokenize unseen text.

Encoding must not:

- count frequencies to learn new rules
- create new vocabulary entries
- modify the tokenizer

Pipeline:

```text
new text
→ UTF-8 byte IDs
→ apply learned merges in learned order
→ BPE token IDs
```

Simple reference implementation:

```text
for every learned merge:
    apply it to the current token sequence
```

Approximate complexity:

```text
O(MN)
```

where:

- `M` = number of merge rules
- `N` = input sequence length

Main invariant:

```python
encode_bpe(training_text, merges) == final_training_tokens
```

Understand why merge priority matters.

---

# Day 5 — Decoding and Persistence

Implement BPE decoding.

Given:

```text
token ID → byte sequence
```

decode by concatenating the vocabulary entries:

```text
token IDs
→ byte sequences
→ combined bytes
→ UTF-8 text
```

Main invariant:

```python
decode_bpe(encode_bpe(text, merges), vocab) == text
```

Also implement:

- tokenizer saving
- tokenizer loading
- vocabulary serialization
- merge-rule serialization

Verify that saving and loading does not change tokenizer behavior.

---

# Day 6 — Evaluation

Train and evaluate the tokenizer on a larger corpus.

Split the corpus before tokenizer training:

```text
training corpus
validation corpus
```

Learn merges only from the training corpus.

Evaluate on held-out text.

Measure:

- UTF-8 byte count
- BPE token count
- compression ratio
- tokens per byte
- tokens per character
- tokens per word
- tokenizer training time

Compare several vocabulary sizes such as:

```text
256
300
500
1000
```

Inspect learned vocabulary entries and test unusual inputs:

- punctuation
- capitalization
- numbers
- whitespace
- accented characters
- non-English scripts
- emojis
- uncommon Unicode

Understand that larger vocabularies usually reduce sequence length but increase model vocabulary size.

---

# Day 7 — Language Model Integration

Replace the Week 1 tokenizer with the BPE tokenizer.

```text
raw text
→ encode_bpe()
→ token IDs
→ torch tensor
→ train/validation split
→ batches
→ model
```

Set:

```python
vocab_size = len(vocab)
```

The language model still performs next-token prediction exactly as before.

For:

```text
B = batch size
T = context length
C = embedding dimension
V = BPE vocabulary size
```

the model produces:

```text
logits.shape == (B, T, V)
```

Generation becomes:

```text
starting text
→ BPE token IDs
→ autoregressive model generation
→ generated token IDs
→ BPE decoder
→ text
```

The goal is integration, not language-model quality.

---

# Core Invariants

The completed tokenizer should satisfy:

```python
decode_bpe(encode_bpe(text, merges), vocab) == text
```

for arbitrary valid text.

Every token must satisfy:

```python
0 <= token_id < len(vocab)
```

Every learned vocabulary entry must satisfy:

```python
vocab[new_token] == vocab[left] + vocab[right]
```

Re-encoding the tokenizer training corpus should reproduce its final training tokenization.

---

# Key Data Structures

## Vocabulary

```text
token ID → bytes
```

Example:

```text
97  → b"a"
256 → b"ab"
```

Used to determine what each token represents and to decode token IDs.

## Merge Rules

```text
(left token, right token) → new token
```

Example:

```text
(97,98) → 256
```

Used to reproduce the learned tokenization process when encoding new input.

The two structures serve different purposes and should remain separate.

---

# Vocabulary Size Tradeoff

A small vocabulary such as:

```text
V = 256
```

is universal but produces long token sequences.

A larger BPE vocabulary usually gives:

```text
larger V
→ shorter token sequences
```

but also creates larger model embedding and output layers.

Therefore the objective is not to make the vocabulary as large as possible.

Tokenizer quality should be evaluated on held-out data rather than only on the corpus used to learn the merges.

---

# Week 2 Final Deliverable

A reusable byte-level BPE tokenizer capable of:

- converting arbitrary UTF-8 text to bytes
- using all 256 byte values as the base vocabulary
- learning BPE merge rules from a training corpus
- encoding unseen text
- decoding token IDs exactly
- saving and loading its vocabulary and merge rules
- measuring compression behavior
- integrating with the Week 1 language model

No external tokenizer implementation or pretrained vocabulary is required.

---

# Completion Criteria

Week 2 is complete when you can explain and implement:

```text
text
→ UTF-8
→ bytes
→ byte token IDs
→ BPE merges
→ compressed token IDs
→ model
→ generated IDs
→ bytes
→ text
```

from first principles.