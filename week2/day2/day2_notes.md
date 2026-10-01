# Week 2 — Day 2: BPE Basics

## Core idea

BPE repeatedly merges the most frequent adjacent token pair.

```text
current tokens
→ count adjacent pairs
→ select most frequent pair
→ assign new token ID
→ replace all non-overlapping occurrences
→ recount pairs
→ repeat
```

## Important distinction

Initially:

```text
token ID == byte value
0..255
```

After merges:

```text
256, 257, ...
```

represent sequences of bytes.

Example:

```text
(97, 98) -> 256
```

means token `256` was created from tokens representing `a` and `b`.

## Why frequencies must be recomputed

A merge changes the token sequence and creates new adjacent pairs.

```text
a b a b a b
→ merge (a,b) as 256
→ 256 256 256
```

Now `(256, 256)` exists even though it did not exist before.

## Overlapping pairs

Merges are non-overlapping and processed left-to-right.

```text
1 1 1
merge (1,1) -> 256
```

becomes:

```text
256 1
```

not:

```text
256 256
```

because the middle `1` cannot be used twice.

## Tokenizer training vs model training

Tokenizer training:
- counts frequencies
- creates merge rules
- uses normal Python data structures
- no gradients or PyTorch required

Model training:
- learns neural-network weights
- uses loss, gradients, and optimization

## Vocabulary size

Larger vocabulary:
- usually reduces token sequence length
- increases embedding/output vocabulary size
- can overfit/memorize a small training corpus

A tokenizer should therefore be evaluated on unseen text, not only the corpus it was trained on.

## Main Day 2 components

- `split_pairs(tokens)`
- `count_pairs(pairs)`
- `select_best_pair(tokens)`
- `merge_pair(tokens, pair, new_token)`
- repeated BPE merge loop

## Main invariant

After every merge:

```text
recompute pair frequencies from the new token sequence
```