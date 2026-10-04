# Week 2 — Day 4: Encoding with Learned BPE Rules

## Goal

Use an already-trained BPE tokenizer to encode new text.

```text
new text
→ UTF-8 bytes
→ initial byte token IDs
→ learned BPE merges
→ final token IDs
```

## Training vs Encoding

Training:

```text
count pairs
→ choose most frequent pair
→ create new token
→ repeat
```

Encoding:

```text
start from byte tokens
→ apply existing merge rules
→ return tokens
```

Encoding must never create new vocabulary entries.

## Merge Order

Merge rules must be applied in the order they were learned.

Example:

```text
(97, 98) -> 256
(256, 99) -> 257
```

For:

```text
abc
```

the sequence becomes:

```text
[97, 98, 99]
→ [256, 99]
→ [257]
```

Later merges can therefore depend on tokens created by earlier merges.

## Simple Encoder

The reference implementation applies each learned merge rule to the token sequence in order.

Its approximate complexity is:

\[
O(MN)
\]

where:

- \(M\) = number of merge rules
- \(N\) = sequence length

This is not optimal but is simple and easy to verify.

## Important Invariant

For the training corpus:

```text
encode_bpe(training_text, merges)
==
final training token sequence
```

## Key Takeaway

`merges` determine **how tokens are formed during encoding**.

The tokenizer learns these rules once during training and then reuses them for unseen text.