# Week 2 — Day 5: Decoding and Persistence

## Goal

Complete the tokenizer by supporting:

```text
token IDs
→ bytes
→ original text
```

and saving/loading the trained tokenizer.

## Decoding

The vocabulary maps:

```text
token ID -> byte sequence
```

Example:

```text
97  -> b"a"
256 -> b"ab"
257 -> b"abab"
```

To decode:

```text
token IDs
→ look up each byte sequence
→ concatenate bytes
→ UTF-8 decode
→ text
```

## Main Invariant

The tokenizer must satisfy:

```text
decode_bpe(encode_bpe(text)) == text
```

for arbitrary valid UTF-8 text.

## Vocabulary vs Merge Rules

Vocabulary:

```text
token ID -> represented bytes
```

Used primarily for decoding.

Merge rules:

```text
(left token, right token) -> new token
```

Used for encoding.

Both must be persisted.

## Saving

JSON cannot directly represent:

- Python `bytes`
- tuple dictionary keys
- integer dictionary keys exactly as Python uses them

Therefore these structures are converted into JSON-friendly values before saving and reconstructed when loading.

## Persistence Invariant

After:

```text
save
→ load
```

the tokenizer should produce exactly the same encodings and decodings as before.

## Required Tests

Test round trips with:

- ASCII
- punctuation
- whitespace
- accented text
- non-English scripts
- emojis
- empty strings
- unseen text

## Key Takeaway

After Day 5, the tokenizer is no longer only a BPE experiment. It is a reusable component that can be trained once, stored, loaded, encoded with, and decoded with.