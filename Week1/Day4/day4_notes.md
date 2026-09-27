# Day 4 Notes — Embeddings, Logits, Softmax, Cross-Entropy

## 1. Embedding table `(V, C)`

The embedding table has shape:

\[
(V, C)
\]

where:

- `V` = vocabulary size
- `C` = embedding dimension

Each row corresponds to one token ID.

If `V = 33` and `C = 8`, then:

```text
embedding_table.shape = (33, 8)
```

Token ID `7` selects:

```text
embedding_table[7]
```

which is a vector of length `8`.

The token ID itself has no learned meaning. The embedding vector is the learnable numerical representation of that token.

---

## 2. Embedding lookup: `(B, T) -> (B, T, C)`

Before embedding:

```text
X.shape = (B, T)
```

Each element of `X` is one integer token ID.

Looking up every token in the embedding table replaces each integer with a vector of length `C`.

Therefore:

\[
(B, T) \rightarrow (B, T, C)
\]

Example:

```text
X.shape              = (3, 10)
embedding_table.shape = (33, 8)
embedded_batch.shape  = (3, 10, 8)
```

Interpretation:

- 3 sequences
- 10 token positions per sequence
- 8 learned features per token

---

## 3. Output weight matrix `(C, V)`

For every token position, the model must produce one score for every possible next token.

The embedding vector has length `C`, but the output must have length `V`.

So we use a weight matrix:

\[
W \in \mathbb{R}^{C \times V}
\]

Example:

```text
W.shape = (8, 33)
```

Then:

\[
(B,T,C)(C,V) \rightarrow (B,T,V)
\]

---

## 4. Logits `(B, T, V)`

Logits are raw prediction scores.

After the matrix multiplication:

```text
logits.shape = (B, T, V)
```

Example:

```text
logits.shape = (3, 10, 33)
```

`logits[b, t]` is a vector containing 33 scores.

Each score corresponds to one possible next token.

A logit is not a probability. It can be negative, positive, or larger than 1.

---

## 5. Softmax

Softmax converts the `V` logits for one prediction into probabilities.

\[
p_i = \frac{e^{z_i}}{\sum_j e^{z_j}}
\]

where `z_i` is one logit.

After softmax:

- every probability is between `0` and `1`
- all probabilities for one prediction sum to `1`
- larger logits receive larger probabilities

Softmax does not change the tensor shape:

\[
(B,T,V) \rightarrow (B,T,V)
\]

Example:

```text
logits.shape = (3, 10, 33)
probs.shape  = (3, 10, 33)
```

and:

```text
probs[0, 0].sum() ~= 1
```

---

## 6. Cross-entropy

For every prediction position, there is one correct next-token ID stored in `Y`.

Example:

```text
logits[b, t] = 33 scores
Y[b, t]      = correct token ID
```

If the correct token receives probability `p_correct`, the loss for that prediction is:

\[
L = -\ln(p_{correct})
\]

High probability on the correct token gives low loss.

Low probability on the correct token gives high loss.

PyTorch cross-entropy should receive the raw logits, not manually computed softmax probabilities.

---

## 7. Why flatten `(B, T, V)` into `(B*T, V)`

Originally:

```text
logits.shape = (B, T, V)
Y.shape      = (B, T)
```

There are really `B*T` independent next-token prediction positions.

Cross-entropy can treat them as one list of predictions:

\[
(B,T,V) \rightarrow (B \cdot T,V)
\]

and:

\[
(B,T) \rightarrow (B \cdot T)
\]

Example:

```text
logits: (3, 10, 33) -> (30, 33)
Y:      (3, 10)     -> (30)
```

This does not recompute anything.

It only rearranges the same predictions so that each row of logits lines up with one correct target token.

---

## 8. Why random loss is approximately `ln(V)`

A randomly initialized model usually starts close to guessing uniformly.

If there are `V` possible tokens:

\[
p_{correct} \approx \frac{1}{V}
\]

Cross-entropy is:

\[
L = -\ln(p_{correct})
\]

so:

\[
L \approx -\ln(1/V) = \ln(V)
\]

For:

```text
V = 33
```

we expect roughly:

\[
\ln(33) \approx 3.50
\]

Your Day 4 loss of about `3.60` is therefore reasonable for an untrained model.

---

# Full Day 4 Shape Flow

```text
X
(B, T)
    |
    | embedding lookup using table (V, C)
    v
embedded batch
(B, T, C)
    |
    | multiply by weights (C, V)
    v
logits
(B, T, V)
    |
    | softmax, for interpretation
    v
probabilities
(B, T, V)

For loss:

logits (B, T, V) -> (B*T, V)
Y      (B, T)    -> (B*T)

cross-entropy -> one scalar loss
```

# What I Should Be Able to Explain

- Why the embedding table has shape `(V, C)`.
- Why embedding lookup changes `(B, T)` into `(B, T, C)`.
- What `C` represents.
- Why the output weight matrix has shape `(C, V)`.
- Why logits have shape `(B, T, V)`.
- What a logit is.
- Why logits are not probabilities.
- What softmax does.
- Why softmax probabilities sum to 1.
- What cross-entropy measures.
- Why high probability on the correct token means low loss.
- Why PyTorch cross-entropy takes raw logits.
- Why `(B, T, V)` can be flattened into `(B*T, V)`.
- Why `Y` becomes `(B*T)`.
- Why an untrained model has loss roughly equal to `ln(V)`.
