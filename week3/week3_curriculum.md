# Week 3 — Causal Self-Attention and Multi-Head Attention

## Overview

This week builds the attention mechanism that will become the core of a decoder-only GPT. The exercises progress from simple weighted aggregation to a trainable, vectorized multi-head causal self-attention module in PyTorch.

**Prerequisites:** Python, basic PyTorch tensor operations, matrix multiplication, embeddings, and the tokenization work from Weeks 1–2.

**By the end of the week, you should be able to:**

- Explain why a token needs information from other tokens in its context.
- Derive and implement scaled dot-product attention.
- Explain and apply causal masking for next-token prediction.
- Build a trainable single-head attention module.
- Implement vectorized multi-head attention without using `nn.MultiheadAttention`.
- Test tensor shapes, gradient flow, and causal independence.

## Tensor notation

| Symbol | Meaning |
| --- | --- |
| `B` | Batch size |
| `T` | Number of tokens in a sequence |
| `C` | Embedding dimension |
| `H` | Number of attention heads |
| `D` | Dimension per head, usually `C // H` |

For a batch of token embeddings, the input tensor has shape `[B, T, C]`.

---

## Day 1 — Weighted aggregation

### Concept

An ordinary embedding represents a token independently of its surrounding context. Attention introduces **contextual representations**: each token's output can be a weighted combination of the representations of other tokens.

Given a sequence matrix `X` of shape `[T, C]` and a weight matrix `A` of shape `[T, T]`:

\[
Z = AX
\]

The `i`-th row of `A` determines how strongly token `i` uses information from every token in the sequence. The resulting `Z` has shape `[T, C]`.

### Exercises

1. Implement a function that multiplies a matrix by its transpose: `X @ X.T`.
2. Given a matrix `[T, C]` and a weight vector `[T]`, compute their weighted aggregation using matrix multiplication.
3. Build a small `[T, T]` weight matrix and calculate `Z = A @ X`.
4. Explain why elementwise multiplication is not a replacement for the matrix product in this exercise.

**Key idea:** A row of attention weights describes **where a token gathers information from**.

## Day 2 — Queries, keys, and values

### Concept

Attention does not use fixed weights. It computes relevance scores from the input itself using three learned projections:

\[
Q=XW_Q, \qquad K=XW_K, \qquad V=XW_V
\]

- **Query (Q):** What information is this token looking for?
- **Key (K):** What information can this token be matched on?
- **Value (V):** What information does this token contribute if attended to?

For `X` of shape `[T, C]` and projection matrices of shape `[C, D]`, all three results have shape `[T, D]`.

### Exercises

1. Implement `calculate_qkv(x, wq, wk, wv)` using matrix multiplication.
2. Check the output shapes of Q, K, and V.
3. Compute `Q @ K.T` and explain why its shape is `[T, T]`.
4. Explain why the final weighted aggregation uses **V**, not K.

**Key idea:** Q and K determine **how much attention** each token receives; V supplies **the information** being aggregated.

## Day 3 — Scaled dot-product attention

### Concept

The basic attention formula is:

\[
\operatorname{Attention}(Q,K,V)
= \operatorname{softmax}\left(\frac{QK^\top}{\sqrt{D}}\right)V
\]

1. `Q @ K.T` produces pairwise compatibility scores.
2. Dividing by `sqrt(D)` controls score magnitude as the head dimension grows.
3. Softmax over the last dimension turns each row into a distribution of nonnegative weights summing to one.
4. Multiplying by V produces contextualized token representations.

Without scaling, large dot products can saturate softmax and make optimization more difficult.

### Exercises

1. Implement `scaled_dot_product_attention(q, k, v)` from tensor operations.
2. Use `k.transpose(-2, -1)` so the implementation also supports batched inputs.
3. Apply `softmax(..., dim=-1)`.
4. Verify that the result has shape `[T, D]` (or `[B, T, D]` for batched inputs).
5. Optionally return the attention weights as well as the output to inspect row sums.

## Day 4 — Causal masking

### Concept

GPT predicts the next token from earlier tokens. A token at position `i` must **not** attend to positions `j > i`, because that would leak future information into the prediction.

Define a causal mask:

\[
M_{ij}=\begin{cases}
0 & j\le i\\
-\infty & j>i
\end{cases}
\]

Then:

\[
Z=\operatorname{softmax}\left(\frac{QK^\top}{\sqrt D}+M\right)V
\]

A practical implementation constructs a lower-triangular mask with `torch.tril`, sets disallowed scores to `-inf` with `masked_fill`, and applies softmax. Those positions then receive probability zero.

### Exercises

1. Add a causal mask to the Day 3 attention function.
2. Check that every attention weight above the main diagonal is zero.
3. Check that each row of attention weights sums to one.
4. Explain why masking the **scores before softmax** prevents future-token leakage.

**Key idea:** All token positions can be processed in parallel during training, even though each position can attend only to itself and earlier positions.

## Day 5 — A trainable single attention head

### Concept

Convert the previous functions into a reusable `torch.nn.Module` with trainable Q, K, and V projections.

For an input `[B, T, C]`, three `nn.Linear(C, D, bias=False)` layers produce Q, K, and V of shape `[B, T, D]`. The causal attention computation returns `[B, T, D]`.

An important distinction:

- `self.Q` is a **layer** holding trainable parameters.
- `self.Q(x)` applies that layer to the input and returns a **tensor**.

### Exercises

1. Create a `CausalSelfAttentionHead(nn.Module)` class.
2. Register three independent linear projections in `__init__`.
3. In `forward(x)`, compute `q = self.Q(x)`, `k = self.K(x)`, and `v = self.V(x)`.
4. Apply scaled causal attention and return its output.
5. Test the output shape for input `[4, 10, 16]` and head dimension `8`.
6. Backpropagate a scalar loss and verify gradients exist for all three projections.
7. Change the last input token and verify earlier output positions do not change.

**Key idea:** `nn.Linear` registers trainable weights automatically, allowing standard PyTorch optimizers to update the attention projections.

## Day 6 — Vectorized multi-head attention

### Concept

A single head learns one set of attention relationships. Multi-head attention learns several sets in parallel, then combines their outputs.

For `H` heads and `D = C // H`:

\[
\operatorname{MultiHead}(X)
= \operatorname{Concat}(Z_1,\ldots,Z_H)W_O
\]

Instead of creating separate Python modules for each head, the implementation from this week uses **three full-width linear projections**, each mapping `C → C`. Their outputs are reshaped into `H` heads. This is a vectorized implementation of the same idea.

### Tensor shapes

For `B=2`, `T=10`, `C=64`, `H=4`, and `D=16`:

| Stage | Shape |
| --- | --- |
| Input | `[2, 10, 64]` |
| Each full-width Q/K/V projection | `[2, 10, 64]` |
| Reshape | `[2, 10, 4, 16]` |
| Transpose to separate heads | `[2, 4, 10, 16]` |
| Attention scores and weights | `[2, 4, 10, 10]` |
| Per-head attention outputs | `[2, 4, 10, 16]` |
| Transpose and concatenate | `[2, 10, 64]` |
| Final output projection | `[2, 10, 64]` |

The attention score multiplication works on the last two dimensions and broadcasts across batch and head dimensions.

### Exercises

1. Implement `MultiheadAttention(nn.Module)` with `embed_dim` and `num_heads`.
2. Assert that `embed_dim % num_heads == 0`.
3. Define Q, K, V projections and a final output projection.
4. Project and reshape each of Q, K, and V to `[B, H, T, D]`.
5. Reuse the causal scaled dot-product attention function.
6. Transpose the head outputs back, make the tensor contiguous, and reshape to `[B, T, C]`.
7. Apply the final linear projection.

For GPT self-attention, Q, K, and V all originate from the **same input sequence**, but use different learned weights.

**Implementation note:** A dropout module was initialized during the exercise but not applied. It is optional at this stage and should either be removed or integrated deliberately in a later Transformer block.

## Day 7 — Validation and inspection

### Concept

Before integrating attention into a GPT, test its mathematical and autoregressive properties. Correct tensor shapes alone are not enough: gradients must reach the parameters, and future tokens must not influence earlier outputs.

### Required tests

1. **Shapes:** `[B, T, C] → [B, T, C]` for the multi-head module.
2. **Head configurations:** Verify `H=1`, `H=4`, and `H=8` with `C=64`.
3. **Gradient flow:** Backpropagate through the output and verify each parameter has a gradient.
4. **Causality:** Change the final input token; earlier output positions must remain unchanged.
5. **Attention weights:** Optionally expose the softmax matrix and verify each row sums to one and every position above the diagonal is zero.

### Optional visualization

Temporarily return both the attention output and the attention weights from your attention function. For batched multi-head attention, the weights have shape `[B, H, T, T]`.

```python
import matplotlib.pyplot as plt

plt.imshow(weights[0, 0].detach().cpu(), cmap="viridis")
plt.colorbar()
plt.show()
```

A lower-triangular pattern confirms causal masking. With randomly initialized parameters, the nonzero weights do **not** yet represent learned linguistic relationships.

---

## Week 3 recap

The implementation developed across the week follows this pipeline:

```text
Token embeddings X              [B, T, C]
       |
       +--> Linear Q --------+
       +--> Linear K --------+--> reshape into H heads
       +--> Linear V --------+    [B, H, T, D]
                                      |
                            Q @ K^T / sqrt(D)
                                      |
                                Causal mask
                                      |
                                  Softmax
                                      |
                                Weights @ V
                                      |
                            [B, H, T, D]
                                      |
                         Concatenate head outputs
                                      |
                                Linear output
                                      |
                               [B, T, C]
```

### Final understanding checklist

- [ ] I can explain the roles of Q, K, and V.
- [ ] I can derive the scaled dot-product attention equation.
- [ ] I understand why the causal mask is needed for GPT.
- [ ] I can track the shapes of every tensor in a multi-head forward pass.
- [ ] I know why each head has separate learned projections.
- [ ] I can verify gradients and causality with small tests.
- [ ] I can explain how vectorized heads correspond to independent attention heads.

### What's next?

Attention is one component of a GPT Transformer block. The next stage is to integrate it with **positional embeddings, layer normalization, residual connections, and a feed-forward network**, then stack Transformer blocks and train a decoder-only language model.

> This curriculum is intended both as a record of the implementation journey and as a reproducible set of exercises for anyone building a small GPT from scratch. Implement each stage yourself before consulting a reference implementation.
