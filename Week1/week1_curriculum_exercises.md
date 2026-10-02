# Week 1 — Foundations of a Tiny Language Model

This week builds the smallest complete language-model pipeline from scratch.

The goal is **not** to build a transformer yet. The goal is to understand the full path from raw text to a trained model that can generate text.

By the end of the week, you should be able to explain and rebuild:

```text
raw text
→ tokenization
→ token IDs
→ train/validation split
→ X/Y batches
→ embeddings
→ logits
→ cross-entropy loss
→ backpropagation
→ optimizer updates
→ autoregressive generation
```

## Rules for the Week

Use:

- Python
- PyTorch tensors
- PyTorch autograd
- NumPy if useful

Avoid:

- transformers
- self-attention
- multi-head attention
- positional embeddings
- LayerNorm
- residual connections
- BPE
- AdamW
- KV caching

The point is to understand the basic training pipeline before adding transformer-specific machinery.

---

# Notation

Use these symbols consistently:

- `B` = batch size
- `T` = sequence/context length
- `C` = embedding/channel dimension
- `V` = vocabulary size

Important tensor shapes:

```text
X                  (B, T)
Y                  (B, T)
embedding table    (V, C)
embedded X         (B, T, C)
output weights     (C, V)
logits             (B, T, V)
flattened logits   (B*T, V)
flattened Y        (B*T)
loss               scalar
```

---

# Day 1 — Next-Token Prediction

## Goal

Understand what a language model is actually trying to predict.

## Concepts

Study:

- text as a sequence
- characters as simple tokens
- next-token prediction
- conditional probability
- self-supervised learning
- context
- probability distributions over possible next tokens

The core objective is:

`P(next token | previous token(s))`

For a simple character model:

```text
h → e
e → l
l → l
l → o
```

## Mathematics

Review conditional probability:

`P(A | B) = P(A and B) / P(B)`

Understand the sequence factorization idea:

```text
P(x1, x2, x3)
=
P(x1)
* P(x2 | x1)
* P(x3 | x1, x2)
```

You do not need advanced probability theory yet.

## Implementation Exercise

Create a small character-transition language model.

1. Load a `.txt` file.
2. Treat each character as a token.
3. Count adjacent character pairs.
4. Estimate `P(next_character | current_character)`.

Example:

```text
a → b : 5
a → c : 3
a → d : 2
```

should become:

```text
P(b | a) = 0.5
P(c | a) = 0.3
P(d | a) = 0.2
```

## Experiment

Generate characters using the learned transition probabilities.

Try:

- always choosing the most likely next character
- sampling according to the learned probabilities

Observe what kinds of repetition occur.

## Tests

For every starting character:

```text
sum of probabilities of all possible next characters ≈ 1
```

## Done When

You can explain:

- what next-token prediction means
- where the training targets come from
- why text can supervise itself
- why the model predicts a distribution instead of one fixed answer
- the difference between joint and conditional probability

---

# Day 2 — Vocabulary, Encoding, and Decoding

## Goal

Convert characters into integer token IDs and back.

## Concepts

Study:

- vocabulary
- token ID
- encoding
- decoding
- out-of-vocabulary characters
- why token IDs are not embedding vectors

The pipeline is:

```text
text
→ tokens
→ integer token IDs
```

Example:

```text
a → 0
b → 1
c → 2
```

Then:

```text
"cab" → [2, 0, 1]
```

## Implementation Exercise

Create:

```text
load_vocabulary()
tokenize() / encode()
de_tokenize() / decode()
reverse_vocabulary()
```

Use a fixed JSON vocabulary.

For the first version, it is fine to raise an error when a character is not in the vocabulary.

## Tests

The key test:

```text
decode(encode(text)) == text
```

for every string containing only valid vocabulary characters.

## Done When

You can explain:

- why a model uses token IDs instead of raw characters
- why token IDs are just indices
- why adding an unseen token after training is not trivial
- the difference between token IDs and embedding vectors

---

# Day 3 — Training Sequences and Batches

## Goal

Turn the long token sequence into model-ready `X` and `Y` batches.

## Concepts

Study:

- context length `T`
- batch size `B`
- train/validation/test splits
- shifted targets
- random batch sampling

Suppose:

```text
data = [7, 4, 11, 11, 14]
T = 3
```

One sample could be:

```text
X = [7, 4, 11]
Y = [4, 11, 11]
```

`Y` is the original sequence shifted one position forward.

## Why?

At every position:

```text
X[t] → predict Y[t]
```

The task is always next-token prediction.

## Implementation Exercise

Create:

```text
split_dataset(...)
x_y_batches(...)
```

A batch should produce:

```text
X.shape = (B, T)
Y.shape = (B, T)
```

Randomly select valid starting positions.

Do not wrap around at the end of the dataset.

## Dataset Split Exercise

Support:

```text
train
test
validation
```

Use cumulative slice boundaries.

Example:

```text
80% train
10% test
10% validation
```

means:

```text
[0%   : 80%]   train
[80%  : 90%]   test
[90%  : 100%]  validation
```

## Tests

For every sampled row:

```text
Y is X shifted one position forward in the original dataset
```

Also verify:

```text
X.shape == (B, T)
Y.shape == (B, T)
X.dtype is integer
Y.dtype is integer
```

## Done When

You can explain:

- why `Y` is shifted
- what `B` means
- what `T` means
- why batches can overlap
- why train and validation data must be separate

---

# Day 4 — Embeddings, Logits, Softmax, and Cross-Entropy

## Goal

Build the forward pass of the first neural next-token model.

## Embeddings

Create an embedding table:

```text
shape = (V, C)
```

Each token ID selects one row.

Example:

```text
token ID 7
→ embedding_table[7]
→ vector of length C
```

Embedding lookup transforms:

```text
(B, T)
→
(B, T, C)
```

## Output Projection

Create output weights:

```text
shape = (C, V)
```

Matrix multiplication gives:

```text
(B, T, C) @ (C, V)
→
(B, T, V)
```

The result is the logits tensor.

## Logits

For one prediction position:

```text
logits[b, t]
```

has shape:

```text
(V,)
```

It contains one raw score for every possible next token.

Logits are not probabilities.

## Softmax

Study:

`p_i = exp(z_i) / sum(exp(z_j))`

Softmax converts logits into a probability distribution.

Verify:

```text
sum(softmax(logits[b, t])) ≈ 1
```

## Cross-Entropy

For one prediction:

`loss = -ln(probability assigned to the correct token)`

High correct-token probability means low loss.

Low correct-token probability means high loss.

PyTorch cross-entropy expects raw logits, not manually softmaxed probabilities.

## Flattening

Cross-entropy can treat all prediction positions as one large list:

```text
logits:
(B, T, V)
→
(B*T, V)

Y:
(B, T)
→
(B*T)
```

No values are recomputed. The tensor is only reshaped.

## Random-Loss Sanity Check

A random model is roughly uniform:

```text
P(correct) ≈ 1 / V
```

Therefore:

```text
loss ≈ ln(V)
```

Use this as a sanity check.

## Done When

You can explain every transformation:

```text
(B,T)
→ (B,T,C)
→ (B,T,V)
→ scalar loss
```

---

# Day 5 — Backpropagation and Training

## Goal

Make the model parameters learn.

## Concepts

Study:

- `torch.nn.Module`
- `torch.nn.Parameter`
- computation graphs
- autograd
- gradients
- SGD
- learning rate
- `zero_grad()`
- training loss
- validation loss
- overfitting

## Model Parameters

For this tiny model, the trainable parameters are:

```text
embedding table (V, C)
output weights  (C, V)
```

Batch data, logits, and loss are not persistent model parameters.

## Backward Pass

Calling:

```text
loss.backward()
```

computes gradients.

It does not update parameters.

Check:

```text
embedding_table.grad.shape == (V, C)
weights.grad.shape == (C, V)
```

## SGD

Understand:

`parameter = parameter - learning_rate * gradient`

In PyTorch:

```text
optimizer.step()
```

updates the parameters.

## Why `zero_grad()`?

PyTorch accumulates gradients.

A normal training step is:

```text
optimizer.zero_grad()
forward
loss.backward()
optimizer.step()
```

## Training Loop Exercise

Repeat for a fixed number of steps:

```text
sample batch
→ zero gradients
→ forward
→ loss
→ backward
→ optimizer step
```

Create the model and optimizer once, outside the training loop.

## Validation Exercise

Every fixed number of training steps:

1. sample validation batches
2. compute validation loss
3. do not call `backward()`
4. do not call `optimizer.step()`
5. use `torch.no_grad()`

Compare:

```text
training loss
validation loss
```

## Overfitting

Typical healthy pattern:

```text
training loss ↓
validation loss ↓
```

Possible overfitting:

```text
training loss ↓
validation loss ↑
```

## Done When

You can explain:

- what a parameter is
- what a gradient means
- what `backward()` does
- what the optimizer does
- why gradients must be cleared
- why validation data is not used for updates

---

# Day 6 — Autoregressive Generation

## Goal

Use the trained model to generate text one token at a time.

## Generation

Training:

```text
X → model → logits → loss → backward
```

Generation:

```text
current token
→ model
→ logits
→ choose next token
→ append
→ repeat
```

## Generation Shape

For this simple model, use:

```text
B = 1
T = 1
```

so:

```text
X.shape = (1, 1)
```

The model returns:

```text
logits.shape = (1, 1, V)
```

The next-token scores are:

```text
logits[0, -1]
```

which has shape:

```text
(V,)
```

Here:

- `0` selects the first sequence
- `-1` selects the final token position
- the `V` dimension remains

## Greedy Decoding

Use:

```text
argmax(logits)
```

This always chooses the highest-scoring token.

Expect deterministic and potentially repetitive generation.

## Sampling

Use:

```text
logits
→ softmax
→ probabilities
→ multinomial sampling
```

Sampling produces more varied output.

## Experiment

Generate 200 characters using:

1. greedy decoding
2. multinomial sampling

Compare the results.

With the current architecture, greedy decoding may fall into loops such as:

```text
the the the the ...
```

## Why Is the Generated Text Bad?

The current model processes every token independently.

It effectively learns approximately:

`P(next token | current token)`

It cannot use long context yet.

That is why it can learn:

```text
t → h
h → e
```

while still producing globally incoherent text.

## Done When

You can explain:

- autoregressive generation
- why generation does not need `Y`
- `argmax`
- multinomial sampling
- why softmax is needed for sampling
- why greedy decoding repeats
- why the current model cannot maintain long-range coherence

---

# Day 7 — Rebuild, Review, and Debug

## Goal

Reconstruct Week 1 without blindly copying previous code.

## Rebuild Exercise

From a blank file, rebuild:

```text
vocabulary loading
tokenization
dataset splitting
batch creation
embedding table
tiny model
loss calculation
training loop
validation
generation
```

Documentation is allowed. Copying your previous implementation line-by-line is not.

## Shape Test

Be able to write these from memory:

```text
X                  (B, T)
Y                  (B, T)
embedding table    (V, C)
embedded X         (B, T, C)
weights            (C, V)
logits             (B, T, V)
flattened logits   (B*T, V)
flattened Y        (B*T)
loss               scalar
```

## Debugging Exercise

Intentionally introduce shape errors.

Example:

Change output weights from:

```text
(C, V)
```

to:

```text
(V, C)
```

Before running the program, predict why:

```text
(B,T,C) @ (V,C)
```

cannot perform the intended matrix multiplication.

Then read and interpret PyTorch's error message.

## Final Experiment

Record:

```text
initial training loss
final training loss
initial validation loss
final validation loss
```

Then generate text with:

```text
greedy decoding
sampling
```

Explain the difference.

---

# Final Week 1 Questions

You should be able to answer these without looking at your implementation.

1. What is a language model predicting?
2. What does `P(x_(t+1) | x_t)` mean?
3. Why can raw text provide its own training labels?
4. What is a vocabulary?
5. What is the difference between a token ID and an embedding?
6. Why is `Y` shifted one token ahead of `X`?
7. What does `B` mean?
8. What does `T` mean?
9. Why is the embedding table `(V, C)`?
10. Why does embedding lookup produce `(B, T, C)`?
11. Why does the output layer need `V` values?
12. What is a logit?
13. What does softmax do?
14. What does cross-entropy measure?
15. Why is random loss approximately `ln(V)`?
16. What does `loss.backward()` compute?
17. What is stored in `.grad`?
18. What does `optimizer.step()` change?
19. Why call `zero_grad()`?
20. Why keep validation data separate?
21. What does overfitting look like?
22. What does autoregressive generation mean?
23. Why does greedy decoding become repetitive?
24. Why is sampling more varied?
25. Why is the current model unable to use long context?

---

# Final Week 1 Deliverable

A working character-level language-model project that can:

```text
load text
→ encode text
→ create train/validation data
→ create X/Y batches
→ embed tokens
→ produce logits
→ calculate cross-entropy
→ backpropagate gradients
→ update parameters
→ evaluate validation loss
→ generate new text
```

The generated text does not need to be good.

The real success criterion is that you can explain why every major step exists and why every important tensor has its shape.

---

# What Comes Next

Week 1 intentionally ends with a model that only understands very local token relationships.

The next major problems are:

- better tokenization
- using more than one previous token as real context
- attention
- transformer blocks

Do not move on until the Week 1 pipeline feels understandable rather than magical.
