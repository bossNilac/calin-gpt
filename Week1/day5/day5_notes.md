# Day 5 Notes — Backpropagation, Gradients, Optimization, Validation

## 1. What the model learns

The trainable parameters in the current model are:

- embedding table: `(V, C)`
- output weight matrix: `(C, V)`

These are the values changed during training.

Everything else, such as `X`, `Y`, embeddings for the current batch, logits, and loss, is temporary computation.

---

## 2. Forward pass

The forward pass uses the current parameters to compute predictions and loss.

```text
X
→ embedding lookup
→ embedded batch
→ logits
→ cross-entropy loss
```

The forward pass does not change the parameters.

---

## 3. Gradients

A gradient tells how the loss would change if a parameter changed slightly.

For a parameter `θ`:

`∇θ L`

means the gradient of the loss with respect to that parameter.

In the current model:

```text
embedding_table.grad  -> shape (V, C)
weight.grad           -> shape (C, V)
```

The gradient shape matches the parameter shape because every parameter value needs its own gradient.

---

## 4. `loss.backward()`

`loss.backward()` uses PyTorch autograd to move backward through the computation graph.

```text
embedding table ──→ embeddings ──┐
                                 ├──→ logits ──→ loss
output weights ──────────────────┘
```

Calling:

```python
loss.backward()
```

computes gradients and stores them in `.grad`.

It does NOT update the parameters.

---

## 5. Optimizer

The optimizer uses the gradients to change the parameters.

For basic SGD:

`θ ← θ - η ∇θ L`

where:

- `θ` = parameter
- `η` = learning rate
- `L` = loss

In PyTorch, `optimizer.step()` performs the parameter update.

---

## 6. Why `zero_grad()` is needed

PyTorch accumulates gradients by default.

Without clearing them:

```text
new gradient + old gradient
```

would be stored in `.grad`.

For ordinary training:

```text
optimizer.zero_grad()
forward
loss.backward()
optimizer.step()
```

---

## 7. Basic training loop

```text
sample X, Y batch
↓
optimizer.zero_grad()
↓
forward pass
↓
calculate loss
↓
loss.backward()
↓
optimizer.step()
↓
repeat
```

The same model and optimizer must persist across training steps.

Do not recreate the model inside the training loop, because the learned parameters need to carry over from one step to the next.

---

## 8. Batch size `B` and sequence length `T`

For the current model:

```text
X.shape = (B, T)
Y.shape = (B, T)
```

There are about `B * T` next-token predictions contributing to one optimizer update.

Example:

```text
B = 3
T = 10
```

gives 30 prediction positions per update.

Larger `B` usually gives a more stable gradient because it samples more independent chunks.

Larger `T` gives more consecutive prediction positions from each sampled chunk.

In the current model, positions do not communicate, so increasing `T` mostly gives more training examples per batch.

Later, with attention, `T` will also determine how much previous context a token can use.

---

## 9. Training loss

Training loss measures how well the model predicts next tokens on batches drawn from the training set.

A decreasing training loss means the learned parameters are becoming better at assigning probability to the correct next tokens.

The loss is not a mismatch between the weights themselves.

For one prediction:

`L = -ln(p_correct)`

Higher probability on the correct token means lower loss.

---

## 10. Validation loss

Validation data is held out from parameter updates.

Validation is used only to measure how well the model performs on unseen data.

During validation:

```text
validation batch
→ forward pass
→ validation loss
```

Do not call `loss.backward()` or `optimizer.step()`.

Use:

```python
with torch.no_grad():
```

because gradients are not needed.

---

## 11. Training loss vs validation loss

A healthy pattern is:

```text
training loss ↓
validation loss ↓
```

A possible overfitting pattern is:

```text
training loss ↓
validation loss ↑
```

Loss values fluctuate because batches are randomly sampled.

The overall trend matters more than one individual value.

---

## 12. What happened in this experiment

Training loss decreased roughly from:

```text
3.65 → 2.05
```

Validation loss also decreased roughly from:

```text
3.63 → 2.03
```

This means the learned parameters improved next-token predictions on both training and held-out text.

The two losses staying fairly close suggests no obvious overfitting in this short experiment.

---

# Important limitation of the current model

The current model processes every token independently:

```text
token ID
→ embedding
→ logits
```

It does not yet combine information from previous positions.

So it effectively learns something close to:

`P(x_(t+1) | x_t)`

Attention later will allow positions to communicate and use longer context.

---

# What I Should Be Able to Explain

- What a trainable parameter is.
- Why the embedding table and output weight matrix are parameters.
- What a forward pass does.
- What a gradient means.
- What `loss.backward()` does.
- Why `loss.backward()` does not update parameters.
- What `.grad` contains.
- What `optimizer.step()` does.
- Why `optimizer.zero_grad()` is necessary.
- The basic SGD update rule.
- Why the same model must persist across training steps.
- How `B` and `T` affect the amount of training data used per update.
- What training loss measures.
- What validation loss measures.
- Why validation does not use backpropagation.
- What overfitting looks like in training and validation loss.
- Why random batch losses fluctuate.
- Why the current model still only uses roughly one-token context.
