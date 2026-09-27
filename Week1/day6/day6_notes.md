# Day 6 Notes — Autoregressive Generation

## 1. What generation does

Training and generation use the same model, but differently.

During training:

```text
X
→ model
→ logits
→ compare with known Y
→ loss
→ backward
→ optimizer update
```

During generation:

```text
current token(s)
→ model
→ logits
→ choose next token
→ append it
→ feed it back into the model
→ repeat
```

There is no target `Y`, no loss, no `backward()`, and no optimizer step during generation.

---

## 2. Autoregressive generation

Autoregressive generation means generating one token at a time.

```text
start token
↓
predict next token
↓
append prediction
↓
use prediction as new input
↓
predict again
↓
repeat
```

The model repeatedly uses its own previous predictions to continue the sequence.

---

## 3. Generation input shape

For the current model, generation can use:

```text
B = 1
T = 1
```

So:

```text
X.shape = (1, 1)
```

`B = 1` because only one sequence is being generated.

`T = 1` is enough because the current model only uses the current token and does not combine information across previous positions.

Later, with attention, `T` will represent actual usable context.

---

## 4. Logits during generation

The model returns:

```text
logits.shape = (B, T, V)
```

For generation with `B = 1`:

```text
logits.shape = (1, T, V)
```

The next-token prediction comes from the final position:

```python
next_logits = logits[0, -1]
```

This has shape:

```text
(V,)
```

It contains one raw score for every possible next token.

---

## 5. Greedy decoding

Greedy decoding chooses the token with the highest logit.

```text
logits = [0.2, 1.7, -0.4, 0.9]

argmax(logits) = 1
```

So token ID `1` is selected.

Greedy decoding is deterministic.

In the experiment, greedy generation produced repeated text like:

```text
the the the the ...
```

because the model learned strong transitions such as:

```text
t -> h
h -> e
e -> space
space -> t
```

Greedy decoding always follows the strongest path, so it can fall into loops.

---

## 6. Sampling

Sampling uses the whole probability distribution instead of always choosing the largest value.

First:

```text
logits
→ softmax
→ probabilities
```

Then:

```python
torch.multinomial(probs, 1)
```

samples one token according to those probabilities.

Example:

```text
a: 0.10
b: 0.60
c: 0.20
d: 0.10
```

`b` is most likely, but the other tokens can still be selected.

Sampling therefore produces more varied output than greedy decoding.

---

## 7. Why softmax is needed before multinomial

Logits are raw scores.

They are not probabilities and do not necessarily:

- lie between 0 and 1
- sum to 1

Softmax converts them into a probability distribution.

```text
(V,) logits
→ softmax
→ (V,) probabilities
→ multinomial sampling
→ one next-token ID
```

---

## 8. Why the generated text looks partly like English

The generated text contained patterns such as:

```text
the
th
he
spaces
punctuation
newlines
```

This means the model learned useful local character-transition statistics.

It learned that some characters are much more likely to follow others.

For example:

```text
t -> h
h -> e
```

can become strong learned transitions.

---

## 9. Why the generated text is still nonsense

The current model effectively learns something close to:

```text
P(next token | current token)
```

It does not use the whole previous sequence.

If the current character is `h`, the model does not know whether the previous context was:

```text
th
sh
wh
ch
```

It only knows that the current token is `h`.

Because of this, the model can produce locally plausible character combinations without understanding longer words, phrases, or sentence structure.

This is the main limitation of the current architecture.

---

## 10. Why attention will matter later

The current model processes each token position independently.

Attention will allow a token to use information from previous positions.

Then generation can depend on something closer to:

```text
P(next token | many previous tokens)
```

instead of only:

```text
P(next token | current token)
```

That is what will allow longer-range coherence.

---

# Full Generation Flow

```text
starting character
↓
tokenize
↓
token ID
↓
reshape to (1, 1)
↓
model
↓
logits: (1, 1, V)
↓
take logits[0, -1]
↓
(V,) logits
↓
greedy:
    argmax

or sampling:
    softmax
    multinomial
↓
next token ID
↓
append to output
↓
reshape as next input
↓
repeat
↓
decode token IDs back into text
```

---

# What I Should Be Able to Explain

- What autoregressive generation means.
- Why generation happens one token at a time.
- Why generation does not need `Y`.
- Why generation does not use loss, backpropagation, or an optimizer.
- Why the model still outputs logits during generation.
- Why `logits[0, -1]` represents the next-token prediction.
- Why one prediction has shape `(V,)`.
- What greedy decoding does.
- What `argmax` returns.
- Why greedy decoding can become repetitive.
- What sampling does.
- Why softmax is applied before multinomial sampling.
- Why multinomial produces more varied output.
- Why the current model generates locally plausible but globally incoherent text.
- Why the current model only uses roughly one-token context.
- Why attention will eventually make `T` matter as real context length.
