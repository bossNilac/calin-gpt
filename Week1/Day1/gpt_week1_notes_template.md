# Week 1, Day 1: Character Transition Notes

My notes for [Day 1](main.py). This was the first predictor, before the
[current GPT](../../README.md).

## Goal and Implementation

Build a letter-by-letter predictor by counting adjacent character pairs in the
local `sample` file and normalizing counts into conditional probabilities.

## Observation

In my local experiment, prediction tended toward repeating "the", following
strong transitions in the sample. This is an observation from that exercise,
not a benchmark of the later language model.

## Conditional Probability

`P(a | b)` is the probability of the next character being `a`, given current
character `b`. Estimate it by dividing the count of pair `(b, a)` by the total
count of pairs starting with `b`.

## What I Should Be Able to Explain

- Why counting pairs gives a next-character distribution.
- Why the outgoing probabilities for an observed character sum to one.
- Why greedy prediction can repeat a short cycle.
- Why a single-character transition model cannot use longer context.

## Open Questions and Later Ideas

None recorded at this stage.
