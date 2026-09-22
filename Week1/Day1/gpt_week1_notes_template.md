# GPT From Scratch — Week 1 Notes

## Day 1

### Goal

- design a simple token predictor by calculating the probabilties of 2 chars being next to each other.

### Concepts

- next character text generation

### Key Math

- simple probability theory


### What I Implemented

- 

### What I Observed

- when letting this psedo-model run by itself it incline to the word the because that is the only sequence in the sample test that has a very high probability.

### Things I Should Now Be Able to Explain

- Probability theory behinf the "ai's" decision'

### Things I Am Still Shaky On

- None

### Day Summary
In my own words:

> Simple letter by letter text predictor based on sample text.

## Useful Formulas

### Conditional Probability

P(a | b) probability of a happening given b happened

### Next-Token Prediction

Based on P (a | b) which is normalised to 1 by deviding the count of the pair (b,a) by all the pairs who start with b

---


## Ideas for Later
- None today
