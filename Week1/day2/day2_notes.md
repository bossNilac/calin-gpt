# Week 1, Day 2: Vocabulary Notes

My notes on [the character tokenizer](main.py), before I learned BPE.
See the [week-one curriculum](../week1_curriculum_exercises.md).

## Things I should now be able to explain
- Vocabulary = fixed mapping between tokens and integer IDs.
- Tokenization here means characters → integer IDs, not vectors.
- Decoding reverses that mapping.
- \(V=26\) for the vocabulary in this exercise.
- Unknown characters cannot simply receive arbitrary new IDs after training.
