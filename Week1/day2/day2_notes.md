## Things I should now be able to explain
- Vocabulary = fixed mapping between tokens and integer IDs.
- Tokenization here means characters → integer IDs, not vectors.
- Decoding reverses that mapping.
- \(V=26\) for your current vocabulary.
- Unknown characters cannot simply receive arbitrary new IDs after training.