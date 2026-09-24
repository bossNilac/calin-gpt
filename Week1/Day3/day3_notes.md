## Things I should now be able to explain
- \(B\): number of sequences sampled together.
- \(T\): number of tokens per sequence.
- Why X and Y?
X is the input the model sees. Y is the correct next-token answer.

- Why is Y shifted by one?
Because the task is next-token prediction.

- Why context length T?
To limit how many consecutive tokens are used in one training example.

- Why batch size B?
To train on several examples at once before updating the model.

- Why random starting positions?
To expose the model to different parts of the dataset.

- Why train/validation/test split?
Train changes the model. Validation/test check performance on unseen data.

- Why shape (B, T)?
B sequences, each containing T token positions.

- Why integer tensors?
Because token IDs are discrete indices, not continuous values.

- Why can batches overlap?
Because each sampled chunk is just another valid training example; overlap does not make it invalid.

- Why not wrap around at the end of the text?
Because the final token is not followed by the first token in the real dataset. Wrapping would invent a false transition.