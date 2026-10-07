import torch

from Week1.day7.main import split_dataset, x_y_batches
from week2.day4.main import encode_bpe
from week2.day5.main import decode_bpe, load_tokenizer


class Day7Model(torch.nn.Module):
    def __init__(self, vocab_size, embedding_size):
        super().__init__()

        self.embedding_table = torch.nn.Parameter(
            torch.randn(vocab_size, embedding_size)
        )

        self.weights = torch.nn.Parameter(
            torch.randn(embedding_size, vocab_size)
        )

    def forward(self, x, y=None):
        embedded = self.embedding_table[x]
        logits = embedded @ self.weights

        if y is None:
            return logits

        b, t, v = logits.shape
        logits_flat = logits.reshape(b * t, v)
        y_flat = y.reshape(b * t)

        loss = torch.nn.functional.cross_entropy(
            logits_flat,
            y_flat
        )

        return logits, loss


def generate(model, tokens, context_length, amount):
    for _ in range(amount):
        x = tokens[:, -context_length:]

        logits = model(x)
        logits = logits[:, -1, :]

        probabilities = torch.softmax(logits, dim=-1)
        next_token = torch.multinomial(probabilities, 1)

        tokens = torch.cat((tokens, next_token), dim=1)

    return tokens


if __name__ == "__main__":
    text = open("text.txt", "r", encoding="utf-8").read()

    vocab, merges = load_tokenizer("tokenizer.json")

    token_ids = encode_bpe(text, merges)

    train_data,_,validation_data = split_dataset(token_ids,0.8,0,0.2)

    vocab_size = len(vocab)
    embedding_size = 32
    context_length = 8
    batch_size = 32

    model = Day7Model(vocab_size, embedding_size)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)

    for step in range(1000):
        x, y = x_y_batches(
            train_data,
            batch_size,
            context_length
        )

        logits, loss = model(x, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 100 == 0:
            print(step, loss.item())

    start = encode_bpe("The", merges)
    start = torch.tensor([start], dtype=torch.long)

    generated = generate(
        model,
        start,
        context_length,
        100
    )

    generated_ids = generated[0].tolist()

    print(decode_bpe(generated_ids, vocab))