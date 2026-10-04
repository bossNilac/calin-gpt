import json

from week2.day4.main import encode_bpe, vocab, merges


def decode_bpe(token_ids, vocab):
    byte_data = b""

    for token_id in token_ids:
        byte_data += vocab[token_id]

    return byte_data.decode("utf-8")


def save_tokenizer(path, vocab, merges):
    data = {
        "vocab": {
            str(token_id): list(byte_value)
            for token_id, byte_value in vocab.items()
        },
        "merges": [
            {
                "left": pair[0],
                "right": pair[1],
                "new_token": new_token
            }
            for pair, new_token in merges.items()
        ]
    }

    with open(path, "w", encoding="utf-8") as file:
        json.dump(data, file)


def load_tokenizer(path):
    with open(path, "r", encoding="utf-8") as file:
        data = json.load(file)

    vocab = {}

    for token_id, byte_value in data["vocab"].items():
        vocab[int(token_id)] = bytes(byte_value)

    merges = {}

    for merge in data["merges"]:
        pair = (merge["left"], merge["right"])
        merges[pair] = merge["new_token"]

    return vocab, merges

if __name__ == "__main__":
    tests = [
        "",
        "hello",
        "Hello World!",
        "café",
        "你好",
        "🙂",
        "hello\n\tworld",
    ]

    for text in tests:
        token_ids = encode_bpe(text, merges)
        decoded = decode_bpe(token_ids, vocab)

        assert decoded == text

    save_tokenizer("tokenizer.json", vocab, merges)

    loaded_vocab, loaded_merges = load_tokenizer("tokenizer.json")

    assert loaded_vocab == vocab
    assert loaded_merges == merges

    for text in tests:
        original_tokens = encode_bpe(text, merges)
        loaded_tokens = encode_bpe(text, loaded_merges)

        assert original_tokens == loaded_tokens
        assert decode_bpe(loaded_tokens, loaded_vocab) == text

    print("All Day 5 tests passed.")