import time

from Week1.Day3.main import split_data
from week2.day3.main import init_vocab, extend_vocab
from week2.day4.main import encode_bpe


def load_corpus(path):
    with open(path, "r", encoding="utf-8") as file:
        return file.read()

def compression_ratio(text, token_ids):
    byte_count = len(text.encode("utf-8"))

    if len(token_ids) == 0:
        return 0

    return byte_count / len(token_ids)

def tokens_per_character(text, token_ids):
    if len(text) == 0:
        return 0

    return len(token_ids) / len(text)

def tokens_per_byte(text, token_ids):
    byte_count = len(text.encode("utf-8"))

    if byte_count == 0:
        return 0

    return len(token_ids) / byte_count

def tokens_per_word(text, token_ids):
    words = text.split()

    if len(words) == 0:
        return 0

    return len(token_ids) / len(words)

def evaluate(text, merges):
    tokens = encode_bpe(text, merges)

    return {
        "characters": len(text),
        "bytes": len(text.encode("utf-8")),
        "tokens": len(tokens),
        "compression_ratio": compression_ratio(text, tokens),
        "tokens_per_character": tokens_per_character(text, tokens),
        "tokens_per_byte": tokens_per_byte(text, tokens),
        "tokens_per_word": tokens_per_word(text, tokens),
    }

def print_evaluation(vocab_size, training_time, results):
    print("\n" + "=" * 60)
    print("VOCAB SIZE:", vocab_size)
    print("=" * 60)

    print("Training time:", round(training_time, 3), "seconds")
    print("Characters:", results["characters"])
    print("UTF-8 bytes:", results["bytes"])
    print("BPE tokens:", results["tokens"])

    print(
        "Compression ratio:",
        round(results["compression_ratio"], 3),
        "x"
    )

    print(
        "Tokens / character:",
        round(results["tokens_per_character"], 4)
    )

    print(
        "Tokens / byte:",
        round(results["tokens_per_byte"], 4)
    )

    print(
        "Tokens / word:",
        round(results["tokens_per_word"], 4)
    )

def inspect_vocab(vocab, start=256, amount=30):
    print("\nLearned tokens:")

    end = min(start + amount, len(vocab))

    for token_id in range(start, end):
        byte_value = vocab[token_id]

        try:
            visible = byte_value.decode("utf-8")
        except UnicodeDecodeError:
            visible = "<not valid UTF-8 alone>"

        print(
            token_id,
            "->",
            repr(byte_value),
            "->",
            repr(visible)
        )

def stress_test(merges):
    tests = [
        "Hello World!",
        "HELLO WORLD!",
        "hello     world",
        "12345678901234567890",
        "!!!???...,,,;;;",
        "café naïve façade",
        "你好世界",
        "こんにちは世界",
        "🙂🙂🙂🚀🔥",
        "hello\n\n\nworld",
        "\t\tindented\ttext",
        "Python_BPE_tokenizer_v2.0",
        "aA1!é你🙂",
    ]

    print("\n" + "=" * 60)
    print("STRESS TEST")
    print("=" * 60)

    for text in tests:
        tokens = encode_bpe(text, merges)

        print("\nText:", repr(text))
        print("Bytes:", len(text.encode("utf-8")))
        print("Tokens:", len(tokens))
        print(
            "Compression:",
            round(compression_ratio(text, tokens), 3),
            "x"
        )


if __name__ == "__main__":

    corpus = load_corpus("bpe_benchmark_corpus.txt")

    train_text,_, validation_text = split_data(0.9,0,0.1,
        corpus
    )

    print("Training characters:", len(train_text))
    print("Validation characters:", len(validation_text))

    vocab_sizes = [
        256,
        300,
        500,
        1000,
    ]

    for vocab_size in vocab_sizes:

        vocab = init_vocab()

        start_time = time.perf_counter()

        final_tokens, merges = extend_vocab(
            train_text,
            vocab_size,
            vocab
        )

        end_time = time.perf_counter()

        training_time = end_time - start_time

        results = evaluate(
            validation_text,
            merges
        )

        print_evaluation(
            vocab_size,
            training_time,
            results
        )

        inspect_vocab(
            vocab,
            start=256,
            amount=20
        )

        stress_test(merges)