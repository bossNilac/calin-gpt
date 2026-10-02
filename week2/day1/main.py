def text_to_bytes(text):
    return text.encode('utf-8')


def bytes_to_token_ids(data):
    output = []

    for byte in data:
        output.append(byte)

    return output


def token_ids_to_bytes(token_ids_):
    return bytes(token_ids_)


def bytes_to_text(data):
    return data.decode('utf-8')


def encode(text):
    return bytes_to_token_ids(text_to_bytes(text))


def decode(token_ids):
    return bytes_to_text(token_ids_to_bytes(token_ids))



test_strings = [
    "",
    "hello",
    "Hello, world!",
    "café",
    "naïve",
    "你好",
    "こんにちは",
    "🙂",
    "🚀🔥",
    "Aé🙂",
    "hello\nworld",
    "\t spaces   ",
    "1234567890",
]

if __name__ == "__main__":
    for text in test_strings:
        encoded = encode(text)
        decoded = decode(encoded)

        # Main round-trip invariant
        assert decoded == text, (
            f"Round-trip failed!\n"
            f"Original: {repr(text)}\n"
            f"Decoded:  {repr(decoded)}"
        )

        # At this stage every token is exactly one byte
        assert all(0 <= token_id < 256 for token_id in encoded)

        # Number of tokens should equal number of UTF-8 bytes
        assert len(encoded) == len(text.encode("utf-8"))

        print(
            repr(text),
            "chars:", len(text),
            "bytes/tokens:", len(encoded),
            "ids:", encoded
        )

    print("\nAll Day 1 tests passed.")